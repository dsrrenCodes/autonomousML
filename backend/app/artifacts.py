"""Run artifacts — where a fitted predictor lives, and what the user can download.

The audit used to produce only markdown. AutoGluon really did fit models, but
`TabularPredictor(...)` was constructed with no `path=`, so it wrote them to
`./AutogluonModels/ag-<timestamp>/` relative to whatever the process cwd happened
to be, and nothing ever recorded the path. The models were real and unreachable —
a verdict of "promote NeuralNetTorch to production" with no NeuralNetTorch to hand
over.

This module gives every audit a run id and a directory under RUNS_DIR, so the
Experiment Node can pin its output somewhere findable and the accepted model can
be exported at the end. Nothing here knows about the graph; nodes and main both
import it, so it must not import them back.
"""

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from autogluon.tabular import TabularPredictor

# Everything a run produces lives under RUNS_DIR/<run_id>/. Overridable so the
# container can point it at a mounted volume (see docker-compose.yml) — without
# that, artifacts live in the container's writable layer and vanish on restart,
# which is exactly the failure this module exists to prevent.
RUNS_DIR = Path(os.getenv("AUDIT_RUNS_DIR",
                          Path(__file__).resolve().parent.parent / "runs"))

# How many completed runs to keep on disk. Each run is a full AutoGluon fit dir
# (tens of MB), so unbounded retention fills the volume in a day of demos.
KEEP_RUNS = int(os.getenv("AUDIT_KEEP_RUNS", "5"))

# A run id is the ONLY caller-supplied component of an artifact path, and it
# arrives from the URL in GET /model/{run_id}. Anchored hex-only: this is the
# path-traversal guard, not a formatting nicety. Do not loosen it.
_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def new_run_id() -> str:
    """A fresh run id. Random, not sequential — ids appear in URLs, and
    sequential ids would let anyone enumerate other people's runs."""
    return uuid.uuid4().hex[:12]


def run_dir(run_id: str) -> Path:
    """The directory for a run. Raises ValueError on anything not a run id."""
    if not _RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"not a valid run id: {run_id!r}")
    return RUNS_DIR / run_id


def model_zip_path(run_id: str) -> Path:
    """Where the downloadable artifact for a run lives (may not exist yet)."""
    return run_dir(run_id) / f"audit_model_{run_id}.zip"


def predictor_path_for_lap(run_id: Optional[str], lap: int) -> Optional[str]:
    """Where the Experiment Node should write lap N's predictor.

    Returns None when there is no run id — the smoke tests and `python -m
    app.agents.graph` call the nodes directly with hand-built states, and they
    should keep AutoGluon's default cwd behaviour rather than needing a run
    scaffold. A None path is passed straight through to TabularPredictor(path=),
    which is its documented default.
    """
    if not run_id:
        return None
    return str(run_dir(run_id) / f"predictor_lap{lap}")


def export_accepted_model(predictor_path: str, selected_model: str,
                          run_id: str) -> Path:
    """Extract the ONE accepted model as a zip the user can load elsewhere.

    clone_for_deployment strips the run down to the named model plus what it
    needs to predict — no training data, no rejected models, no fit metadata.
    That is deliberate: the artifact should contain exactly what the Judge
    accepted, so a model the audit refused cannot be quietly loaded out of the
    same download.

    Raises if selected_model is not a real model in the predictor — the Judge
    supplies that name and it is free-text from an LLM, so callers must handle
    the failure rather than assume it exists.
    """
    predictor = TabularPredictor.load(predictor_path, require_version_match=False)

    clone_dir = run_dir(run_id) / "accepted_model"
    if clone_dir.exists():
        shutil.rmtree(clone_dir, ignore_errors=True)
    predictor.clone_for_deployment(path=str(clone_dir), model=selected_model,
                                   dirs_exist_ok=True)

    zip_base = run_dir(run_id) / f"audit_model_{run_id}"
    shutil.make_archive(str(zip_base), "zip", root_dir=str(clone_dir))

    # The clone is redundant once zipped, and it is the larger of the two.
    shutil.rmtree(clone_dir, ignore_errors=True)
    return zip_base.with_suffix(".zip")


def prune_old_runs(keep: int = KEEP_RUNS, exclude: Optional[str] = None) -> List[str]:
    """Delete all but the `keep` most recent runs. Returns the ids removed.

    `exclude` is the in-flight run: a concurrent audit's directory is brand new
    but its predictor is still being written, and mtime ordering alone would
    happily delete a run that is mid-fit. Excluding it means a busy server keeps
    keep+1 runs, which is the right way to be wrong.

    Never raises — pruning is housekeeping, and a locked file on Windows must not
    take down the audit that triggered it.
    """
    if not RUNS_DIR.exists():
        return []
    runs = [p for p in RUNS_DIR.iterdir()
            if p.is_dir() and _RUN_ID_RE.match(p.name) and p.name != exclude]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    removed = []
    for old in runs[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        if not old.exists():
            removed.append(old.name)
    return removed
