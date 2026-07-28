from typing import Dict
from fastapi import FastAPI , File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from langgraph.errors import GraphRecursionError
from app.health import router
from app.agents.graph import app_graph, RECURSION_LIMIT
from app.agents.nodes import reporter_node
from app.artifacts import (
    export_accepted_model,
    model_zip_path,
    new_run_id,
    prune_old_runs,
    run_dir,
)
import io
import logging
import os
import tempfile
import pandas as pd

log = logging.getLogger(__name__)
app = FastAPI()

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Hello from autonomousml!"}

def _parse_csv_or_400(raw: bytes) -> pd.DataFrame:
    """Parse an upload, or 400. The one place both endpoints agree on 'unusable'.

    Kept as a shared helper so /profile and /audit can never drift into
    disagreeing about which uploads are acceptable — a file /profile happily
    previews must be a file /audit will at least attempt.
    """
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse the upload as CSV: {e}")
    if df.empty:
        raise HTTPException(400, "The uploaded CSV has no rows.")
    return df


# autodetect target column
def detect_target(df: pd.DataFrame) -> tuple[str, str]:
    common = {"target", "label", "y", "class", "outcome"}
    for col in df.columns:
        if col.lower() in common:
            return col, "name match"
    return df.columns[-1], "last column (convention)"


@app.post('/profile')
async def profile(file: UploadFile = File(...))-> Dict :
    """Preview an upload and guess its target column, ahead of the /audit call.

    Same failure policy as /audit: an upload we cannot parse is the caller's
    problem (400), not a 200 carrying an {"error": ...} the frontend has to
    sniff for before it can trust any other key in the body.
    """
    raw = await file.read()
    df = _parse_csv_or_400(raw)

    target,reason = detect_target(df)
    return {
    "columns": df.columns.tolist(),
    "n_rows": len(df),
    "suggested_target": target,
    "reason": reason,
    "preview": df.head(10).to_dict(orient="records"),
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
}


def _validate_upload(raw: bytes, target_column: str) -> None:
    """Reject the uploads the graph cannot audit — as 400s, before we spend 60s.

    Runs against the bytes we already hold, so a typo'd target_column costs
    milliseconds instead of a full AutoGluon fit that dies inside the graph and
    surfaces as an opaque 500. These mirror the hard stops data_node flags as
    halt_recommended; everything softer is the audit's job to report on, not
    this endpoint's job to refuse.
    """
    df = _parse_csv_or_400(raw)

    if target_column not in df.columns:
        raise HTTPException(
            400,
            f"target_column '{target_column}' is not a column in the upload. "
            f"Columns present: {df.columns.tolist()}",
        )
    if df[target_column].dropna().nunique() < 2:
        raise HTTPException(
            400,
            f"target_column '{target_column}' has fewer than 2 distinct non-null "
            "values — there is nothing to train on.",
        )


def _export_model(state: Dict, report: Dict) -> Dict:
    """Package the accepted model for download. Returns the model_artifact dict.

    Accept-only, by design: this tool exists to gate what ships, so a run it
    rejected — or one that ended with no verdict at all — must not hand back a
    loadable model out the side door. The reason is returned rather than left
    blank so the UI can say why the button is missing instead of just omitting it.

    Never raises. selected_model is free text from an LLM and predictor_path
    points at a directory a 60s fit may have half-written; neither is worth
    turning a completed audit into a 500 over. A failed export downgrades to
    available=False with the reason attached, and the report survives intact.
    """
    if report.get("verdict") != "accept":
        return {"available": False,
                "reason": f"No model is offered for a run that ended "
                          f"'{report.get('status')}' — only an accepted model "
                          f"is packaged for download."}

    run_id, selected = state.get("run_id"), report.get("selected_model")
    predictor_path = state.get("predictor_path")
    if not (run_id and selected and predictor_path):
        return {"available": False,
                "reason": "The run finished without recording a fitted predictor."}

    try:
        zip_path = export_accepted_model(predictor_path, selected, run_id)
    except Exception as e:
        log.exception("Model export failed for run %s", run_id)
        return {"available": False,
                "reason": f"Could not package '{selected}': {type(e).__name__}: {e}"}

    return {
        "available": True,
        "model": selected,
        "filename": zip_path.name,
        "size_bytes": zip_path.stat().st_size,
        "download_path": f"/model/{run_id}",
    }


def _run_audit_graph(initial_state: Dict) -> Dict:
    """Run the graph to completion and return the report. Blocking — call in a thread.

    Streams instead of invoking so we keep the last state we saw. If the agent
    loops past RECURSION_LIMIT, LangGraph raises and `invoke` would hand us
    nothing — but a run that never reached a verdict is exactly the run
    reporter_node was written to render (it coerces status to 'exhausted' and
    fires the degraded banner). So we re-run the Reporter over the last state
    and return a real report — with the true critic findings and the repairs the
    agent actually applied — rather than throwing the whole audit away.

    Keeping `last_state` also gives us predictor_path, which the report itself
    does not carry: the Reporter renders a decision record, not a filesystem map.
    """
    last_state = dict(initial_state)
    try:
        for state in app_graph.stream(
            initial_state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="values"
        ):
            last_state = state
        report = last_state["report"]
    except GraphRecursionError:
        report = reporter_node(last_state)["report"]

    report["model_artifact"] = _export_model(last_state, report)

    # Housekeeping last, and never fatal: the artifact this run just produced is
    # more important than the disk it is using. Excludes the current run so a
    # concurrent audit's in-flight directory cannot be swept out from under it.
    removed = prune_old_runs(exclude=last_state.get("run_id"))
    if removed:
        log.info("Pruned %d old run(s): %s", len(removed), ", ".join(removed))
    return report


@app.post('/audit')
async def audit(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> Dict:
    """Run the full audit graph on an uploaded CSV and return the report.

    The graph reads its data from `dataset_path` on disk (and re-reads it on every
    remediation lap), so we persist the upload to a temp file for the duration of
    the run rather than holding it in memory. The graph run is synchronous and
    slow (AutoGluon fits at 60s each + the LLM tool-loop), so it runs in a worker
    thread via run_in_threadpool — keeping FastAPI's event loop free to answer
    /health while an audit is in flight.

    Failure policy: a bad upload is a 400 (see _validate_upload), an agent that
    never decided still returns 200 with a degraded report (see _run_audit_graph),
    and anything else is our bug — a 500, with the traceback left to propagate
    into the server log rather than swallowed into the response body.
    """
    raw = await file.read()
    _validate_upload(raw, target_column)

    run_id = new_run_id()
    run_dir(run_id).mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        tmp.write(raw)
        tmp.close()

        # The initial AgentState — the entry contract for build_graph("data").
        # Seed the loop bookkeeping the tools read (retry_count/max_retries bound
        # the refits) so the graph starts clean. run_id is what lets the fitted
        # models be found again afterwards.
        initial_state = {
            "dataset_path": tmp.name,
            "target_column": target_column,
            "run_id": run_id,
            "status": "running",
            "retry_count": 0,
            "max_retries": 2,
            "remediation_history": [],
            "messages": [],
        }

        return await run_in_threadpool(_run_audit_graph, initial_state)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"The audit failed: {type(e).__name__}: {e}")
    finally:
        # Best-effort: on Windows a lingering handle makes unlink raise, and an
        # exception in `finally` would replace the report we just built. A leaked
        # temp file is the cheaper failure.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.get('/model/{run_id}')
async def download_model(run_id: str) -> FileResponse:
    """Download the accepted model from a completed run, as a zip.

    Contains a clone_for_deployment of exactly the model the Judge accepted —
    load it with `TabularPredictor.load(<unzipped dir>)`. Runs that were rejected
    or ended without a verdict have no artifact here, by design; so do runs old
    enough to have been pruned (AUDIT_KEEP_RUNS), which is a 404 rather than an
    error because it is the expected end of a run's life.
    """
    try:
        path = model_zip_path(run_id)
    except ValueError:
        # run_id is the only user-controlled part of an artifact path. A reject
        # here is the path-traversal guard doing its job, so say nothing useful.
        raise HTTPException(400, "Malformed run id.")

    if not path.exists():
        raise HTTPException(
            404,
            f"No model artifact for run '{run_id}'. Either the run was not "
            "accepted, it has been pruned, or the export failed — check "
            "model_artifact.reason in the audit report.",
        )
    return FileResponse(path, media_type="application/zip", filename=path.name)

