import streamlit as st
import requests
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="ML Pipeline Auditor", layout="wide")
st.title("🔍 ML Pipeline Auditor")
st.caption(
    "Upload a training CSV. The agent trains models, hunts for the data defects that "
    "inflate a leaderboard score, **repairs what it can and retrains to prove the fix**, "
    "then either hands you the model or refuses to ship it."
)

# The caption sells the outcome; this sells the method. Collapsed by default —
# a reviewer who already trusts the pipeline shouldn't have to scroll past it,
# but the "why should I believe the verdict" answer has to be one click away.
with st.expander("How the audit works"):
    st.markdown(
        """
**1 · Profile** — the backend reads your CSV and proposes a target column. You confirm
it before anything trains, because every test below is measured against that choice.

**2 · Fit** — AutoGluon trains a leaderboard of candidate models (~60s).

**3 · Critique** — three deterministic, code-based tests run on the data. No LLM
touches these, so the numbers are reproducible:

| Test | Measures | Fails when |
|---|---|---|
| **Leakage** | strongest feature↔target correlation | > 0.95 |
| **Contamination** | fraction of duplicate rows | > 1% |
| **Imbalance** | minority-class recall | < 0.50 |

**4 · Repair** — an LLM agent reads the failures and writes real pandas against your
data to fix them (typically dropping a leaking column). It sees the actual result of
every line it runs, and it cannot delete or collapse the target column. Every
accepted step is recorded and shown to you as code.

**5 · Refit and re-check** — the agent retrains on the repaired data and re-runs the
same three tests, so the verdict is measured on the data the shipped model was
actually fit on — not on the raw upload. Refits are budget-capped.

**6 · Verdict** — *accept* releases that model as a download; *reject* releases
nothing. Some defects (class imbalance) cannot be cleaned away, and the agent is
instructed to reject rather than paper over them. If it accepts past a failing test
it must justify that in writing, and the run is flagged for human sign-off.

The headline number this produces is the **score delta**: what the leaderboard claimed
before the repair versus after. On a leaking dataset that is often a fall from a
perfect 1.0000 to something honest — the inflated score is the thing being caught.
        """
    )

# --- session state init ---
for key, default in {
    "profile": None,
    "confirmed": False,
    "target_column": None,
    "last_file_id": None,
    "report": None,
    # The fetched model zip. Cached here so Streamlit's rerun-on-every-widget
    # doesn't re-download several MB each time the user clicks anything.
    "model_bytes": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- sidebar: backend health ---
with st.sidebar:
    st.header("Backend")
    st.write(f"`{backend_url}`")
    if st.button("Test backend health"):
        try:
            res = requests.get(f"{backend_url}/health", timeout=5)
            st.write(res.status_code, res.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")


# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------
def _backend_detail(resp) -> str:
    """The backend's own explanation for a non-2xx, or the raw body as a fallback."""
    try:
        return resp.json().get("detail") or resp.text
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"


def run_audit(uploaded_file, target: str) -> dict:
    """POST the CSV + confirmed target to /audit and return the report dict.

    Uses .getvalue() (not the file handle) because Streamlit reruns the whole
    script on every interaction; the bytes survive reruns, a consumed stream does
    not. The audit runs AutoGluon fits + an LLM loop, so the timeout is generous.
    """
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    data = {"target_column": target}
    resp = requests.post(f"{backend_url}/audit", files=files, data=data, timeout=600)
    if not resp.ok:
        # The backend answers a bad upload with 400 + {"detail": "..."} naming the
        # actual problem (unknown target column, single-class target). Surfacing
        # raise_for_status' generic text instead would throw that away.
        raise RuntimeError(_backend_detail(resp))
    return resp.json()


# ---------------------------------------------------------------------------
# Report rendering (hybrid: native components + the Reporter's markdown)
# ---------------------------------------------------------------------------
def render_verdict_banner(report: dict) -> None:
    """Render the headline verdict banner and any override / manual-review flags.

    This is the first thing a reviewer sees, so the visual treatment has to match
    how serious the outcome is. Relevant report keys:
        report["verdict"]                  -> "accept" | "reject" | None
        report["status"]                   -> "accepted" | "rejected" | "exhausted"
        report["selected_model"]           -> str | None  (set only on accept)
        report["flagged_for_manual_review"]-> bool  (accept past a hard-fail)
        report["overrode_rules"]           -> bool  (Judge disagreed with thresholds)
        report["rule_based_verdict"]       -> "accept" | "reject"
    """
    # Manual-review / override callouts go FIRST: they must sit above the verdict
    # so a green accept banner can never bury a "a human must sign this off" signal.
    if report.get("flagged_for_manual_review"):
        st.warning(
            "⚠️ **Flagged for manual review** — the model was accepted past a failed "
            "critic test. A human must sign this off before anything ships."
        )
    if report.get("overrode_rules"):
        st.warning(
            f"⚖️ **Judge overrode the rules.** Thresholds alone said "
            f"`{report.get('rule_based_verdict')}`; the Judge returned "
            f"`{report.get('verdict')}`. Read the justification before trusting this."
        )

    # The headline verdict, severity-matched to the outcome.
    verdict = report.get("verdict")
    if verdict == "accept":
        model = report.get("selected_model") or "a model"
        st.success(f"✅ **ACCEPTED** — promoting `{model}` to production.")
    elif verdict == "reject":
        st.error("❌ **REJECTED** — nothing was promoted.")
    else:
        # No verdict: the agent ran out of refits or declined to decide (status
        # 'exhausted'). Degraded, not clean — warn rather than pretend it passed.
        st.warning(
            f"🟠 **NO VERDICT** — the run ended without a decision "
            f"(status: `{report.get('status', 'unknown')}`). Nothing was promoted."
        )


def fetch_model_zip(download_path: str) -> bytes:
    """Pull the model artifact from the backend into this process.

    Deliberately proxied rather than linked. Inside docker-compose the backend is
    reachable at http://backend:8000 on the internal network only, so a link the
    user's browser follows would 404 — Streamlit has to fetch the bytes and hand
    them over itself. The model is a few MB, so holding it in memory is fine.
    """
    resp = requests.get(f"{backend_url}{download_path}", timeout=120)
    resp.raise_for_status()
    return resp.content


def render_model_download(report: dict) -> None:
    """Offer the accepted model for download, or explain why there isn't one.

    Only an ACCEPTED run produces an artifact — the backend refuses to package a
    model it rejected. When there's nothing to offer we say why rather than
    silently omitting the button, because an absent button is indistinguishable
    from a broken one.
    """
    artifact = report.get("model_artifact") or {}

    if not artifact.get("available"):
        reason = artifact.get("reason", "No model artifact was produced.")
        st.info(f"📦 **No model to download.** {reason}")
        return

    st.subheader("Accepted model")
    size_mb = artifact.get("size_bytes", 0) / 1_048_576
    st.caption(
        f"`{artifact.get('model')}` — {size_mb:.1f} MB. Unzip, then load with "
        "`TabularPredictor.load('<unzipped dir>')`."
    )

    # Two-step: fetch on click, then hand over. Fetching eagerly on every rerun
    # would re-download several MB each time the user touches any other widget.
    if st.session_state.get("model_bytes") is None:
        if st.button("📦 Prepare model download", type="primary"):
            with st.spinner("Fetching model from backend…"):
                try:
                    st.session_state.model_bytes = fetch_model_zip(
                        artifact["download_path"])
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not fetch the model: {e}")
    else:
        st.download_button(
            f"⬇ Download {artifact.get('filename')}",
            data=st.session_state.model_bytes,
            file_name=artifact.get("filename", "audit_model.zip"),
            mime="application/zip",
            type="primary",
        )


def render_report(report: dict) -> None:
    """Render the whole audit result: banner, metrics, tables, and raw markdown."""
    if "error" in report:
        st.error(f"Audit failed: {report['error']}")
        return

    render_verdict_banner(report)

    # Metric cards — the numbers-at-a-glance row.
    dataset = report.get("dataset", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run status", report.get("status", "—"))
    c2.metric("Selected model", report.get("selected_model") or "—")
    c3.metric("Failing tests", len(report.get("failing_tests", [])))
    c4.metric("Refit laps", report.get("retry_count", 0))

    # Critic findings — the deterministic test results.
    st.subheader("Critic findings")
    findings = report.get("critic_findings", [])
    if findings:
        df = pd.DataFrame(findings)
        df["result"] = df["passed"].map({True: "✅ PASS", False: "❌ FAIL"})
        st.dataframe(
            df[["test", "result", "measured_value", "threshold", "detail"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No critic findings — the Critic node did not run.")

    # Leaderboard — top models AutoGluon fit.
    st.subheader("Leaderboard (top 3)")
    leaderboard = report.get("leaderboard_top", [])
    if leaderboard:
        st.dataframe(pd.DataFrame(leaderboard), use_container_width=True, hide_index=True)
    else:
        st.info("No leaderboard — the Experiment node did not run.")

    # Repairs the agent applied (the cleaning steps that compounded across laps).
    repairs = report.get("remediation_history", [])
    if repairs:
        st.subheader("Repairs applied")
        for i, rem in enumerate(repairs, start=1):
            if rem.get("code"):
                st.markdown(f"**Step {i} — cleaning code:**")
                st.code(rem["code"], language="python")
            else:
                st.markdown(f"**Step {i}:** {rem}")

    render_model_download(report)

    # The full Reporter markdown + a download button for the artifact.
    with st.expander("Full audit report (markdown)"):
        st.markdown(report.get("markdown", "_(no markdown rendered)_"))
    st.download_button(
        "⬇ Download report.md",
        data=report.get("markdown", ""),
        file_name="audit_report.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Flow: upload -> profile -> confirm target -> run audit -> report
# ---------------------------------------------------------------------------
file = st.file_uploader("Upload a CSV file", type=["csv"])

# Only re-profile when a genuinely new file is uploaded.
if file is not None and file.file_id != st.session_state.last_file_id:
    try:
        resp = requests.post(f"{backend_url}/profile", files={"file": file}, timeout=30)
        if not resp.ok:
            raise RuntimeError(_backend_detail(resp))
        st.session_state.profile = resp.json()
        st.session_state.last_file_id = file.file_id
        st.session_state.confirmed = False  # new file -> reset confirmation
        st.session_state.report = None      # new file -> drop the stale report
        st.session_state.model_bytes = None # new file -> drop the stale model
    except Exception as e:
        st.error(f"Profiling failed: {e}")
        st.session_state.profile = None

profile = st.session_state.profile

# Step 1 + 2: preview + confirm the target column.
if profile and not st.session_state.confirmed:
    st.dataframe(profile["preview"], use_container_width=True)
    st.info(f"Suggested target: {profile['suggested_target']}\n\nReason: {profile['reason']}")

    choice = st.selectbox(
        "Check target column",
        options=profile["columns"],
        index=profile["columns"].index(profile["suggested_target"]),
    )

    if st.button("Confirm target"):
        st.session_state.target_column = choice
        st.session_state.confirmed = True
        st.rerun()

# Step 3 + 4: run the audit and render the report.
elif profile and st.session_state.confirmed:
    st.success(f"Target column confirmed: {st.session_state.target_column}")
    col_run, col_change = st.columns(2)
    with col_change:
        if st.button("Change target"):
            st.session_state.confirmed = False
            st.session_state.report = None
            st.session_state.model_bytes = None
            st.rerun()
    with col_run:
        if st.button("🚀 Run audit", type="primary"):
            with st.spinner("Auditing… fitting models + running critic tests (can take a few minutes)"):
                # A fresh audit means a fresh artifact: clearing this first stops
                # the previous run's model being offered next to the new report.
                st.session_state.model_bytes = None
                try:
                    st.session_state.report = run_audit(file, st.session_state.target_column)
                except Exception as e:
                    st.error(f"Audit failed: {e}")
                    st.session_state.report = None

    if st.session_state.report:
        st.divider()
        render_report(st.session_state.report)
