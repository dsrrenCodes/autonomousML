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
    "Upload a training CSV — the agent fits models with AutoGluon, runs the critic "
    "tests (leakage / contamination / imbalance), then gives a ship / no-ship verdict."
)

# --- session state init ---
for key, default in {
    "profile": None,
    "confirmed": False,
    "target_column": None,
    "last_file_id": None,
    "report": None,
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
def run_audit(uploaded_file, target: str) -> dict:
    """POST the CSV + confirmed target to /audit and return the report dict.

    Uses .getvalue() (not the file handle) because Streamlit reruns the whole
    script on every interaction; the bytes survive reruns, a consumed stream does
    not. The audit runs AutoGluon fits + an LLM loop, so the timeout is generous.
    """
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    data = {"target_column": target}
    resp = requests.post(f"{backend_url}/audit", files=files, data=data, timeout=600)
    resp.raise_for_status()
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
    # TODO(human): choose the banner treatment from the report fields above.
    # - Use st.success / st.error / st.warning to signal accept / reject / degraded.
    # - Name the selected_model on an accept.
    # - Surface flagged_for_manual_review and overrode_rules as st.warning callouts
    #   (these are the "a human must look at this" signals — don't let them hide).
    pass


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
        resp.raise_for_status()
        st.session_state.profile = resp.json()
        st.session_state.last_file_id = file.file_id
        st.session_state.confirmed = False  # new file -> reset confirmation
        st.session_state.report = None      # new file -> drop the stale report
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
            st.rerun()
    with col_run:
        if st.button("🚀 Run audit", type="primary"):
            with st.spinner("Auditing… fitting models + running critic tests (can take a few minutes)"):
                try:
                    st.session_state.report = run_audit(file, st.session_state.target_column)
                except Exception as e:
                    st.error(f"Audit failed: {e}")
                    st.session_state.report = None

    if st.session_state.report:
        st.divider()
        render_report(st.session_state.report)
