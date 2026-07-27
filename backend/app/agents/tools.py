

from typing import Annotated, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.agents.nodes import (
    CriticFinding,
    JudgeDecision,
    _apply_override_guardrail,
    _load_dataset,
    _rule_based_verdict,
    _run_cleaning_code,
    critic_node,
    experiment_node,
)


MAX_RETRIES = 2


def _findings_from(state: dict) -> List[CriticFinding]:
    return [CriticFinding(**f) for f in (state.get("critic_findings") or [])]


def _cited(findings: List[CriticFinding], empty: str) -> List[str]:
    """The audit trail: the detail strings of the failing tests, or `empty`."""
    return [f.detail for f in findings if not f.passed] or [empty]


def _df_preview(df, max_rows: int = 5) -> str:
    dtypes = ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items())
    return (f"shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
            f"columns: {dtypes}\n"
            f"head:\n{df.head(max_rows).to_string()}")


def _summarize_refit(findings: List[dict], leaderboard: List[dict]) -> str:
    """A compact, LLM-facing readout of the post-refit state."""
    lines = ["Refit complete. New critic findings:"]
    for f in findings:
        mark = "PASS" if f.get("passed") else "FAIL"
        lines.append(f"  - {f.get('test')}: {mark} "
                     f"(measured {f.get('measured_value')}, threshold {f.get('threshold')})")
    top = leaderboard[0] if leaderboard else {}
    lines.append(f"Top model now: {top.get('model')} (score_val {top.get('score_val')}).")
    fails = [f.get("test") for f in findings if not f.get("passed", True)]
    lines.append(f"Still failing: {', '.join(fails)}." if fails
                 else "All critic tests now pass.")
    return "\n".join(lines)


@tool
def run_cleaning_code(
    code: str,
    state: Annotated[dict, InjectedState], 
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Run Python against the live training DataFrame `df` and see the real result.

    In scope: `df` (the current data), `pd`, `np`. That is ALL — no import, no
    open(), no file or OS access, just DataFrame work. Print to inspect; reassign
    `df` to clean.

      - INSPECT (nothing is saved):  print(df['Age'].value_counts())
      - CLEAN (saved as a step):     df = df.drop(columns=['leaky_col'])

    A reassignment that changes `df` is saved and compounds with earlier steps.
    Code that errors, or that removes/breaks the target column, is refused and the
    data is left unchanged — you'll see why and can retry. Target the specific
    problems named in the critic findings. When the data looks right, call
    refit_and_recritique to re-run the checks.
    """
    target = state.get("target_column")
    df = _load_dataset(state)                       # raw CSV + every saved step so far
    new_df, out, err = _run_cleaning_code(code, df)
    out = out.rstrip()
    head = (out + "\n") if out else ""

    #if error
    if err:
        return Command(update={"messages": [ToolMessage(
            f"{head}ERROR: {err}\nThe data was NOT changed. Fix your code and try again.",
            tool_call_id=tool_call_id)]})

    #nothing has changed. do nothing. just return the state unchanged
    if new_df.equals(df):
        return Command(update={"messages": [ToolMessage(
            f"{head}--- df unchanged (inspection only) ---\n{_df_preview(new_df)}",
            tool_call_id=tool_call_id)]})

    # Runtime guards that replace the static screen: the target must survive.
    if target not in new_df.columns:
        return Command(update={"messages": [ToolMessage(
            f"{head}REJECTED: your code removed the target column '{target}'. The "
            "pipeline cannot train without it — df was NOT changed.",
            tool_call_id=tool_call_id)]})
    #cannot classify with 1 target class
    if new_df[target].dropna().nunique() < 2:
        return Command(update={"messages": [ToolMessage(
            f"{head}REJECTED: after your code the target '{target}' has fewer than 2 "
            "classes — df was NOT changed.", tool_call_id=tool_call_id)]})
    #otherwise its a real and valid change, so save it
    return Command(update={
        "remediation_history": [{"code": code}],   
        "messages": [ToolMessage(
            f"{head}--- df after your code (SAVED as a cleaning step) ---\n"
            f"{_df_preview(new_df)}\nCall refit_and_recritique when ready to re-check, "
            "or keep cleaning.", tool_call_id=tool_call_id)],
    })


@tool
def refit_and_recritique(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Refit AutoGluon on the cleaned data and re-run all three critic tests.

    Expensive (~60s). Call after cleaning to get fresh findings + leaderboard, then
    decide with accept_model / reject_run. Bounded: only a limited number of refits
    are allowed per run, so clean first, then refit — don't refit to explore.
    """
    max_r = state.get("max_retries", MAX_RETRIES)
    if state.get("retry_count", 0) >= max_r:
        return Command(update={"messages": [ToolMessage(
            f"Refit budget ({max_r}) is exhausted — no further refits. Decide now "
            "with accept_model or reject_run.", tool_call_id=tool_call_id)]})

    # Both read the cleaned frame through _load_dataset (raw + saved code steps).
    exp = experiment_node(state)
    crit = critic_node(state)
    return Command(update={
        "leaderboard": exp["leaderboard"],
        "leaderboard_history": exp["leaderboard_history"],
        # Must be forwarded, or the download would ship the PREVIOUS lap's
        # predictor — a model fit on the data before this refit cleaned it.
        "predictor_path": exp["predictor_path"],
        "critic_findings": crit["critic_findings"],
        "findings_history": crit["findings_history"],
        "cleaned_data_summary": exp["cleaned_data_summary"],
        "retry_count": state.get("retry_count", 0) + 1,
        "messages": [ToolMessage(
            _summarize_refit(crit["critic_findings"], exp["leaderboard"]),
            tool_call_id=tool_call_id)],
    })


@tool
def accept_model(
    selected_model: str,
    justification: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Promote a model to production and END the review.

    selected_model must be one of the leaderboard candidate names given to you.
    justification is mandatory — and if you are accepting despite a failed critic
    test, you MUST explain there why the failure does not block promotion (that
    de-escalation is allowed but is flagged for human sign-off).
    """
    findings = _findings_from(state)
    rule_verdict = _rule_based_verdict(findings)
    decision = JudgeDecision(
        verdict="accept",
        selected_model=selected_model,
        justification=justification,
        cited_evidence=_cited(findings, "All critic tests pass."),
        rule_based_verdict=rule_verdict,
        overrode_rules=(rule_verdict != "accept"),
    )
    # Same guardrail as call_judge: accept past a hard-fail with no justification
    # reverts to reject; with justification it is flagged for manual review.
    decision = _apply_override_guardrail(decision, rule_verdict)
    status = "accepted" if decision.verdict == "accept" else "rejected"
    return Command(update={
        "judge_decision": decision.model_dump(),
        "status": status,
        "messages": [ToolMessage(
            f"Verdict recorded: {decision.verdict}"
            + (f" ({selected_model})." if decision.verdict == "accept" else "."),
            tool_call_id=tool_call_id)],
    })


@tool
def reject_run(
    justification: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Reject the run — promote NOTHING — and END the review.

    Use when the data has a defect that no cleaning can fix (for example class
    imbalance, which drop/transform cannot repair). justification is mandatory.
    """
    findings = _findings_from(state)
    rule_verdict = _rule_based_verdict(findings)
    decision = JudgeDecision(
        verdict="reject",
        selected_model=None,
        justification=justification,
        cited_evidence=_cited(findings, "No failing critic tests; rejected on judgment."),
        rule_based_verdict=rule_verdict,
        overrode_rules=(rule_verdict != "reject"),
    )
    return Command(update={
        "judge_decision": decision.model_dump(),
        "status": "rejected",
        "messages": [ToolMessage("Verdict recorded: reject.", tool_call_id=tool_call_id)],
    })


JUDGE_TOOLS = [run_cleaning_code, refit_and_recritique, accept_model, reject_run]
TERMINAL_TOOLS = {"accept_model", "reject_run"}
