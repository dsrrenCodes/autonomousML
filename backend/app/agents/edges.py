from app.state.agent_state import AgentState


def route_by_status(state: AgentState) -> str:
    """
    Router — reads state only, per LangGraph's conditional-edge rule.
    Does NOT increment retry_count itself (can't write from a conditional
    edge) — see prepare_retry() below for that.

    accepted / rejected -> reporter (run ends)
    exhausted            -> reporter (run ends — either retries burned out,
                             or the Judge's structured output was
                             unrecoverable twice, per nodes.py's fallback path)
    retry, retry_count < max_retries -> retry_prep (loop back to Experiment)
    retry, retry_count >= max_retries -> reporter (treat as exhausted)
    """
    current_status = state["status"]
    current_retries_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)  # Day-0 §10: max 2, default here
                                                 # if not explicitly set in initial state

    if current_status in ("accepted", "rejected", "exhausted"):
        return "reporter"

    if current_status == "retry":
        if current_retries_count < max_retries:
            return "retry_prep"
        return "reporter"  # retry budget burned out -> treat as exhausted

    # Shouldn't normally hit this (e.g. status still "running"), but fail
    # safe to reporter rather than looping forever.
    return "reporter"


def prepare_retry(state: AgentState) -> dict:
    """
    Mechanical plumbing node — NOT one of the "4 nodes, 1 agent" per the
    terminology spec, just the write-side companion to route_by_status
    (conditional edges can only read state, this is where the mutation
    actually happens). Increments retry_count and resets status so the
    loop back to Experiment starts clean.
    """
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "status": "running",
    }