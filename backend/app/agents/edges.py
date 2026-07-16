"""
The Router — decides where the graph goes after the Judge Agent.

This is NOT a node and NOT an agent (Day-0 §1). It is the routing function
handed to add_conditional_edges; don't call it an "Orchestrator Agent". It makes
no decisions of its own — it reads a decision the Judge already made and
enforces a bound on it.

A routing function returns an edge label and NOTHING else. Any state it assigns
to is thrown away, so this file cannot record that the bound was hit. The
Reporter infers it instead: reaching the Reporter with status=='retry' can only
mean this function refused another lap (see reporter_node).
"""

from app.state.agent_state import AgentState

# Day-0 §10: the retry bound is enforced HERE, in the Router, never inside a
# node. Used only when max_retries is absent from state — the invoke should
# normally set it. A default matters because `state.get('max_retries')` with no
# default returns None, and `int < None` is a TypeError that only fires on the
# first retry, i.e. never during a happy-path test run.
MAX_RETRIES = 2


def route_by_status(state: AgentState) -> str:
    """Judge -> Experiment (another remediation lap) or Judge -> Reporter (done).

    Returns 'retry' or 'report'; graph.py maps those to the real node names.
    'report' MUST map to the Reporter and never to END — a run that reports
    nothing has no evidence trail, which is the entire product.

    Total by construction: every status that isn't a within-bound 'retry' falls
    through to 'report'. That matters more than it looks. An if/elif chain with
    no else returns None for 'accepted' and 'rejected' — the two most common
    outcomes — and LangGraph cannot route None, so the happy path dies at the
    conditional edge while the retry path looks fine.

    Counting, because the `<=` looks off-by-one and isn't:
        call_judge increments retry_count BEFORE this function ever sees it, so
        retry_count means "retries requested so far, including the one being
        decided right now". Granting while retry_count <= max_retries therefore
        yields exactly max_retries extra laps:

            judge#1 retry -> retry_count=1 -> 1 <= 2  grant  (lap 1)
            judge#2 retry -> retry_count=2 -> 2 <= 2  grant  (lap 2)
            judge#3 retry -> retry_count=3 -> 3 <= 2  refuse -> report

        So max_retries=2 means the Experiment Node runs three times: the initial
        fit plus two retries. Each lap is a full AutoGluon refit (~60s), which is
        why the bound is small and why call_judge only spends it on a retry whose
        remediation actually changes the data.
    """
    if state["status"] == "retry" and state.get("retry_count", 0) <= state.get("max_retries", MAX_RETRIES):
        return "retry"      # -> experiment_node: refit on the repaired data
    return "report"         # accepted / rejected / exhausted / bound reached
