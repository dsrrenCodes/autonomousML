import operator

from pydantic import BaseModel
from typing import Literal
from typing import TypedDict, Annotated,List, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages



class AgentState(TypedDict):
    dataset_path: str
    target_column: str
    cleaned_data_summary: dict
    leaderboard: list[dict]              # AutoGluon leaderboard, ranked
    critic_findings: list[dict]          # {test, threshold, measured_value, passed}
    leaderboard_candidates_checked: list[str]
    judge_decision: dict                 # full JudgeDecision
    status: Literal["running", "retry", "accepted", "rejected", "exhausted"]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int #intialise this to 0 remember when doing invoke
    max_retries: int
    report: dict

    # --- remediation cycle -------------------------------------------------
    # Every repair the Judge has prescribed, in the order applied. This is the
    # single source of truth for "what is the data, right now": Experiment and
    # Critic both read through nodes._load_dataset(), which REPLAYS this whole
    # list onto the raw CSV on every read. So a retry lap refits and re-measures
    # data that has genuinely CHANGED — the only reason the loop converges. The
    # Critic is deterministic on a fixed frame, so with no repair applied lap N+1
    # would be bit-identical to lap N and the Judge would repeat itself until the
    # retry bound killed the run.
    #
    # Replayed rather than stored as a single "active" directive because repairs
    # must COMPOUND — drop-a-column on lap 1 and dedupe on lap 2 must both be in
    # effect on lap 3, or the loop undoes its own fix. The uploaded CSV is never
    # rewritten; dataset_path always means what the user actually gave us.
    #
    # operator.add is the reducer, so a node appends by returning a ONE-ELEMENT
    # list ({"remediation_history": [directive]}) — never the whole list, which
    # would concatenate it onto itself. It earns its place twice over:
    #   1. termination — call_judge refuses a directive already in here, since
    #      re-applying it cannot change the data (see _validate_remediation).
    #   2. evidence — it is the "what did the agent actually DO" trail that the
    #      Reporter renders and EVIDENCE.md wants (Day-0 §8).
    #
    # Holds plain dicts, not models (same dict-at-the-boundary rule as
    # critic_findings), so state stays JSON-serializable. Two entry shapes, both
    # replayed by nodes._apply_remediation:
    #   * {"drop_columns": [...], "dedupe": bool}  closed-set directive — the
    #     structured-output call_judge path (graph.py).
    #   * {"code": "df = df.drop(...)"}            a REPL cleaning step — the
    #     tool-calling path (graph_agent.py / tools.run_cleaning_code). The code
    #     is re-exec'd on replay, so the saved snippet IS the reproducible repair.
    remediation_history: Annotated[list[dict], operator.add]

    # --- per-lap evidence --------------------------------------------------
    # One entry per lap, appended by critic_node / experiment_node.
    #
    # These exist because critic_findings and leaderboard are LastValue channels:
    # lap N+1 overwrites lap N. That is correct for the nodes — they must reason
    # about the data as it is NOW — but fatal for the report, because a repair
    # that works destroys the evidence that justified it. A leakage run ends with
    # every test passing and a 0.862 leaderboard, so the finished report shows a
    # clean ACCEPT and no trace of the 0.9991 score or the 0.9949 correlation we
    # actually caught. The better the loop works, the less the report proves.
    #
    # So the nodes keep writing current truth to critic_findings/leaderboard, and
    # additionally append here; the Reporter reads these to show before-vs-after.
    # Indices line up (experiment then critic, once each per lap), and
    # remediation_history[i] is the repair prescribed AFTER lap i — so it has one
    # FEWER entry than these on a converged run. It can have the SAME count when
    # the Router refuses the last lap: call_judge appends the directive before the
    # Router ever runs, so a bound-hit run carries a trailing repair that was
    # never applied to anything. The Reporter pairs repair[i] with lap i+1 and
    # drops any trailing entry rather than claiming a repair that never happened.
    findings_history: Annotated[list[list[dict]], operator.add]
    leaderboard_history: Annotated[list[list[dict]], operator.add]







