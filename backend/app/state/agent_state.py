import operator

from typing import Literal
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages



class AgentState(TypedDict):
    dataset_path: str
    target_column: str
    # Identifies this audit's artifact directory (app/artifacts.py). Set by the
    # /audit endpoint; absent on direct node/CLI runs, where the Experiment Node
    # falls back to AutoGluon's default cwd behaviour.
    run_id: str
    # Where the LATEST lap's fitted predictor was written. LastValue, not append:
    # the accepted model always comes from the final fit, so an earlier lap's
    # predictor (fit on data that still had the defect) must not win.
    predictor_path: str
    cleaned_data_summary: dict
    leaderboard: list[dict]              # AutoGluon leaderboard, ranked
    critic_findings: list[dict]          # {test, threshold, measured_value, passed}
    judge_decision: dict                 # full JudgeDecision
    # accepted/rejected are set by the terminal tools; 'exhausted' is the Reporter's
    # label for a run that ended with NO verdict (the agent declined to act, or spent
    # its refit budget without deciding). There is no 'retry': the tool-calling Judge
    # loops on its own tools rather than through a Router.
    status: Literal["running", "accepted", "rejected", "exhausted"]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int #intialise this to 0 remember when doing invoke
    max_retries: int #FOR number of times to refit the model used in refit_and_recritique tool
    report: dict


    remediation_history: Annotated[list[dict], operator.add] #recipe of cleaning steps the agent has applied

    findings_history: Annotated[list[list[dict]], operator.add]
    leaderboard_history: Annotated[list[list[dict]], operator.add]







