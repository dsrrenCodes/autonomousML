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
    judge_decision: dict                 # full JudgeDecision, see below — not just a verdict string
    retry_count: int
    status: Literal["running", "retry", "accepted", "rejected", "exhausted"]
    messages: Annotated[Sequence[BaseMessage], add_messages]


class JudgeInput(BaseModel):
    critic_findings: str #to be replaced
    leaderboard_candidates: List[dict]
    candidates_checked: List[str]

class JudgeDecision(BaseModel):
    verdict: Literal['accept','retry','reject'] #for control/router flow
    seleced_model: Optional[str] 
    rule_based_verdict: Literal['accept','retry','reject'] #by critic node
    overrode_rules: bool
    justification: str
    cited_evidence: List[str] #from critic



