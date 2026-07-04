from typing import TypedDict, Annotated,List, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages




class AgentState(TypedDict):
    #inputs
    dataset_path: str 
    target_column : str 

    #data agent
    dataset_profile: dict

    #experiment agent
    leaderbord: List[dict]

    #crtitic agent
    critic_verdicts: List[dict] #{rule, passed, detail}

    #judge
    judge_decision: dict

    #general

    retry_count: int 
    messages: Annotated[Sequence[BaseMessage], add_messages]

