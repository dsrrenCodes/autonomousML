
from app.state.agent_state import AgentState


def route_after_agent(state: AgentState) -> str:
    """tools if the agent called one; else end (it declined to act)."""
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    if last is not None and getattr(last, "tool_calls", None):
        return "tools"
    return "reporter"


def route_after_tools(state: AgentState) -> str:
    """Reporter once a terminal tool set a verdict; else back to the agent."""
    return "reporter" if state.get("status") in ("accepted", "rejected") else "judge_agent"
