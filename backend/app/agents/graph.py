from langgraph.graph import StateGraph, END

from app.state.agent_state import AgentState
from app.agents.data_node import data_node
from app.agents.experiment_node import experiment_node
from app.agents.critic_node import critic_node
from app.agents.nodes import call_judge
from app.agents.reporter_node import reporter_node
from app.agents.edges import route_by_status, prepare_retry


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("data", data_node)
    graph.add_node("experiment", experiment_node)
    graph.add_node("critic", critic_node)
    graph.add_node("judge", call_judge)
    graph.add_node("retry_prep", prepare_retry)
    graph.add_node("reporter", reporter_node)

    graph.set_entry_point("data")

    graph.add_edge("data", "experiment")
    graph.add_edge("experiment", "critic")
    graph.add_edge("critic", "judge")

    graph.add_conditional_edges(
        "judge",
        route_by_status,
        {
            "reporter": "reporter",
            "retry_prep": "retry_prep",
        },
    )

    # retry_prep loops back to Experiment (re-run AutoGluon fit), not Data
    # (data itself hasn't changed, just re-running the experiment/judge cycle)
    graph.add_edge("retry_prep", "experiment")

    graph.add_edge("reporter", END)

    return graph.compile()


if __name__ == "__main__":
    # Smoke test — runs the full compiled graph against one real dataset,
    # confirms it executes end-to-end without needing to inspect every
    # node's output manually.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    app = build_graph()

    DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "adversarial_suite"
    test_path = DATA_DIR / "titanic_clean_1.csv"

    initial_state = {
        "dataset_path": str(test_path),
        "target_column": "Survived",
        "retry_count": 0,
        "max_retries": 2,
        "status": "running",
        "leaderboard_candidates_checked": [],
    }

    result = app.invoke(initial_state)

    print("=== Graph run complete ===")
    print(f"Final status: {result['status']}")
    print(f"Judge verdict: {result['judge_decision']['verdict']}")
    print(f"Retry count: {result['retry_count']}")
    print(f"\nTop 3 leaderboard scores: {result['leaderboard'][:3]}")
    print("\n--- Report ---")
    print(result.get("report", "(no report field)"))