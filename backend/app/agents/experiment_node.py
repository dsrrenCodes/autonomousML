"""
Experiment Node — deterministic wrapper around AutoGluon. No LLM call.

Fits a TabularPredictor on the dataset, packages the leaderboard into
AgentState-compatible shape. This is a library call, not a decision —
all judgment happens downstream in the Critic Node and Judge Agent.

How AutoGluon infers problem type (documented here per Day-0 §3, verbatim
for EVIDENCE.md): it inspects the target column's values. Two unique
values -> binary classification. A small number of discrete values ->
multiclass. Many unique continuous values -> regression. Confirmed
against the Titanic 'Survived' column (0/1) -> correctly inferred as
'binary' in every run tonight (see spike.py output history).

Owner: Person B
"""

from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor


def experiment_node(state: dict) -> dict:
    """
    LangGraph node entry point.

    Input (from AgentState):  dataset_path, target_column
    Output (to AgentState):   leaderboard — list[dict], ranked best-first,
                               each entry carrying at minimum a 'model' key
                               and a 'score_val' key (Judge/Critic depend on
                               'model' specifically — see nodes.py fallback
                               path: leaderboard[0]["model"]).

    time_limit and presets are pinned per Day-0 §3 / §4 (sequential
    AutoGluon -> Ollama execution, medium_quality for spike speed).
    Do not relitigate these in isolation — they're a shared VRAM/timing
    decision, not just an Experiment Node choice.
    """
    data = TabularDataset(state["dataset_path"])
    target_column = state["target_column"]

    predictor = TabularPredictor(
        label=target_column,
        verbosity=0,  # keep node output clean when run inside the graph;
                      # spike.py's __main__ block below still prints full logs
    ).fit(
        data,
        time_limit=60,
        presets="medium_quality",
    )

    leaderboard_df = predictor.leaderboard(silent=True)
    leaderboard = leaderboard_df.to_dict(orient="records")

    return {
        "leaderboard": leaderboard,
        "cleaned_data_summary": {
            **state.get("cleaned_data_summary", {}),
            "problem_type": predictor.problem_type,
            "fit_time_seconds": round(sum(
                row.get("fit_time", 0) for row in leaderboard
            ), 2),
        },
    }


if __name__ == "__main__":
    # Smoke test against the real adversarial suite, mirrors critic_node.py's
    # __main__ block so both nodes can be sanity-checked the same way.
    DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "adversarial_suite"

    test_cases = {
        "clean_1": ("titanic_clean_1.csv", "Survived"),
        "leakage": ("titanic_leakage.csv", "Survived"),
        "duplicates": ("titanic_duplicates.csv", "Survived"),
        "imbalance": ("titanic_imbalance.csv", "Survived"),
    }

    for name, (filename, target) in test_cases.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"{name:12s} SKIPPED — file not found at {path}")
            continue
        result = experiment_node({"dataset_path": str(path), "target_column": target})
        top = result["leaderboard"][0]
        print(f"{name:12s} problem_type={result['cleaned_data_summary']['problem_type']:8s} "
              f"top_model={top['model']:20s} score_val={top['score_val']:.4f}")