"""
Disagreement eval set — Week 2 §Day 11.

Builds constructed cases where the rule-based verdict and a genuinely
reasoning LLM Judge SHOULD plausibly disagree, then runs them through the
real pipeline (Data -> Experiment -> Critic -> Judge) to see whether the
override path actually fires — not just that it exists in code.

Two case types, per Day-11 spec:
    1. borderline_escalate  — every individual threshold technically PASSES,
       but the combination (near-ceiling correlation + near-ceiling
       contamination + suspiciously high leaderboard score) is the kind of
       pattern a raw rule-check can't see holistically. rule_based_verdict
       should be "accept" — the interesting question is whether the Judge
       escalates to "reject" anyway.
    2. near_miss_leakage — a single feature sitting just under the 0.95
       ceiling (0.90-0.93). Technically passes, but close enough that it's
       worth checking whether the Judge treats it as evidence worth citing
       even without failing outright.

This is real data run through the real pipeline — not hand-typed mock
JudgeDecision objects — so the result is genuine evidence, not a fixture.

Owner: Person B
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/ on path

from app.agents.data_node import data_node
from app.agents.experiment_node import experiment_node
from app.agents.critic_node import critic_node
from app.agents.nodes import call_judge

np.random.seed(7)

SOURCE_PATH = Path(__file__).resolve().parents[3] / "data" / "Titanic-Dataset.csv"
OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "adversarial_suite"
OUT_DIR.mkdir(exist_ok=True)


def build_borderline_escalate_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every individual test should PASS, but combined signals are suspicious:
      - correlation ~0.90 (under 0.95 ceiling, passes)
      - ~0.8% duplicate rows (under 1% ceiling, passes)
    Individually fine. Together: a feature that's oddly predictive AND
    some contamination — the kind of combination worth a second look.
    """
    out = df.copy()
    noise = np.random.normal(0, 0.18, size=len(out))  # bigger noise -> lower correlation than the leakage case
    out["engagement_score"] = out["Survived"] + noise

    n_dupes = int(len(out) * 0.008)  # ~0.8%, just under the 1% ceiling
    dupe_rows = out.sample(n=n_dupes, random_state=7)
    out = pd.concat([out, dupe_rows], ignore_index=True)
    return out


def build_near_miss_leakage_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single feature at ~0.91-0.93 correlation — clearly passes the 0.95
    ceiling, but close enough that it's a legitimate "worth citing" signal,
    not a hard-fail.
    """
    out = df.copy()
    noise = np.random.normal(0, 0.28, size=len(out))
    out["risk_index"] = out["Survived"] + noise
    return out


def run_case(name: str, path: Path, target_column: str = "Survived") -> dict:
    """Runs the real Data -> Experiment -> Critic -> Judge chain."""
    state = {"dataset_path": str(path), "target_column": target_column}

    data_result = data_node(state)
    state.update(data_result)

    exp_result = experiment_node(state)
    state.update(exp_result)

    critic_result = critic_node(state)
    state.update(critic_result)
    state["leaderboard_candidates_checked"] = []

    judge_result = call_judge(state)
    decision = judge_result["judge_decision"]

    return {
        "name": name,
        "critic_findings": critic_result["critic_findings"],
        "top_leaderboard_score": exp_result["leaderboard"][0]["score_val"],
        "rule_based_verdict": decision["rule_based_verdict"],
        "llm_verdict": decision["verdict"],
        "overrode_rules": decision["overrode_rules"],
        "flagged_for_manual_review": decision.get("flagged_for_manual_review"),
        "justification": decision["justification"],
        "cited_evidence": decision["cited_evidence"],
    }


def main():
    df = pd.read_csv(SOURCE_PATH)

    cases = {
        "borderline_escalate": build_borderline_escalate_case(df),
        "near_miss_leakage": build_near_miss_leakage_case(df),
    }

    results = []
    for name, case_df in cases.items():
        path = OUT_DIR / f"titanic_{name}.csv"
        case_df.to_csv(path, index=False)
        print(f"Built {name} -> {path}")

        result = run_case(name, path)
        results.append(result)

        print(f"\n=== {name} ===")
        for f in result["critic_findings"]:
            mark = "PASS" if f["passed"] else "FAIL"
            print(f"  [{mark}] {f['test']:15s} measured={f['measured_value']}  {f['detail']}")
        print(f"  Top leaderboard score: {result['top_leaderboard_score']:.4f}")
        print(f"  rule_based_verdict: {result['rule_based_verdict']}")
        print(f"  llm_verdict:        {result['llm_verdict']}")
        print(f"  overrode_rules:     {result['overrode_rules']}")
        print(f"  flagged_for_review: {result['flagged_for_manual_review']}")
        print(f"  justification:      {result['justification']}")
        print(f"  cited_evidence:     {result['cited_evidence']}")

    print("\n--- Disagreement eval summary (for EVIDENCE.md) ---")
    disagreements = [r for r in results if r["overrode_rules"]]
    print(f"Cases run: {len(results)}")
    print(f"Cases where Judge disagreed with rule-based verdict: {len(disagreements)}")
    for r in disagreements:
        print(f'  - {r["name"]}: rule={r["rule_based_verdict"]} -> llm={r["llm_verdict"]}')


if __name__ == "__main__":
    main()