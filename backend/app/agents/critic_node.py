"""
Critic Node — deterministic, unit-tested. No LLM call.

Runs three fixed rule-based tests against the cleaned dataset + AutoGluon
leaderboard, and returns a list of exactly 3 CriticFinding dicts (leakage,
contamination, imbalance) — matching app/agents/nodes.py's CriticFinding
schema and app/mocks/critic_findings.py's fixture shape exactly.

Thresholds (Day-0 §5 / mocks/critic_findings.py — single source of truth,
do not diverge from these without updating both places):
    leakage         max |feature-target correlation|   fails when > 0.95
    contamination   fraction of exact/near-dup rows     fails when > 0.01
    imbalance       minority-class recall               fails when < 0.50

Owner: Lokav
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

LEAKAGE_THRESHOLD = 0.95
CONTAMINATION_THRESHOLD = 0.01
IMBALANCE_THRESHOLD = 0.50


def _finding(test: str, threshold: float, measured_value: float,
             passed: bool, detail: str = "") -> dict:
    """One CriticFinding as a plain dict — matches app/agents/nodes.py exactly."""
    return {
        "test": test,
        "threshold": threshold,
        "measured_value": round(float(measured_value), 4),
        "passed": bool(passed),
        "detail": detail,
    }


def check_leakage(df: pd.DataFrame, target_column: str) -> dict:
    """
    Leakage test: max absolute correlation between any single feature and
    the target. Only numeric features are checked directly; categorical
    features are label-encoded first so they aren't silently skipped.
    """
    work_df = df.copy()

    # Encode non-numeric columns so they're included in the correlation scan,
    # not silently dropped by .corr(). This matters — a leaking categorical
    # column should still be caught.
    for col in work_df.columns:
        if work_df[col].dtype == "object":
            work_df[col] = work_df[col].astype("category").cat.codes

    if target_column not in work_df.columns:
        return _finding("leakage", LEAKAGE_THRESHOLD, 0.0, True,
                         f"Target column '{target_column}' not found — skipped.")

    correlations = work_df.corr(numeric_only=True)[target_column].drop(target_column, errors="ignore")
    correlations = correlations.abs().dropna()

    if correlations.empty:
        return _finding("leakage", LEAKAGE_THRESHOLD, 0.0, True,
                         "No numeric/encodable features to check against target.")

    worst_col = correlations.idxmax()
    worst_val = correlations.max()
    passed = worst_val <= LEAKAGE_THRESHOLD

    detail = (
        f"Column '{worst_col}' correlates {worst_val:.4f} with target "
        f"'{target_column}' — {'exceeds' if not passed else 'below'} the "
        f"{LEAKAGE_THRESHOLD} ceiling."
    )
    return _finding("leakage", LEAKAGE_THRESHOLD, worst_val, passed, detail)


def check_contamination(df: pd.DataFrame) -> dict:
    """
    Contamination test: fraction of exact-duplicate rows (hash-based).
    Near-duplicate detection (fuzzy match) is a possible future upgrade —
    Day-0 spec starts with exact/near-dup via hashing; this implementation
    covers exact duplicates first since that's what the adversarial suite's
    duplication case actually produces.
    """
    n_total = len(df)
    if n_total == 0:
        return _finding("contamination", CONTAMINATION_THRESHOLD, 0.0, True,
                         "Empty dataset — nothing to check.")

    n_duplicate_rows = df.duplicated(keep="first").sum()
    fraction = n_duplicate_rows / n_total
    passed = fraction <= CONTAMINATION_THRESHOLD

    detail = (
        f"{n_duplicate_rows} of {n_total} rows ({fraction:.2%}) are exact "
        f"duplicates — {'exceeds' if not passed else 'below'} the "
        f"{CONTAMINATION_THRESHOLD:.0%} ceiling."
    )
    return _finding("contamination", CONTAMINATION_THRESHOLD, fraction, passed, detail)


def check_imbalance(df: pd.DataFrame, target_column: str) -> dict:
    """
    Imbalance test: minority-class recall using a quick baseline classifier
    (LogisticRegression on numeric-encoded features), NOT AutoGluon's own
    models — this is a fast, independent check, not a repeat of the
    Experiment Node's work.

    Binary classification only for now (matches the adversarial suite's
    Survived 0/1 target). Multiclass support is a possible follow-up.
    """
    if target_column not in df.columns:
        return _finding("imbalance", IMBALANCE_THRESHOLD, 1.0, True,
                         f"Target column '{target_column}' not found — skipped.")

    work_df = df.copy()
    y = work_df[target_column]
    X = work_df.drop(columns=[target_column])

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True)).select_dtypes(include=[np.number])

    if y.nunique() != 2:
        return _finding("imbalance", IMBALANCE_THRESHOLD, 1.0, True,
                         f"Target has {y.nunique()} classes — imbalance check "
                         "currently only supports binary targets, skipped.")

    minority_class = y.value_counts().idxmin()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        true_positives = ((preds == minority_class) & (y_test == minority_class)).sum()
        actual_minority = (y_test == minority_class).sum()
        recall = true_positives / actual_minority if actual_minority > 0 else 0.0
    except Exception as e:
        return _finding("imbalance", IMBALANCE_THRESHOLD, 0.0, False,
                         f"Imbalance check failed to run ({e}) — treated as fail, "
                         "needs manual review.")

    passed = recall >= IMBALANCE_THRESHOLD
    overall_accuracy = (preds == y_test).mean()

    detail = (
        f"Minority class '{minority_class}' recall {recall:.2f} while overall "
        f"accuracy {overall_accuracy:.2f} — "
        f"{'below floor, headline metric masks failure' if not passed else 'above floor'}."
    )
    return _finding("imbalance", IMBALANCE_THRESHOLD, recall, passed, detail)


def critic_node(state: dict) -> dict:
    """
    LangGraph node entry point. Reads dataset_path + target_column from
    AgentState, runs all three checks, writes critic_findings back.

    Input (from AgentState): dataset_path, target_column
    Output (to AgentState):  critic_findings — list[dict], always exactly
                              3 entries (leakage, contamination, imbalance),
                              in that fixed order.
    """
    df = pd.read_csv(state["dataset_path"])
    target_column = state["target_column"]

    findings = [
        check_leakage(df, target_column),
        check_contamination(df),
        check_imbalance(df, target_column),
    ]

    return {"critic_findings": findings}


if __name__ == "__main__":
    # Smoke test against the real adversarial suite built tonight.
    from pathlib import Path

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
        result = critic_node({"dataset_path": str(path), "target_column": target})
        fails = [f["test"] for f in result["critic_findings"] if not f["passed"]]
        print(f"\n{name:12s} failing_tests={fails or '-'}")
        for f in result["critic_findings"]:
            mark = "PASS" if f["passed"] else "FAIL"
            print(f"  [{mark}] {f['test']:15s} measured={f['measured_value']:<8} {f['detail']}")