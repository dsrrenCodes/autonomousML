"""
Data Node — deterministic. No LLM call.

Input:  raw CSV at dataset_path, target_column (post target-column-confirmation UI)
Output: cleaned_data_summary — dtype report + edge-case flags.

This node does NOT rewrite the dataset to disk. It inspects the raw CSV and
writes a summary dict to AgentState["cleaned_data_summary"]; downstream
nodes (Experiment, Critic) still read the original dataset_path directly.
If real cleaning transforms are added later (imputation, dropping bad
columns, etc.), that's a deliberate scope expansion — not implied by the
current AgentState shape, which only has a dict, not a second file path.

Edge cases flagged (Day-0 §1 / spec):
    wrong_dtype        column dtype looks inconsistent with its values
                        (e.g. numeric-looking column stored as object)
    single_class_target  target column has only 1 unique value —
                        nothing to predict, should halt the pipeline
    empty_column        column is 100% null

Owner: Person B
"""

import pandas as pd
import numpy as np


def _detect_wrong_dtype_columns(df: pd.DataFrame) -> list[str]:
    """
    Flags object-dtype columns where every non-null value is actually
    numeric (e.g. a column of "1", "2", "3" strings stored as object
    instead of int/float — common CSV-parsing artifact).
    """
    flagged = []
    for col in df.columns:
        if df[col].dtype == "object":
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            numeric_coerced = pd.to_numeric(non_null, errors="coerce")
            if numeric_coerced.notna().all():
                flagged.append(col)
    return flagged


def _detect_empty_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are 100% null."""
    return [col for col in df.columns if df[col].isna().all()]


def _detect_single_class_target(df: pd.DataFrame, target_column: str) -> bool:
    """True if the target column has only one unique non-null value."""
    if target_column not in df.columns:
        return False
    return df[target_column].dropna().nunique() <= 1


def _dtype_report(df: pd.DataFrame) -> dict:
    """Per-column dtype + null count + unique count, for the summary."""
    report = {}
    for col in df.columns:
        report[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_fraction": round(float(df[col].isna().mean()), 4),
            "unique_count": int(df[col].nunique(dropna=True)),
        }
    return report


def data_node(state: dict) -> dict:
    """
    LangGraph node entry point.

    Input (from AgentState):  dataset_path, target_column
    Output (to AgentState):   cleaned_data_summary — dict with:
        n_rows, n_columns, dtype_report, wrong_dtype_columns,
        empty_columns, single_class_target (bool), target_column,
        halt_recommended (bool) — True if single_class_target or the
        target column itself is empty; downstream nodes/Router should
        treat this as a hard stop, not just a warning.
    """
    df = pd.read_csv(state["dataset_path"])
    target_column = state["target_column"]

    wrong_dtype_cols = _detect_wrong_dtype_columns(df)
    empty_cols = _detect_empty_columns(df)
    single_class = _detect_single_class_target(df, target_column)
    target_missing = target_column not in df.columns
    target_is_empty = target_column in empty_cols

    halt_recommended = single_class or target_missing or target_is_empty

    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "target_column": target_column,
        "dtype_report": _dtype_report(df),
        "wrong_dtype_columns": wrong_dtype_cols,
        "empty_columns": empty_cols,
        "single_class_target": single_class,
        "target_column_missing": target_missing,
        "halt_recommended": halt_recommended,
    }

    return {"cleaned_data_summary": summary}


if __name__ == "__main__":
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "adversarial_suite"

    test_cases = {
        "clean_1": ("titanic_clean_1.csv", "Survived"),
        "leakage": ("titanic_leakage.csv", "Survived"),
        "duplicates": ("titanic_duplicates.csv", "Survived"),
        "imbalance": ("titanic_imbalance.csv", "Survived"),
        "prompt_injection": ("titanic_prompt_injection.csv", "Survived"),
    }

    for name, (filename, target) in test_cases.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"{name:18s} SKIPPED — file not found at {path}")
            continue
        result = data_node({"dataset_path": str(path), "target_column": target})
        s = result["cleaned_data_summary"]
        print(f"\n{name:18s} rows={s['n_rows']} cols={s['n_columns']} "
              f"halt_recommended={s['halt_recommended']}")
        if s["wrong_dtype_columns"]:
            print(f"  wrong_dtype_columns: {s['wrong_dtype_columns']}")
        if s["empty_columns"]:
            print(f"  empty_columns: {s['empty_columns']}")
        if s["single_class_target"]:
            print(f"  single_class_target: True")