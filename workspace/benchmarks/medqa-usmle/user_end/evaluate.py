#!/usr/bin/env python3
"""
evaluate.py — scorer.

Takes a submission CSV and compares it against the held-out test set.
Run via run_eval.sh from inside user_end/.

Usage:
    python evaluate.py \
        --test_csv     ./MedQA-USMLE-4-options/test.csv \
        --submission   ./submission.csv \
        [--output_dir  ./eval_output]

==============================================================================
                      SUBMISSION FILE FORMAT (submission.csv)
==============================================================================

The submission MUST be a CSV with:
  - UTF-8 encoding
  - A header row
  - Exactly two columns, in this order:

      id,predicted_idx

  - `id`            : integer, 0-indexed row number matching test.csv
                      (i.e. the first row of test.csv has id=0, etc.)
                      Must be UNIQUE and cover every row of test.csv.
  - `predicted_idx` : string, one of {'A', 'B', 'C', 'D'}

Example:

    id,predicted_idx
    0,B
    1,D
    2,A
    3,C
    ...

Row order does not matter — the evaluator joins on `id`.
Missing id, duplicate id, or value outside {A,B,C,D}  =>  counted wrong.
==============================================================================
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


VALID_CHOICES = {"A", "B", "C", "D"}
REQUIRED_SUBMISSION_COLUMNS = ["id", "predicted_idx"]


def load_ground_truth(test_csv: str) -> pd.DataFrame:
    """Load test.csv and attach an `id` column (= row index)."""
    df = pd.read_csv(test_csv)
    missing = [c for c in ["answer_idx", "meta_info"] if c not in df.columns]
    if missing:
        raise ValueError(f"test.csv is missing required columns: {missing}")
    df = df.reset_index(drop=False).rename(columns={"index": "id"})
    return df[["id", "answer_idx", "meta_info"]]


def load_and_validate_submission(submission_csv: str, truth: pd.DataFrame) -> pd.DataFrame:
    """Read the submission CSV and validate its format. Return a merged DF."""
    path = Path(submission_csv)
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")

    try:
        sub = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Could not read submission as CSV: {e}")

    missing_cols = [c for c in REQUIRED_SUBMISSION_COLUMNS if c not in sub.columns]
    if missing_cols:
        raise ValueError(
            f"Submission is missing required column(s) {missing_cols}. "
            f"Got columns: {list(sub.columns)}. "
            f"Expected exactly: {REQUIRED_SUBMISSION_COLUMNS}"
        )
    sub = sub[REQUIRED_SUBMISSION_COLUMNS].copy()

    try:
        sub["id"] = sub["id"].astype(int)
    except Exception:
        raise ValueError("Column `id` must be integer-castable.")
    sub["predicted_idx"] = (
        sub["predicted_idx"].astype(str).str.strip().str.upper()
    )

    dup_mask = sub.duplicated(subset="id", keep="first")
    if dup_mask.any():
        print(
            f"[WARN] {dup_mask.sum()} duplicate id(s) in submission — "
            f"keeping first occurrence."
        )
        sub = sub.loc[~dup_mask]

    merged = truth.merge(sub, on="id", how="left")

    n_missing = merged["predicted_idx"].isna().sum()
    if n_missing:
        print(f"[WARN] {n_missing} test row(s) have no prediction — counted as wrong.")

    invalid_mask = ~merged["predicted_idx"].isin(VALID_CHOICES)
    n_invalid = invalid_mask.sum()
    if n_invalid:
        print(
            f"[WARN] {n_invalid} prediction(s) are outside {sorted(VALID_CHOICES)} "
            f"(or missing) — counted as wrong."
        )
    merged.loc[invalid_mask, "predicted_idx"] = ""

    extra_ids = set(sub["id"]) - set(truth["id"])
    if extra_ids:
        print(
            f"[WARN] Submission contains {len(extra_ids)} id(s) not in test.csv "
            f"— ignored."
        )

    return merged


def score(merged: pd.DataFrame) -> dict:
    """Compute overall and per-meta_info accuracy."""
    merged = merged.copy()
    merged["correct"] = (merged["predicted_idx"] == merged["answer_idx"]).astype(int)

    overall_acc = float(merged["correct"].mean())
    by_meta_raw = (
        merged.groupby("meta_info")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .to_dict(orient="index")
    )
    by_meta = {
        k: {"accuracy": float(v["accuracy"]), "n": int(v["n"])}
        for k, v in by_meta_raw.items()
    }

    return {
        "overall_accuracy": overall_acc,
        "n_total": int(len(merged)),
        "n_correct": int(merged["correct"].sum()),
        "by_meta_info": by_meta,
    }


def pretty_print(result: dict) -> None:
    print("\n" + "=" * 60)
    print(" " * 20 + "EVALUATION RESULT")
    print("=" * 60)
    print(
        f"Overall Accuracy : {result['overall_accuracy']:.4f}  "
        f"({result['n_correct']} / {result['n_total']})"
    )
    print("-" * 60)
    print("By meta_info:")
    for meta, v in sorted(result["by_meta_info"].items()):
        print(f"  {meta:<10s}  acc = {v['accuracy']:.4f}   (n = {v['n']})")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MedQA evaluator")
    parser.add_argument("--test_csv", required=True, help="Path to test.csv with ground truth")
    parser.add_argument("--submission", required=True, help="Path to submission.csv")
    parser.add_argument("--output_dir", default="./eval_output",
                        help="Directory to write audit files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading ground truth from {args.test_csv}")
    truth = load_ground_truth(args.test_csv)
    print(f"[INFO] Test set size: {len(truth)}")

    print(f"[INFO] Loading submission from {args.submission}")
    merged = load_and_validate_submission(args.submission, truth)

    audit_path = out_dir / "graded_predictions.csv"
    merged[["id", "answer_idx", "predicted_idx", "meta_info"]].to_csv(audit_path, index=False)
    print(f"[INFO] Wrote audit trail to {audit_path}")

    result = score(merged)

    score_path = out_dir / "score.json"
    with open(score_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Wrote score to {score_path}")

    pretty_print(result)


if __name__ == "__main__":
    main()
