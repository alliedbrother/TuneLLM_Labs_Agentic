#!/usr/bin/env python3
"""Validate a prepared training dataset.

Checks format compliance, runs deduplication analysis, computes quality stats.

Usage:
    python validate_dataset.py --dataset-dir workspace/datasets/ds-v0.1.0-math/
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(filepath):
    """Load records from a JSONL file."""
    records = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON on line {i + 1}")
    return records


def check_format(records, filepath):
    """Check that all records have required Alpaca fields."""
    issues = []
    required = {"instruction", "output"}

    for i, record in enumerate(records):
        missing = required - set(record.keys())
        if missing:
            issues.append(f"Record {i}: missing fields {missing}")
        if not record.get("instruction", "").strip():
            issues.append(f"Record {i}: empty instruction")
        if not record.get("output", "").strip():
            issues.append(f"Record {i}: empty output")

    return issues


def check_duplicates(records):
    """Find exact and near-duplicate records."""
    seen_hashes = {}
    exact_dupes = 0

    for i, record in enumerate(records):
        # Hash the instruction for exact dedup
        text = record.get("instruction", "") + "|||" + record.get("output", "")
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_hashes:
            exact_dupes += 1
        else:
            seen_hashes[h] = i

    return exact_dupes, len(seen_hashes)


def compute_stats(records):
    """Compute dataset statistics."""
    instruction_lengths = [len(r.get("instruction", "").split()) for r in records]
    output_lengths = [len(r.get("output", "").split()) for r in records]

    return {
        "total_records": len(records),
        "avg_instruction_words": round(sum(instruction_lengths) / max(len(instruction_lengths), 1), 1),
        "avg_output_words": round(sum(output_lengths) / max(len(output_lengths), 1), 1),
        "min_instruction_words": min(instruction_lengths) if instruction_lengths else 0,
        "max_instruction_words": max(instruction_lengths) if instruction_lengths else 0,
        "min_output_words": min(output_lengths) if output_lengths else 0,
        "max_output_words": max(output_lengths) if output_lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a training dataset")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_file = dataset_dir / "train.jsonl"
    eval_file = dataset_dir / "eval.jsonl"

    report = {"dataset_dir": str(dataset_dir), "valid": True, "issues": []}

    for name, filepath in [("train", train_file), ("eval", eval_file)]:
        if not filepath.exists():
            report["issues"].append(f"Missing {name} file: {filepath}")
            report["valid"] = False
            continue

        records = load_jsonl(filepath)
        logger.info(f"Loaded {len(records)} records from {name}")

        # Format check
        format_issues = check_format(records, filepath)
        if format_issues:
            report["issues"].extend([f"[{name}] {i}" for i in format_issues[:10]])
            if len(format_issues) > 10:
                report["issues"].append(f"[{name}] ... and {len(format_issues) - 10} more issues")

        # Dedup check
        exact_dupes, unique = check_duplicates(records)
        if exact_dupes > 0:
            report["issues"].append(f"[{name}] {exact_dupes} exact duplicates found")

        # Stats
        stats = compute_stats(records)
        report[f"{name}_stats"] = stats
        report[f"{name}_duplicates"] = exact_dupes

    if report["issues"]:
        report["valid"] = len([i for i in report["issues"] if "Missing" in i]) == 0
        logger.warning(f"Validation issues found: {len(report['issues'])}")
    else:
        logger.info("Dataset validation passed!")

    print(f"__VALIDATION__:{json.dumps(report)}")


if __name__ == "__main__":
    main()
