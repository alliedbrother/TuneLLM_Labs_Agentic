#!/usr/bin/env python3
"""Generate a manifest.yaml for a dataset directory.

Usage:
    python create_manifest.py --dataset-dir workspace/datasets/ds-v0.1.0-math/ --source "hf://gsm8k" --weakness "math_reasoning"
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def count_jsonl(filepath):
    """Count records in a JSONL file."""
    count = 0
    with open(filepath) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_file_size(filepath):
    """Get file size in MB."""
    return round(os.path.getsize(filepath) / (1024 * 1024), 2)


def main():
    parser = argparse.ArgumentParser(description="Create dataset manifest")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--source", required=True, help="Data source (e.g., 'hf://gsm8k')")
    parser.add_argument("--weakness", default="general", help="Target weakness category")
    parser.add_argument("--version", default=None, help="Dataset version (auto-detected from dir name)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_file = dataset_dir / "train.jsonl"
    eval_file = dataset_dir / "eval.jsonl"

    # Auto-detect version from directory name
    version = args.version or dataset_dir.name

    manifest = {
        "version": version,
        "source": args.source,
        "weakness_target": args.weakness,
        "format": "alpaca",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    total_records = 0
    total_size_mb = 0

    for name, filepath in [("train", train_file), ("eval", eval_file)]:
        if filepath.exists():
            count = count_jsonl(filepath)
            size = get_file_size(filepath)
            manifest["files"][name] = {
                "path": str(filepath.name),
                "records": count,
                "size_mb": size,
            }
            total_records += count
            total_size_mb += size

    manifest["total_records"] = total_records
    manifest["total_size_mb"] = round(total_size_mb, 2)

    # Write manifest
    manifest_path = dataset_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Manifest written to {manifest_path}")
    logger.info(f"Total records: {total_records}, Size: {total_size_mb:.2f}MB")
    print(f"__MANIFEST__:{json.dumps(manifest, default=str)}")


if __name__ == "__main__":
    main()
