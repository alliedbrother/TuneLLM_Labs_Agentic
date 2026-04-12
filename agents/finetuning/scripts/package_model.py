#!/usr/bin/env python3
"""Package a trained model checkpoint with metadata for the Model Registry.

Creates metadata.yaml with full provenance information.

Usage:
    python package_model.py --model-dir workspace/models/v0.1.0-math/ \
        --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
        --dataset-version "ds-v0.1.0-math" \
        --method lora --weakness "math_reasoning" \
        --training-metrics '{"final_loss": 0.42, "best_eval_loss": 0.51}'
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


def main():
    parser = argparse.ArgumentParser(description="Package model with metadata")
    parser.add_argument("--model-dir", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--base-model", required=True, help="Base model name")
    parser.add_argument("--dataset-version", required=True, help="Dataset version used for training")
    parser.add_argument("--method", required=True, help="Training method (lora, qlora, dpo, full)")
    parser.add_argument("--weakness", default="general", help="Weakness targeted")
    parser.add_argument("--run-name", default=None, help="Training run name")
    parser.add_argument("--training-metrics", default="{}", help="Training metrics JSON string")
    parser.add_argument("--config-path", default=None, help="Path to training config used")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    # Parse training metrics
    try:
        metrics = json.loads(args.training_metrics)
    except json.JSONDecodeError:
        metrics = {}

    # Detect model version from directory name
    version = model_dir.name

    # List model files
    model_files = []
    total_size = 0
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            model_files.append({
                "path": os.path.relpath(fp, model_dir),
                "size_mb": round(size / (1024 * 1024), 2),
            })
            total_size += size

    # Load training config if provided
    training_config = None
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path) as f:
            training_config = yaml.safe_load(f)

    metadata = {
        "version": version,
        "base_model": args.base_model,
        "method": args.method,
        "dataset_version": args.dataset_version,
        "weakness_targeted": args.weakness,
        "run_name": args.run_name or version,
        "stage": "registered",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "training_metrics": metrics,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "files_count": len(model_files),
        "artifacts_path": str(model_dir.resolve()),
    }

    if training_config:
        metadata["training_config"] = training_config

    # Write metadata
    metadata_path = model_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    # Write training metrics as JSON too (easier for scripts to parse)
    metrics_path = model_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model packaged: {version}")
    logger.info(f"  Base: {args.base_model}")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Dataset: {args.dataset_version}")
    logger.info(f"  Size: {metadata['total_size_mb']}MB ({len(model_files)} files)")
    logger.info(f"  Metadata: {metadata_path}")

    print(f"__MODEL_PACKAGED__:{json.dumps(metadata, default=str)}")


if __name__ == "__main__":
    main()
