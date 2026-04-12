#!/usr/bin/env python3
"""Register a new model version in the registry.

Usage:
    python register_model.py --version "v0.1.0-math" \
        --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
        --dataset "ds-v0.1.0-math" --method lora \
        --weakness "math_reasoning" \
        --artifacts-path workspace/models/v0.1.0-math/
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "workspace", "registry", "registry.yaml")


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return yaml.safe_load(f) or {}
    return {"models": {}, "current_production": None}


def save_registry(registry):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Register a model version")
    parser.add_argument("--version", required=True, help="Model version (e.g., v0.1.0-math)")
    parser.add_argument("--base-model", required=True, help="Base model name")
    parser.add_argument("--dataset", required=True, help="Dataset version used")
    parser.add_argument("--method", required=True, help="Training method")
    parser.add_argument("--weakness", default="general", help="Weakness targeted")
    parser.add_argument("--artifacts-path", required=True, help="Path to model artifacts")
    parser.add_argument("--training-metrics", default="{}", help="Training metrics JSON")
    args = parser.parse_args()

    registry = load_registry()
    if "models" not in registry:
        registry["models"] = {}

    if args.version in registry["models"]:
        logger.warning(f"Version {args.version} already exists, updating...")

    try:
        metrics = json.loads(args.training_metrics)
    except json.JSONDecodeError:
        metrics = {}

    entry = {
        "stage": "registered",
        "base_model": args.base_model,
        "dataset": args.dataset,
        "training_method": args.method,
        "weakness_targeted": args.weakness,
        "artifacts_path": os.path.abspath(args.artifacts_path),
        "training_metrics": metrics,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    registry["models"][args.version] = entry
    save_registry(registry)

    logger.info(f"Registered model {args.version} (stage: registered)")
    print(f"__MODEL_REGISTERED__:{json.dumps({'version': args.version, 'stage': 'registered'})}")


if __name__ == "__main__":
    main()
