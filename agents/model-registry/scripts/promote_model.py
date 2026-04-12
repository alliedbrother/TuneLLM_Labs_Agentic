#!/usr/bin/env python3
"""Promote a model version to a new stage.

Stages: registered → evaluated → staged → production → retired

Usage:
    python promote_model.py --version "v0.1.0-math" --stage "evaluated"
    python promote_model.py --version "v0.1.0-math" --stage "production"
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

VALID_STAGES = ["registered", "evaluated", "staged", "production", "retired"]


def load_registry():
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f) or {}


def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Promote a model version")
    parser.add_argument("--version", required=True, help="Model version to promote")
    parser.add_argument("--stage", required=True, choices=VALID_STAGES, help="New stage")
    args = parser.parse_args()

    registry = load_registry()
    models = registry.get("models", {})

    if args.version not in models:
        logger.error(f"Version {args.version} not found in registry")
        sys.exit(1)

    old_stage = models[args.version].get("stage", "unknown")
    models[args.version]["stage"] = args.stage
    models[args.version][f"{args.stage}_at"] = datetime.now(timezone.utc).isoformat()

    # If promoting to production, update current_production and retire the old one
    if args.stage == "production":
        old_production = registry.get("current_production")
        if old_production and old_production != args.version and old_production in models:
            models[old_production]["stage"] = "retired"
            models[old_production]["retired_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"Retired previous production model: {old_production}")
        registry["current_production"] = args.version

    save_registry(registry)

    logger.info(f"Promoted {args.version}: {old_stage} → {args.stage}")
    print(f"__MODEL_PROMOTED__:{json.dumps({'version': args.version, 'old_stage': old_stage, 'new_stage': args.stage})}")


if __name__ == "__main__":
    main()
