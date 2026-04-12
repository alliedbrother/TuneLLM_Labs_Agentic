#!/usr/bin/env python3
"""Query the model registry for version info.

Usage:
    python get_model_info.py --version "v0.1.0-math"
    python get_model_info.py --stage production
    python get_model_info.py --list
"""

import argparse
import json
import logging
import os
import sys

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "workspace", "registry", "registry.yaml")


def load_registry():
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Query model registry")
    parser.add_argument("--version", type=str, help="Get info for a specific version")
    parser.add_argument("--stage", type=str, help="Find model(s) at a given stage")
    parser.add_argument("--list", action="store_true", help="List all model versions")
    args = parser.parse_args()

    registry = load_registry()
    models = registry.get("models", {})

    if args.version:
        if args.version in models:
            info = {"version": args.version, **models[args.version]}
            print(json.dumps(info, indent=2, default=str))
        else:
            logger.error(f"Version {args.version} not found")
            sys.exit(1)

    elif args.stage:
        matches = {v: info for v, info in models.items() if info.get("stage") == args.stage}
        if matches:
            for v, info in matches.items():
                print(json.dumps({"version": v, **info}, indent=2, default=str))
        else:
            logger.info(f"No models at stage '{args.stage}'")

    elif args.list:
        current = registry.get("current_production", "none")
        print(f"Current production: {current}\n")
        for v, info in models.items():
            marker = " ◄ PRODUCTION" if v == current else ""
            print(f"  {v}: stage={info.get('stage', '?')}, "
                  f"base={info.get('base_model', '?')}, "
                  f"method={info.get('training_method', info.get('type', '?'))}"
                  f"{marker}")
    else:
        # Default: show current production
        current = registry.get("current_production")
        if current and current in models:
            info = {"version": current, "is_production": True, **models[current]}
            print(json.dumps(info, indent=2, default=str))
        else:
            logger.info("No production model set")


if __name__ == "__main__":
    main()
