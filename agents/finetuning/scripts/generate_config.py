#!/usr/bin/env python3
"""Generate a training configuration YAML from task parameters.

Usage:
    python generate_config.py --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
        --dataset-path workspace/datasets/ds-v0.1.0-math/train.jsonl \
        --method lora --run-name "v0.1.0-math" --output configs/run.yaml
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


DEFAULT_LORA_CONFIG = {
    "run_name": "finetune-run",
    "method": "lora",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset": {
        "source": "",
        "eval_split": 0.1,
        "max_samples": None,
        "prompt_template": (
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        ),
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "max_length": 2048,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "quantization": None,
    "save_full_model": False,
    "use_wandb": False,
}

DEFAULT_QLORA_CONFIG = {
    **DEFAULT_LORA_CONFIG,
    "method": "qlora",
    "quantization": "4bit",
    "training": {
        **DEFAULT_LORA_CONFIG["training"],
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate training config YAML")
    parser.add_argument("--base-model", required=True, help="Base model name/path")
    parser.add_argument("--dataset-path", required=True, help="Path to training dataset")
    parser.add_argument("--method", default="lora", choices=["lora", "qlora", "dpo", "full"],
                       help="Training method")
    parser.add_argument("--run-name", required=True, help="Run name")
    parser.add_argument("--output", required=True, help="Output config YAML path")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--lora-rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    args = parser.parse_args()

    # Start from template
    if args.method == "qlora":
        config = dict(DEFAULT_QLORA_CONFIG)
        config["training"] = dict(DEFAULT_QLORA_CONFIG["training"])
        config["lora"] = dict(DEFAULT_QLORA_CONFIG["lora"])
        config["dataset"] = dict(DEFAULT_QLORA_CONFIG["dataset"])
    else:
        config = dict(DEFAULT_LORA_CONFIG)
        config["training"] = dict(DEFAULT_LORA_CONFIG["training"])
        config["lora"] = dict(DEFAULT_LORA_CONFIG["lora"])
        config["dataset"] = dict(DEFAULT_LORA_CONFIG["dataset"])

    # Apply task parameters
    config["run_name"] = args.run_name
    config["method"] = args.method
    config["base_model"] = args.base_model
    config["dataset"]["source"] = args.dataset_path

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.lora_rank:
        config["lora"]["r"] = args.lora_rank
        config["lora"]["alpha"] = args.lora_rank * 2
    if args.max_samples:
        config["dataset"]["max_samples"] = args.max_samples
    if args.max_length:
        config["training"]["max_length"] = args.max_length

    # Write config
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Config written to {args.output}")
    logger.info(f"  Model: {config['base_model']}")
    logger.info(f"  Method: {config['method']}")
    logger.info(f"  Dataset: {config['dataset']['source']}")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  LR: {config['training']['learning_rate']}")

    print(f"__CONFIG_GENERATED__:{json.dumps({'path': os.path.abspath(args.output), 'run_name': args.run_name})}")


if __name__ == "__main__":
    main()
