#!/usr/bin/env python3
"""Main training entry point."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure local imports resolve regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML, JSON, or environment variable.

    If config_path is 'env', reads from JOB_CONFIG environment variable.
    """
    if config_path == "env":
        config_str = os.environ.get("JOB_CONFIG")
        if config_str:
            return json.loads(config_str)
        raise ValueError("--config=env but JOB_CONFIG environment variable not set")

    path = Path(config_path)
    if not path.exists():
        # Fallback: try loading from environment variable
        config_str = os.environ.get("JOB_CONFIG")
        if config_str:
            return json.loads(config_str)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        if path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="TuneLLM Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="/models", help="Output directory")
    parser.add_argument("--evaluate-before", action="store_true", help="Evaluate base model before training")
    parser.add_argument("--evaluate-after", action="store_true", help="Evaluate fine-tuned model after training")
    parser.add_argument("--test-dataset", type=str, default=None, help="Path to test dataset for evaluation")
    parser.add_argument("--eval-max-samples", type=int, default=50, help="Max samples for evaluation")
    args = parser.parse_args()

    # Load configuration
    console.print("[bold blue]TuneLLM Training[/bold blue]")
    console.print(f"Loading config from: {args.config}")

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Display config
    console.print(f"Run name: {config.get('run_name', 'unnamed')}")
    console.print(f"Base model: {config.get('base_model')}")
    console.print(f"Method: {config.get('method', 'lora')}")

    # Check GPU availability
    if torch.cuda.is_available():
        console.print(f"[green]GPU available: {torch.cuda.get_device_name(0)}[/green]")
        console.print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        console.print("[yellow]Warning: No GPU available, training will be slow[/yellow]")

    # Select training method
    method = config.get("method", "lora").lower()

    try:
        if method == "lora":
            from lora_trainer import LoRATrainer

            trainer = LoRATrainer(config, args.output_dir)
        elif method == "qlora":
            from qlora_trainer import QLoRATrainer

            trainer = QLoRATrainer(config, args.output_dir)
        elif method == "dpo":
            from dpo_trainer import DPOTrainer

            trainer = DPOTrainer(config, args.output_dir)
        elif method == "ppo":
            from ppo_trainer import PPOTrainer

            trainer = PPOTrainer(config, args.output_dir)
        elif method == "full":
            from lora_trainer import LoRATrainer

            # Full fine-tuning uses the same trainer without PEFT
            config["lora"] = None
            trainer = LoRATrainer(config, args.output_dir)
        else:
            logger.error(f"Unknown training method: {method}")
            sys.exit(1)

        # Pre-training evaluation
        if args.evaluate_before and args.test_dataset:
            console.print("[cyan]Running pre-training evaluation...[/cyan]")
            try:
                from evaluate import ModelEvaluator, load_test_data

                test_data = load_test_data(args.test_dataset)
                evaluator = ModelEvaluator(
                    model_name=config.get("base_model"),
                    max_new_tokens=128,
                )
                evaluator.load_model()
                baseline_results = evaluator.evaluate_qa(test_data, max_samples=args.eval_max_samples)

                # Save baseline metrics
                baseline_path = os.path.join(args.output_dir, "baseline_metrics.json")
                os.makedirs(args.output_dir, exist_ok=True)
                with open(baseline_path, "w") as f:
                    json.dump(baseline_results["metrics"], f, indent=2)
                console.print(f"[green]Baseline metrics saved to {baseline_path}[/green]")
                print(f"__BASELINE_METRICS__:{json.dumps(baseline_results['metrics'])}")

                # Clean up to free memory
                del evaluator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Pre-training evaluation failed: {e}")

        # Run training
        console.print(f"[cyan]Starting {method.upper()} training...[/cyan]")
        trainer.train()
        console.print("[green]Training completed successfully![/green]")

        # Post-training evaluation
        if args.evaluate_after and args.test_dataset:
            console.print("[cyan]Running post-training evaluation...[/cyan]")
            try:
                from evaluate import ModelEvaluator, load_test_data

                test_data = load_test_data(args.test_dataset)

                # Find the saved adapter/model
                run_name = config.get("run_name", "unnamed")
                adapter_path = os.path.join(args.output_dir, run_name, "final")
                if not os.path.exists(adapter_path):
                    adapter_path = os.path.join(args.output_dir, run_name)

                evaluator = ModelEvaluator(
                    model_name=config.get("base_model"),
                    adapter_path=adapter_path if os.path.exists(adapter_path) else None,
                    max_new_tokens=128,
                )
                evaluator.load_model()
                final_results = evaluator.evaluate_qa(test_data, max_samples=args.eval_max_samples)

                # Save final metrics
                final_path = os.path.join(args.output_dir, "final_metrics.json")
                with open(final_path, "w") as f:
                    json.dump(final_results["metrics"], f, indent=2)
                console.print(f"[green]Final metrics saved to {final_path}[/green]")
                print(f"__FINAL_METRICS__:{json.dumps(final_results['metrics'])}")
            except Exception as e:
                logger.warning(f"Post-training evaluation failed: {e}")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
