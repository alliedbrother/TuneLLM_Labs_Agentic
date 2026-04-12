#!/usr/bin/env python3
"""Model evaluation script for Q&A tasks.

Loads a model (base or fine-tuned with adapter), generates answers
for a test dataset, and computes evaluation metrics.

Usage:
    python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --test-dataset test.jsonl --output-dir ./eval_results
    python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter ./lora_adapter --test-dataset test.jsonl
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure local imports resolve regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate a language model on Q&A tasks."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

        if self.adapter_path:
            logger.info(f"Loading adapter from: {self.adapter_path}")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        logger.info("Model loaded successfully")

    def generate_answer(self, instruction: str, input_text: str = "") -> str:
        """Generate an answer for a single Q&A pair."""
        if input_text:
            prompt = f"Context: {input_text}\n\nQuestion: {instruction}\n\nAnswer:"
        else:
            prompt = f"Question: {instruction}\n\nAnswer:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer

    def evaluate_qa(self, test_data: list[dict], max_samples: int = None) -> dict:
        """Evaluate the model on a Q&A test dataset.

        Args:
            test_data: List of dicts with 'instruction', 'input', 'output' keys
            max_samples: Limit evaluation to N samples (for speed)

        Returns:
            Dictionary with metrics and per-sample results
        """
        if max_samples:
            test_data = test_data[:max_samples]

        logger.info(f"Evaluating on {len(test_data)} samples...")

        predictions = []
        references = []

        for i, item in enumerate(test_data):
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            reference = item.get("output", "")

            prediction = self.generate_answer(instruction, input_text)
            predictions.append(prediction)
            references.append(reference)

            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(test_data)}")

        # Compute metrics
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.metrics import compute_all_qa_metrics

        metrics = compute_all_qa_metrics(predictions, references)

        logger.info(f"\nEvaluation Results:")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key}: {value:.4f}")

        return {
            "metrics": metrics,
            "num_samples": len(test_data),
            "predictions": predictions,
            "references": references,
        }


def load_test_data(path: str) -> list[dict]:
    """Load test data from a JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on Q&A tasks")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--test-dataset", type=str, required=True, help="Path to test JSONL")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    # Load test data
    test_data = load_test_data(args.test_dataset)
    logger.info(f"Loaded {len(test_data)} test samples from {args.test_dataset}")

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_name=args.model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )
    evaluator.load_model()

    # Run evaluation
    results = evaluator.evaluate_qa(test_data, max_samples=args.max_samples)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)

    details_path = os.path.join(args.output_dir, "eval_details.jsonl")
    with open(details_path, "w") as f:
        for pred, ref in zip(results["predictions"], results["references"]):
            f.write(json.dumps({"prediction": pred, "reference": ref}) + "\n")

    logger.info(f"\nResults saved to {args.output_dir}")
    # Also print metrics as JSON to stdout for the agent to parse
    print(f"\n__EVAL_METRICS__:{json.dumps(results['metrics'])}")


if __name__ == "__main__":
    main()
