#!/usr/bin/env python3
"""Select and prepare a training dataset.

Downloads from HuggingFace Hub or filters local files, formats as Alpaca JSONL,
splits into train/eval, and writes to the workspace.

Usage:
    python select_data.py --source "hf://gsm8k" --output-dir workspace/datasets/ds-v0.1.0-math/
    python select_data.py --source /path/to/data.jsonl --max-samples 5000 --output-dir workspace/datasets/ds-v0.1.0-test/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)


def load_from_huggingface(source, split="train", max_samples=None, config_name=None):
    """Load dataset from HuggingFace Hub."""
    from datasets import load_dataset

    dataset_name = source.replace("hf://", "") if source.startswith("hf://") else source
    logger.info(f"Loading from HuggingFace: {dataset_name}")

    kwargs = {"split": split}
    if config_name:
        kwargs["name"] = config_name

    try:
        ds = load_dataset(dataset_name, **kwargs)
    except ValueError as e:
        # If config name is required, try 'main' as default
        if "Config name is missing" in str(e):
            logger.info("Config name required, using 'main'")
            kwargs["name"] = "main"
            ds = load_dataset(dataset_name, **kwargs)
        else:
            raise

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
        logger.info(f"Sampled {max_samples} from {len(ds)} examples")

    return ds


def load_from_local(source, max_samples=None):
    """Load dataset from a local file."""
    from datasets import load_dataset

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    logger.info(f"Loading from local file: {source}")

    if path.suffix in (".json", ".jsonl"):
        ds = load_dataset("json", data_files=str(path), split="train")
    elif path.suffix == ".csv":
        ds = load_dataset("csv", data_files=str(path), split="train")
    elif path.suffix == ".parquet":
        ds = load_dataset("parquet", data_files=str(path), split="train")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    return ds


def detect_format(ds):
    """Detect the dataset format and return column mapping."""
    columns = ds.column_names

    # Alpaca format
    if "instruction" in columns and "output" in columns:
        return "alpaca", {"instruction": "instruction", "input": "input", "output": "output"}

    # OpenAI format
    if "prompt" in columns and "completion" in columns:
        return "openai", {"instruction": "prompt", "input": None, "output": "completion"}

    # Chat format
    if "messages" in columns:
        return "chat", {"messages": "messages"}

    # GSM8K format
    if "question" in columns and "answer" in columns:
        return "gsm8k", {"instruction": "question", "input": None, "output": "answer"}

    # Generic Q&A
    if "question" in columns and "response" in columns:
        return "qa", {"instruction": "question", "input": None, "output": "response"}

    logger.warning(f"Unknown format. Columns: {columns}")
    return "unknown", {}


def convert_to_alpaca(ds, format_name, mapping):
    """Convert dataset to Alpaca format (instruction/input/output)."""
    records = []

    for row in ds:
        if format_name == "chat":
            # Extract from chat messages
            messages = row.get("messages", [])
            instruction = ""
            output = ""
            for msg in messages:
                if msg.get("role") == "user":
                    instruction = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    output = msg.get("content", "")
            if instruction and output:
                records.append({"instruction": instruction, "input": "", "output": output})
        else:
            instruction = row.get(mapping.get("instruction", ""), "")
            input_text = row.get(mapping.get("input", ""), "") if mapping.get("input") else ""
            output = row.get(mapping.get("output", ""), "")
            if instruction and output:
                records.append({
                    "instruction": str(instruction),
                    "input": str(input_text) if input_text else "",
                    "output": str(output),
                })

    return records


def split_dataset(records, eval_ratio=0.1, seed=42):
    """Split records into train and eval sets."""
    import random
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_set = shuffled[:eval_size]
    train_set = shuffled[eval_size:]

    return train_set, eval_set


def write_jsonl(records, filepath):
    """Write records as JSONL."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Select and prepare training dataset")
    parser.add_argument("--source", required=True, help="HF dataset (hf://name) or local file path")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to select")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Eval split ratio (default: 0.1)")
    parser.add_argument("--split", type=str, default="train", help="HF dataset split to use")
    parser.add_argument("--format", type=str, default="auto", help="Force format: alpaca, openai, chat, auto")
    args = parser.parse_args()

    # Load dataset
    source = args.source
    if source.startswith("hf://") or (not os.path.exists(source) and "/" in source and not source.startswith(".")):
        ds = load_from_huggingface(source, split=args.split, max_samples=args.max_samples)
    else:
        ds = load_from_local(source, max_samples=args.max_samples)

    logger.info(f"Loaded {len(ds)} examples")

    # Detect and convert format
    format_name, mapping = detect_format(ds)
    logger.info(f"Detected format: {format_name}")

    records = convert_to_alpaca(ds, format_name, mapping)
    logger.info(f"Converted {len(records)} records to Alpaca format")

    if not records:
        logger.error("No valid records after conversion!")
        sys.exit(1)

    # Split
    train_set, eval_set = split_dataset(records, eval_ratio=args.eval_split)
    logger.info(f"Split: {len(train_set)} train, {len(eval_set)} eval")

    # Write outputs
    output_dir = args.output_dir
    write_jsonl(train_set, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(eval_set, os.path.join(output_dir, "eval.jsonl"))

    # Output summary
    summary = {
        "source": args.source,
        "format": format_name,
        "total_records": len(records),
        "train_records": len(train_set),
        "eval_records": len(eval_set),
        "output_dir": os.path.abspath(output_dir),
    }
    print(f"__DATASET_READY__:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
