#!/usr/bin/env python3
"""
predict.py — inference script for MedQA-USMLE-4-options benchmark.

LoRA-aware: can load a base model + optional LoRA adapter and generate
predictions (A/B/C/D) for each question.

Run (from the agent_end/ directory):
    # Base model only
    python predict.py \
        --test_csv    ./MedQA-USMLE-4-options/test.csv \
        --model_path  meta-llama/Llama-3.2-1B-Instruct \
        --output_csv  ./submission.csv

    # Base model + LoRA adapter
    python predict.py \
        --test_csv    ./MedQA-USMLE-4-options/test.csv \
        --model_path  meta-llama/Llama-3.2-1B-Instruct \
        --adapter_path /path/to/lora/adapter \
        --output_csv  ./submission.csv

Output (submission.csv) columns: `id,predicted_idx` where
predicted_idx ∈ {A, B, C, D}. Do not change the output schema.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

import pandas as pd


VALID_CHOICES = ("A", "B", "C", "D")


# ==============================================================================
#                          Model loading and prediction
# ==============================================================================
def load_model(model_path: str, adapter_path: str = None):
    """Load a HuggingFace model and optional LoRA adapter.

    Returns a dict with model, tokenizer, and generation config.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[predict.py] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[predict.py] Loading base model from {model_path} (device={device}, dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)

    if adapter_path:
        print(f"[predict.py] Loading LoRA adapter from {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    print("[predict.py] Model ready")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }


def _build_prompt(question: str, options: dict) -> str:
    """Build a prompt asking the model to pick A/B/C/D."""
    opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return (
        "You are a medical expert answering a USMLE-style question. "
        "Read the clinical vignette and choose the single best answer.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        "Answer with only the letter (A, B, C, or D). Answer:"
    )


def _extract_letter(text: str) -> str:
    """Extract the first A/B/C/D letter from generated text."""
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)
    for ch in text.upper():
        if ch in VALID_CHOICES:
            return ch
    return "A"  # fallback


def predict_one(
    model,
    question: str,
    options: dict,
    meta_info: str,
    metamap_phrases: str,
) -> str:
    """Generate an answer for a single question. Returns 'A', 'B', 'C', or 'D'."""
    import torch

    prompt = _build_prompt(question, options)
    tokenizer = model["tokenizer"]
    m = model["model"]
    device = model["device"]

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = m.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _extract_letter(text)


# ==============================================================================
# Infrastructure (do not edit)
# ==============================================================================

def _parse_options(opt_str: str) -> dict:
    try:
        parsed = ast.literal_eval(opt_str)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_inference(test_csv: str, model_path: str, output_csv: str, adapter_path: str = None) -> None:
    test_df = pd.read_csv(test_csv)

    required = ["question", "options", "meta_info"]
    missing = [c for c in required if c not in test_df.columns]
    if missing:
        raise ValueError(f"test.csv is missing columns: {missing}")

    test_df = test_df.reset_index(drop=False).rename(columns={"index": "id"})

    print(f"[predict.py] Loaded {len(test_df)} test rows from {test_csv}")
    model = load_model(model_path, adapter_path=adapter_path)
    print(f"[predict.py] Running inference on {len(test_df)} questions...")

    predictions = []
    for i, row in enumerate(test_df.itertuples(index=False), start=1):
        options = _parse_options(row.options)
        metamap = row.metamap_phrases if "metamap_phrases" in test_df.columns else ""
        if pd.isna(metamap):
            metamap = ""

        try:
            pred = predict_one(
                model=model,
                question=str(row.question),
                options=options,
                meta_info=str(row.meta_info),
                metamap_phrases=str(metamap),
            )
        except Exception as e:
            print(f"[predict.py] WARN: predict_one failed on id={row.id}: {e}")
            pred = "A"

        pred = str(pred).strip().upper()
        if pred not in VALID_CHOICES:
            print(f"[predict.py] WARN: id={row.id} got invalid prediction {pred!r}, defaulting to 'A'")
            pred = "A"

        predictions.append({"id": int(row.id), "predicted_idx": pred})

        if i % 50 == 0 or i == len(test_df):
            print(f"[predict.py]   {i} / {len(test_df)} done")

    sub = pd.DataFrame(predictions, columns=["id", "predicted_idx"])
    sub.to_csv(output_csv, index=False)
    print(f"[predict.py] Wrote submission to {output_csv}  ({len(sub)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Generate submission.csv from test.csv")
    parser.add_argument("--test_csv", required=True, help="Path to test.csv")
    parser.add_argument("--model_path", required=True,
                        help="HuggingFace model name or local path (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--adapter_path", default=None,
                        help="Optional path to LoRA adapter directory")
    parser.add_argument("--output_csv", default="submission.csv",
                        help="Where to write submission.csv (default: ./submission.csv)")
    args = parser.parse_args()

    run_inference(args.test_csv, args.model_path, args.output_csv, adapter_path=args.adapter_path)


if __name__ == "__main__":
    main()
