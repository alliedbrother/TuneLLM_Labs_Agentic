#!/usr/bin/env python3
"""PEFT-based LoRA fine-tuning script.

Compatible with torch 2.4.x + transformers 4.44.x + peft 0.18.x.
Does NOT require unsloth.

Usage:
  JOB_CONFIG='{"base_model": "...", ...}' python3 train_peft.py
  python3 train_peft.py --config config.json
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("peft-trainer")


def load_config() -> dict:
    config_str = os.environ.get("JOB_CONFIG")
    if config_str:
        return json.loads(config_str)
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
        with open(config_path) as f:
            if config_path.endswith((".yaml", ".yml")):
                import yaml
                return yaml.safe_load(f)
            return json.load(f)
    raise ValueError("No config. Set JOB_CONFIG or use --config <path>")


def run_evaluation(model, tokenizer, test_path, max_samples=7):
    """Run evaluation and return metrics dict."""
    from datasets import load_dataset
    import torch

    logger.info(f"Running evaluation on {test_path} (max {max_samples} samples)...")
    test_ds = load_dataset("json", data_files=test_path, split="train")
    if len(test_ds) > max_samples:
        test_ds = test_ds.select(range(max_samples))

    model.eval()
    predictions, references = [], []

    for row in test_ds:
        user_msg = row.get("instruction", row.get("question", ""))
        if row.get("input"):
            user_msg += f"\n\nContext: {row['input']}"
        reference = row.get("output", row.get("answer", ""))

        messages = [{"role": "user", "content": user_msg}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"### Instruction:\n{user_msg}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append(answer)
        references.append(reference)
        logger.info(f"  Q: {user_msg[:80]}...")
        logger.info(f"  A: {answer[:100]}...")

    # Compute F1, EM, ROUGE
    f1_scores, em_scores = [], []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            em_scores.append(0.0)
            continue
        common = pred_tokens & ref_tokens
        if not common:
            f1_scores.append(0.0)
        else:
            p = len(common) / len(pred_tokens)
            r = len(common) / len(ref_tokens)
            f1_scores.append(2 * p * r / (p + r))
        em_scores.append(float(pred.strip().lower() == ref.strip().lower()))

    metrics = {
        "f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
        "exact_match": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0,
        "num_samples": len(predictions),
    }

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        r1, rL = [], []
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            r1.append(s["rouge1"].fmeasure)
            rL.append(s["rougeL"].fmeasure)
        metrics["rouge1"] = round(sum(r1) / len(r1), 4)
        metrics["rougeL"] = round(sum(rL) / len(rL), 4)
    except ImportError:
        pass

    logger.info(f"Eval results: {metrics}")
    return metrics


def main():
    config = load_config()

    base_model = config["base_model"]
    run_name = config.get("run_name", "peft-run")
    training_config = config.get("training", {})
    lora_config = config.get("lora", {})
    dataset_config = config.get("dataset", {})

    logger.info(f"=== PEFT LoRA Training ===")
    logger.info(f"Model: {base_model}, Run: {run_name}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import load_dataset

    print("__PHASE__:loading_model", flush=True)

    # V100 has 32GB — load 3B in fp16 (~6.4GB), no quantization needed
    dtype = torch.float16  # V100 doesn't support bfloat16 well
    logger.info(f"Loading tokenizer for {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Loading model in fp16")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Pre-eval
    if config.get("evaluate_before") and config.get("test_dataset"):
        print("__PHASE__:evaluating_base", flush=True)
        baseline = run_evaluation(model, tokenizer, config["test_dataset"])
        print(f"__BASELINE_METRICS__:{json.dumps(baseline)}", flush=True)

    # Apply LoRA
    lora_r = lora_config.get("r", 16)
    lora_alpha = lora_config.get("alpha", 32)
    lora_dropout = lora_config.get("dropout", 0.05)
    target_modules = lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset_source = dataset_config.get("source", "")
    print("__PHASE__:downloading_data", flush=True)
    if os.path.exists(dataset_source):
        raw_dataset = load_dataset("json", data_files=dataset_source, split="train")
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_source}")

    # Format dataset
    def format_row(row):
        user_msg = row.get("instruction", row.get("question", ""))
        if row.get("input"):
            user_msg += f"\n\nContext: {row['input']}"
        assistant_msg = row.get("output", row.get("answer", ""))
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = f"### Instruction:\n{user_msg}\n\n### Response:\n{assistant_msg}"
        return {"text": text}

    formatted_dataset = raw_dataset.map(format_row, remove_columns=raw_dataset.column_names)
    logger.info(f"Dataset ready: {len(formatted_dataset)} samples. Sample: {formatted_dataset[0]['text'][:200]}")

    # Training
    epochs = training_config.get("epochs", training_config.get("num_epochs", 2))
    batch_size = training_config.get("batch_size", 2)
    grad_accum = training_config.get("gradient_accumulation_steps", 4)
    lr = training_config.get("learning_rate", 2e-4)
    max_seq_length = training_config.get("max_seq_length", 1024)

    output_dir = config.get("output_dir", f"/workspace/tunellm/outputs/{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    # V100 supports fp16 but not bf16; use adamw_torch (no paged) since no 4-bit
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=False,
        fp16=True,
        logging_steps=5,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=10,
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print("__PHASE__:training", flush=True)
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    logger.info(f"Training done in {elapsed:.1f}s. Loss: {train_result.training_loss:.4f}")

    # Save adapter
    save_path = os.path.join(output_dir, "final")
    print("__PHASE__:saving_model", flush=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Adapter saved to {save_path}")

    # Post-eval
    if config.get("evaluate_after") and config.get("test_dataset"):
        print("__PHASE__:evaluating_final", flush=True)
        final = run_evaluation(model, tokenizer, config["test_dataset"])
        print(f"__FINAL_METRICS__:{json.dumps(final)}", flush=True)

    print(f"__TRAINING_COMPLETE__:{json.dumps({'loss': round(train_result.training_loss, 4), 'output_dir': save_path})}", flush=True)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
