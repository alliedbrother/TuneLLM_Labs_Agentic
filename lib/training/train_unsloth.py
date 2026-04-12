#!/usr/bin/env python3
"""TuneLLM Unified Training Script — powered by Unsloth.

Handles all models (Qwen, Llama, Phi, Mistral) with automatic chat template detection.
Supports LoRA/QLoRA fine-tuning with 2x speed and 70% less VRAM.

Config is loaded from:
  - JOB_CONFIG environment variable (JSON string)
  - Or a JSON/YAML file via --config flag

Progress is reported via stdout markers that the agent parses:
  __PROGRESS__:{"step":N,"total_steps":M,"loss":X,"epoch":Y}
  __BASELINE_METRICS__:{...}
  __FINAL_METRICS__:{...}
"""

import json
import logging
import os
import sys
import time

# Ensure local imports resolve regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tunellm-trainer")

# --- Model → Chat Template Mapping ---

TEMPLATE_MAP = {
    "qwen": "chatml",
    "llama": "llama-3",
    "phi": "phi-3",
    "mistral": "mistral",
    "tinyllama": "chatml",
    "gemma": "gemma",
    "deepseek": "chatml",
    "yi": "chatml",
}


def detect_chat_template(model_name: str) -> str:
    """Auto-detect the chat template from the model name."""
    name_lower = model_name.lower()
    for key, template in TEMPLATE_MAP.items():
        if key in name_lower:
            return template
    # Default to chatml (works for most instruction-tuned models)
    return "chatml"


def load_config() -> dict:
    """Load training config from env var or file."""
    # Try env var first
    config_str = os.environ.get("JOB_CONFIG")
    if config_str:
        return json.loads(config_str)

    # Try CLI arg
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
        if config_path == "env":
            config_str = os.environ.get("JOB_CONFIG", "{}")
            return json.loads(config_str)
        with open(config_path) as f:
            if config_path.endswith((".yaml", ".yml")):
                import yaml
                return yaml.safe_load(f)
            return json.load(f)

    raise ValueError("No config provided. Set JOB_CONFIG env var or use --config <path>")


# --- Progress Callback ---

class ProgressCallback:
    """HuggingFace Trainer callback that prints structured progress for the agent to parse."""

    def __init__(self, total_steps: int = 0):
        self.total_steps = total_steps
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        progress = {
            "step": state.global_step,
            "total_steps": state.max_steps or self.total_steps,
            "epoch": round(state.epoch or 0, 2),
            "loss": round(logs.get("loss", logs.get("train_loss", 0)), 4),
            "learning_rate": logs.get("learning_rate", 0),
            "elapsed_seconds": round(time.time() - self.start_time, 1),
        }
        if "eval_loss" in logs:
            progress["eval_loss"] = round(logs["eval_loss"], 4)
        # Print marker for agent to parse
        print(f"__PROGRESS__:{json.dumps(progress)}", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        logger.info(f"Training started: {state.max_steps} steps, {args.num_train_epochs} epochs")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        logger.info(f"Training finished in {elapsed:.1f}s ({state.global_step} steps)")


def make_trainer_callback():
    """Create a TrainerCallback class compatible with HF Trainer."""
    from transformers import TrainerCallback

    class UnslothProgressCallback(TrainerCallback):
        def __init__(self):
            self._cb = ProgressCallback()

        def on_log(self, args, state, control, logs=None, **kwargs):
            self._cb.on_log(args, state, control, logs, **kwargs)

        def on_train_begin(self, args, state, control, **kwargs):
            self._cb.on_train_begin(args, state, control, **kwargs)

        def on_train_end(self, args, state, control, **kwargs):
            self._cb.on_train_end(args, state, control, **kwargs)

    return UnslothProgressCallback()


# --- Dataset Formatting ---

def format_dataset_for_training(dataset, tokenizer, config):
    """Format dataset rows into chat-template-formatted text for SFTTrainer."""

    def format_row(row):
        # Handle Alpaca format: instruction, input, output
        if "instruction" in row and "output" in row:
            messages = []
            user_msg = row["instruction"]
            if row.get("input"):
                user_msg += f"\n\nContext: {row['input']}"
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": row["output"]})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        # Handle chat format: messages list
        if "messages" in row:
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        # Handle prompt/completion format
        if "prompt" in row and "completion" in row:
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        # Handle question/answer format (e.g. SQuAD, GSM8K)
        if "question" in row and "answer" in row:
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        # Raw text
        if "text" in row:
            return {"text": row["text"]}

        # Fallback: concatenate all string values
        text = " ".join(str(v) for v in row.values() if isinstance(v, str))
        return {"text": text}

    return dataset.map(format_row, remove_columns=dataset.column_names)


# --- Evaluation ---

def run_evaluation(model, tokenizer, test_path, max_samples=50):
    """Run evaluation on a test dataset and return metrics."""
    from datasets import load_dataset

    logger.info(f"Running evaluation on {test_path} (max {max_samples} samples)...")

    test_ds = load_dataset("json", data_files=test_path, split="train")
    if len(test_ds) > max_samples:
        test_ds = test_ds.select(range(max_samples))

    predictions = []
    references = []

    # Enable inference mode for faster generation
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()

    import torch

    for row in test_ds:
        # Build prompt
        if "instruction" in row:
            user_msg = row["instruction"]
            if row.get("input"):
                user_msg += f"\n\nContext: {row['input']}"
            reference = row.get("output", "")
        elif "question" in row:
            user_msg = row["question"]
            reference = row.get("answer", "")
        else:
            continue

        messages = [{"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        predictions.append(answer)
        references.append(reference)

    # Compute metrics
    metrics = {}

    # F1 and Exact Match
    f1_scores = []
    em_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            em_scores.append(float(pred.strip().lower() == ref.strip().lower()))
            continue
        common = pred_tokens & ref_tokens
        if not common:
            f1_scores.append(0.0)
        else:
            p = len(common) / len(pred_tokens)
            r = len(common) / len(ref_tokens)
            f1_scores.append(2 * p * r / (p + r))
        em_scores.append(float(pred.strip().lower() == ref.strip().lower()))

    metrics["f1"] = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0
    metrics["exact_match"] = round(sum(em_scores) / len(em_scores), 4) if em_scores else 0

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge1_scores = []
        rougeL_scores = []
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)
        metrics["rouge1"] = round(sum(rouge1_scores) / len(rouge1_scores), 4)
        metrics["rougeL"] = round(sum(rougeL_scores) / len(rougeL_scores), 4)
    except ImportError:
        logger.warning("rouge-score not installed, skipping ROUGE")

    logger.info(f"Evaluation results: {metrics}")
    return metrics


# --- Main Training Function ---

def main():
    config = load_config()

    base_model = config["base_model"]
    method = config.get("method", "lora")
    run_name = config.get("run_name", "tunellm-run")
    training_config = config.get("training", {})
    lora_config = config.get("lora", {})
    dataset_config = config.get("dataset", {})

    logger.info(f"=== TuneLLM Training (Unsloth) ===")
    logger.info(f"Model: {base_model}")
    logger.info(f"Method: {method}")
    logger.info(f"Run: {run_name}")

    # --- 1. Load model with Unsloth ---
    print("__PHASE__:loading_model", flush=True)

    from unsloth import FastLanguageModel
    import torch

    max_seq_length = training_config.get("max_seq_length", 2048)
    load_in_4bit = method in ("qlora", "lora")  # 4-bit for memory efficiency

    logger.info(f"Loading model: {base_model} (4bit={load_in_4bit}, max_seq={max_seq_length})")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
    )

    # --- 2. Apply chat template ---
    template = detect_chat_template(base_model)
    logger.info(f"Chat template: {template}")

    try:
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template=template)
    except Exception as e:
        logger.warning(f"Failed to apply unsloth chat template '{template}': {e}")
        # Fallback: tokenizer may already have a chat template
        if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
            logger.warning("No chat template available, using basic formatting")

    # --- 3. Apply LoRA ---
    lora_r = lora_config.get("r", 16)
    lora_alpha = lora_config.get("alpha", 32)
    lora_dropout = lora_config.get("dropout", 0)

    target_modules = lora_config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    logger.info(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, modules={target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # --- 4. Load dataset ---
    dataset_source = dataset_config.get("source", "")
    if not dataset_source:
        raise ValueError("No dataset source in config. Set config.dataset.source")

    print("__PHASE__:downloading_data", flush=True)
    logger.info(f"Loading dataset: {dataset_source}")

    from datasets import load_dataset

    if dataset_source.startswith("hf://"):
        raw_dataset = load_dataset(dataset_source[5:], split="train")
    elif os.path.exists(dataset_source):
        raw_dataset = load_dataset("json", data_files=dataset_source, split="train")
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_source}")

    logger.info(f"Dataset loaded: {len(raw_dataset)} samples")

    # Format with chat template
    formatted_dataset = format_dataset_for_training(raw_dataset, tokenizer, config)
    logger.info(f"Dataset formatted. Sample: {formatted_dataset[0]['text'][:200]}...")

    # --- 5. Pre-training evaluation ---
    if config.get("evaluate_before") and config.get("test_dataset"):
        print("__PHASE__:evaluating_base", flush=True)
        test_path = config["test_dataset"]
        if os.path.exists(test_path):
            baseline = run_evaluation(model, tokenizer, test_path)
            print(f"__BASELINE_METRICS__:{json.dumps(baseline)}", flush=True)
            # Switch back to training mode
            model.train()

    # --- 6. Training ---
    from trl import SFTTrainer
    from transformers import TrainingArguments

    epochs = training_config.get("epochs", training_config.get("num_epochs", 3))
    batch_size = training_config.get("batch_size", 2)
    grad_accum = training_config.get("gradient_accumulation_steps", 4)
    lr = training_config.get("learning_rate", 2e-4)
    warmup_steps = training_config.get("warmup_steps", 10)
    logging_steps = training_config.get("logging_steps", 5)
    mixed_precision = training_config.get("mixed_precision", "bf16")

    output_dir = config.get("output_dir", f"/workspace/outputs/{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    use_bf16 = mixed_precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = mixed_precision == "fp16" and not use_bf16

    logger.info(f"Training: {epochs} epochs, batch={batch_size}, lr={lr}, grad_accum={grad_accum}")

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=logging_steps,
        optim="adamw_8bit",
        weight_decay=training_config.get("weight_decay", 0.01),
        lr_scheduler_type=training_config.get("lr_scheduler", "linear"),
        seed=training_config.get("seed", 42),
        output_dir=output_dir,
        report_to="none",
        save_strategy="no",  # Save manually at the end
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        callbacks=[make_trainer_callback()],
    )

    print("__PHASE__:training", flush=True)
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Loss: {train_result.training_loss:.4f}")

    # --- 7. Save model ---
    save_path = os.path.join(output_dir, "final")
    print("__PHASE__:saving_model", flush=True)
    logger.info(f"Saving adapter to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # --- 8. Post-training evaluation ---
    if config.get("evaluate_after") and config.get("test_dataset"):
        print("__PHASE__:evaluating_final", flush=True)
        test_path = config["test_dataset"]
        if os.path.exists(test_path):
            final = run_evaluation(model, tokenizer, test_path)
            print(f"__FINAL_METRICS__:{json.dumps(final)}", flush=True)

    logger.info("=== Training complete ===")
    print(f"__TRAINING_COMPLETE__:{json.dumps({'loss': round(train_result.training_loss, 4), 'output_dir': save_path})}", flush=True)


if __name__ == "__main__":
    main()
