"""LoRA Fine-tuning Trainer."""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.data_loader import DataLoader, load_dataset_from_config
from utils.metrics import MetricsTracker, compute_metrics
from utils.model_utils import (
    count_trainable_parameters,
    get_tokenizer,
    load_base_model,
    save_model,
)

logger = logging.getLogger(__name__)


class LoRATrainer:
    """LoRA (Low-Rank Adaptation) fine-tuning trainer."""

    def __init__(self, config: dict, output_dir: str):
        """Initialize the LoRA trainer.

        Args:
            config: Training configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract configuration
        self.base_model = config["base_model"]
        self.run_name = config.get("run_name", "lora-finetune")

        # Training hyperparameters
        training_config = config.get("training", {})
        self.num_epochs = training_config.get("num_epochs", training_config.get("epochs", 3))
        self.batch_size = training_config.get("batch_size", 4)
        self.gradient_accumulation_steps = training_config.get(
            "gradient_accumulation_steps", 4
        )
        self.learning_rate = training_config.get("learning_rate", 2e-4)
        self.warmup_ratio = training_config.get("warmup_ratio", 0.03)
        self.max_length = training_config.get("max_length", 2048)
        self.weight_decay = training_config.get("weight_decay", 0.01)
        self.lr_scheduler_type = training_config.get("lr_scheduler_type", "cosine")

        # LoRA configuration
        lora_config = config.get("lora", {})
        self.lora_r = lora_config.get("r", 16)
        self.lora_alpha = lora_config.get("alpha", 32)
        self.lora_dropout = lora_config.get("dropout", 0.05)
        self.lora_target_modules = lora_config.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        # Model settings
        model_config = config.get("model", {})
        self.trust_remote_code = model_config.get("trust_remote_code", False)
        self.attn_implementation = model_config.get("attn_implementation")

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.metrics_tracker = None

    def setup(self) -> None:
        """Setup model, tokenizer, and dataset."""
        logger.info("Setting up LoRA trainer...")

        # Load tokenizer
        self.tokenizer = get_tokenizer(
            self.base_model,
            trust_remote_code=self.trust_remote_code,
        )

        # Load base model
        quantization = self.config.get("quantization")
        self.model = load_base_model(
            self.base_model,
            quantization=quantization,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation,
        )

        # Prepare for k-bit training if quantized
        if quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        # Check if this is full fine-tuning (no LoRA)
        if self.config.get("lora") is None:
            logger.info("Full fine-tuning mode (no LoRA)")
        else:
            # Apply LoRA
            self._apply_lora()

        # Load and prepare dataset
        self._prepare_dataset()

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            use_wandb=self.config.get("use_wandb", False),
            wandb_project=self.config.get("wandb_project"),
        )

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the model."""
        logger.info("Applying LoRA adapters...")

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)

        # Log trainable parameters
        trainable, total, percent = count_trainable_parameters(self.model)
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({percent:.2f}%)")

        self.model.print_trainable_parameters()

    def _prepare_dataset(self) -> None:
        """Load and prepare the training dataset."""
        logger.info("Preparing dataset...")

        # Load dataset
        raw_dataset = load_dataset_from_config(self.config)

        # Create data loader
        data_loader = DataLoader(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            prompt_template=self.config.get("dataset", {}).get("prompt_template"),
        )

        # Prepare dataset
        self.dataset = data_loader.prepare_dataset(raw_dataset)
        logger.info(f"Dataset prepared: {len(self.dataset)} examples")

        # Split into train/eval if needed
        eval_split = self.config.get("dataset", {}).get("eval_split", 0.1)
        if eval_split > 0:
            split = self.dataset.train_test_split(test_size=eval_split, seed=42)
            self.train_dataset = split["train"]
            self.eval_dataset = split["test"]
            logger.info(
                f"Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}"
            )
        else:
            self.train_dataset = self.dataset
            self.eval_dataset = None

    def train(self) -> None:
        """Run the training loop."""
        self.setup()

        logger.info("Starting training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=500 if self.eval_dataset else None,
            load_best_model_at_end=bool(self.eval_dataset),
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            greater_is_better=False,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to=["wandb"] if self.config.get("use_wandb") else [],
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics if self.eval_dataset else None,
        )

        # Train
        trainer.train()

        # Save final model
        logger.info("Saving final model...")
        save_model(
            self.model,
            self.tokenizer,
            str(self.output_dir / "final"),
            save_full_model=self.config.get("save_full_model", False),
        )

        # Finish metrics tracking
        self.metrics_tracker.finish()

        logger.info(f"Training complete! Model saved to {self.output_dir / 'final'}")
