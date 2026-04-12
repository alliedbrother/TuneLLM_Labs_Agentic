"""DPO (Direct Preference Optimization) Trainer.

DPO is a reinforcement learning from human feedback (RLHF) method
that directly optimizes the model on preference data without
requiring a separate reward model.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer as TRLDPOTrainer

from utils.data_loader import DPODataLoader, load_dataset_from_config
from utils.metrics import MetricsTracker
from utils.model_utils import (
    count_trainable_parameters,
    get_tokenizer,
    load_base_model,
    save_model,
)

logger = logging.getLogger(__name__)


class DPOTrainer:
    """DPO (Direct Preference Optimization) fine-tuning trainer.

    Uses preference data (chosen vs rejected responses) to align
    the model with human preferences.
    """

    def __init__(self, config: dict, output_dir: str):
        """Initialize the DPO trainer.

        Args:
            config: Training configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract configuration
        self.base_model = config["base_model"]
        self.run_name = config.get("run_name", "dpo-finetune")

        # Training hyperparameters
        training_config = config.get("training", {})
        self.num_epochs = training_config.get("num_epochs", 1)
        self.batch_size = training_config.get("batch_size", 4)
        self.gradient_accumulation_steps = training_config.get(
            "gradient_accumulation_steps", 4
        )
        self.learning_rate = training_config.get("learning_rate", 5e-7)
        self.warmup_ratio = training_config.get("warmup_ratio", 0.1)
        self.max_length = training_config.get("max_length", 1024)
        self.max_prompt_length = training_config.get("max_prompt_length", 512)

        # DPO-specific configuration
        dpo_config = config.get("dpo", {})
        self.beta = dpo_config.get("beta", 0.1)  # KL penalty coefficient
        self.loss_type = dpo_config.get("loss_type", "sigmoid")  # 'sigmoid' or 'hinge'
        self.label_smoothing = dpo_config.get("label_smoothing", 0.0)

        # LoRA configuration
        lora_config = config.get("lora", {})
        self.use_lora = lora_config.get("enabled", True)
        self.lora_r = lora_config.get("r", 16)
        self.lora_alpha = lora_config.get("alpha", 32)
        self.lora_dropout = lora_config.get("dropout", 0.05)
        self.lora_target_modules = lora_config.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        # Model settings
        model_config = config.get("model", {})
        self.trust_remote_code = model_config.get("trust_remote_code", False)
        self.quantization = config.get("quantization")

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.dataset = None
        self.metrics_tracker = None

    def setup(self) -> None:
        """Setup model, tokenizer, and dataset."""
        logger.info("Setting up DPO trainer...")

        # Load tokenizer
        self.tokenizer = get_tokenizer(
            self.base_model,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",  # DPO requires left padding
        )

        # Load model
        self.model = load_base_model(
            self.base_model,
            quantization=self.quantization,
            trust_remote_code=self.trust_remote_code,
        )

        # Prepare for k-bit training if quantized
        if self.quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA if enabled
        if self.use_lora:
            self._apply_lora()

        # Load reference model (frozen copy for KL divergence)
        # For efficiency, we can share the base model if using LoRA
        if self.use_lora:
            logger.info("Using base model as reference (LoRA mode)")
            self.ref_model = None  # TRL DPO trainer handles this
        else:
            logger.info("Loading separate reference model")
            self.ref_model = load_base_model(
                self.base_model,
                quantization=self.quantization,
                trust_remote_code=self.trust_remote_code,
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

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

    def _prepare_dataset(self) -> None:
        """Load and prepare the preference dataset."""
        logger.info("Preparing DPO dataset...")

        # Load dataset
        raw_dataset = load_dataset_from_config(self.config)

        # Validate DPO format
        dpo_loader = DPODataLoader(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        self.dataset = dpo_loader.prepare_dataset(raw_dataset)

        logger.info(f"Dataset prepared: {len(self.dataset)} preference pairs")

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
        """Run the DPO training loop."""
        self.setup()

        logger.info("Starting DPO training...")

        # DPO training config
        dpo_config = DPOConfig(
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=500 if self.eval_dataset else None,
            load_best_model_at_end=bool(self.eval_dataset),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to=["wandb"] if self.config.get("use_wandb") else [],
            # DPO-specific settings
            beta=self.beta,
            loss_type=self.loss_type,
            label_smoothing=self.label_smoothing,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
        )

        # Initialize DPO trainer
        trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
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
