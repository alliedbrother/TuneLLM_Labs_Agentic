"""PPO (Proximal Policy Optimization) Trainer.

PPO is a reinforcement learning from human feedback (RLHF) method
that uses a reward model to optimize the policy.

Note: This is a stub implementation. Full PPO requires:
1. A trained reward model
2. More complex training loop with rollouts
3. Value function estimation
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import PPOConfig, PPOTrainer as TRLPPOTrainer, AutoModelForCausalLMWithValueHead

from utils.data_loader import PPODataLoader, load_dataset_from_config
from utils.metrics import MetricsTracker
from utils.model_utils import (
    count_trainable_parameters,
    get_tokenizer,
    load_base_model,
    save_model,
)

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO (Proximal Policy Optimization) fine-tuning trainer.

    Uses a reward model to optimize the policy through
    reinforcement learning.
    """

    def __init__(self, config: dict, output_dir: str):
        """Initialize the PPO trainer.

        Args:
            config: Training configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract configuration
        self.base_model = config["base_model"]
        self.reward_model = config.get("reward_model")
        self.run_name = config.get("run_name", "ppo-finetune")

        # Training hyperparameters
        training_config = config.get("training", {})
        self.num_epochs = training_config.get("num_epochs", 1)
        self.batch_size = training_config.get("batch_size", 4)
        self.mini_batch_size = training_config.get("mini_batch_size", 1)
        self.learning_rate = training_config.get("learning_rate", 1.41e-5)
        self.max_length = training_config.get("max_length", 512)

        # PPO-specific configuration
        ppo_config = config.get("ppo", {})
        self.ppo_epochs = ppo_config.get("ppo_epochs", 4)
        self.kl_penalty = ppo_config.get("kl_penalty", "kl")
        self.init_kl_coef = ppo_config.get("init_kl_coef", 0.2)
        self.target_kl = ppo_config.get("target_kl", 6.0)
        self.gamma = ppo_config.get("gamma", 1.0)
        self.lam = ppo_config.get("lam", 0.95)
        self.cliprange = ppo_config.get("cliprange", 0.2)
        self.cliprange_value = ppo_config.get("cliprange_value", 0.2)

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
        self.reward_model_fn = None
        self.dataset = None
        self.metrics_tracker = None

    def setup(self) -> None:
        """Setup model, tokenizer, and dataset."""
        logger.info("Setting up PPO trainer...")

        # Load tokenizer
        self.tokenizer = get_tokenizer(
            self.base_model,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",
        )

        # Load model with value head for PPO
        logger.info(f"Loading model with value head: {self.base_model}")

        # Load base model first
        base_model = load_base_model(
            self.base_model,
            quantization=self.quantization,
            trust_remote_code=self.trust_remote_code,
        )

        # Prepare for k-bit training if quantized
        if self.quantization:
            base_model = prepare_model_for_kbit_training(base_model)

        # Apply LoRA if enabled
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            base_model = get_peft_model(base_model, lora_config)

            trainable, total, percent = count_trainable_parameters(base_model)
            logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({percent:.2f}%)")

        # Wrap with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

        # Setup reward model/function
        self._setup_reward_model()

        # Load and prepare dataset
        self._prepare_dataset()

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            use_wandb=self.config.get("use_wandb", False),
            wandb_project=self.config.get("wandb_project"),
        )

    def _setup_reward_model(self) -> None:
        """Setup the reward model or function."""
        if self.reward_model:
            # Load reward model
            logger.info(f"Loading reward model: {self.reward_model}")
            from transformers import AutoModelForSequenceClassification

            reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_model,
                trust_remote_code=self.trust_remote_code,
            )
            reward_model.eval()
            reward_model.to("cuda" if torch.cuda.is_available() else "cpu")

            def reward_fn(texts):
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(reward_model.device)

                with torch.no_grad():
                    outputs = reward_model(**inputs)
                    rewards = outputs.logits.squeeze(-1)

                return rewards.tolist()

            self.reward_model_fn = reward_fn
        else:
            # Use a simple length-based reward as placeholder
            logger.warning(
                "No reward model specified. Using placeholder reward function. "
                "For real PPO training, please provide a reward model."
            )

            def placeholder_reward(texts):
                # Simple reward based on response length (placeholder)
                return [min(len(t) / 100, 1.0) for t in texts]

            self.reward_model_fn = placeholder_reward

    def _prepare_dataset(self) -> None:
        """Load and prepare the dataset."""
        logger.info("Preparing PPO dataset...")

        # Load dataset
        raw_dataset = load_dataset_from_config(self.config)

        # Prepare for PPO format
        ppo_loader = PPODataLoader(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        self.dataset = ppo_loader.prepare_dataset(raw_dataset)

        logger.info(f"Dataset prepared: {len(self.dataset)} queries")

    def train(self) -> None:
        """Run the PPO training loop."""
        self.setup()

        logger.info("Starting PPO training...")

        # PPO config
        ppo_config = PPOConfig(
            model_name=self.base_model,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            mini_batch_size=self.mini_batch_size,
            ppo_epochs=self.ppo_epochs,
            init_kl_coef=self.init_kl_coef,
            target_kl=self.target_kl,
            gamma=self.gamma,
            lam=self.lam,
            cliprange=self.cliprange,
            cliprange_value=self.cliprange_value,
            log_with="wandb" if self.config.get("use_wandb") else None,
        )

        # Initialize PPO trainer
        trainer = TRLPPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
        )

        # Training loop
        generation_kwargs = {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            for batch in trainer.dataloader:
                # Get query tensors
                query_tensors = batch["input_ids"]

                # Generate responses
                response_tensors = trainer.generate(
                    query_tensors,
                    **generation_kwargs,
                )

                # Decode responses
                responses = self.tokenizer.batch_decode(
                    response_tensors,
                    skip_special_tokens=True,
                )

                # Get rewards
                rewards = self.reward_model_fn(responses)
                rewards = [torch.tensor(r) for r in rewards]

                # Run PPO step
                stats = trainer.step(query_tensors, response_tensors, rewards)

                # Log metrics
                self.metrics_tracker.log({
                    "ppo/reward_mean": sum(r.item() for r in rewards) / len(rewards),
                    "ppo/policy_loss": stats.get("ppo/policy/loss", 0),
                    "ppo/value_loss": stats.get("ppo/value/loss", 0),
                    "ppo/kl": stats.get("ppo/mean_kl", 0),
                })

        # Save final model
        logger.info("Saving final model...")
        trainer.save_pretrained(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))

        # Finish metrics tracking
        self.metrics_tracker.finish()

        logger.info(f"Training complete! Model saved to {self.output_dir / 'final'}")
