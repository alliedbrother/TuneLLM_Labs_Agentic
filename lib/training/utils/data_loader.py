"""Data loading utilities for fine-tuning."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_dataset_from_config(config: dict) -> Dataset:
    """Load dataset based on configuration.

    Supports:
    - HuggingFace Hub datasets
    - Local JSON/JSONL files
    - Local CSV files
    - Local Parquet files
    """
    dataset_config = config.get("dataset", {})
    source = dataset_config.get("source", "")

    # Check if it's a HuggingFace dataset
    if dataset_config.get("hub_dataset"):
        logger.info(f"Loading dataset from HuggingFace Hub: {source}")
        ds = load_dataset(
            source,
            split=dataset_config.get("split", "train"),
            trust_remote_code=dataset_config.get("trust_remote_code", False),
        )
    elif source.startswith("hf://"):
        # HuggingFace Hub shorthand
        dataset_name = source[5:]
        logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
        ds = load_dataset(
            dataset_name,
            split=dataset_config.get("split", "train"),
        )
    else:
        # Local file
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {source}")

        logger.info(f"Loading dataset from local file: {source}")

        if path.suffix == ".json":
            ds = load_dataset("json", data_files=str(path), split="train")
        elif path.suffix == ".jsonl":
            ds = load_dataset("json", data_files=str(path), split="train")
        elif path.suffix == ".csv":
            ds = load_dataset("csv", data_files=str(path), split="train")
        elif path.suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(path), split="train")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    # Apply subset if specified
    max_samples = dataset_config.get("max_samples")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
        logger.info(f"Using {len(ds)} samples")

    return ds


class DataLoader:
    """Data loader for fine-tuning datasets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        prompt_template: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or self._default_template()

    def _default_template(self) -> str:
        """Default instruction template."""
        return (
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )

    def format_instruction(self, example: dict) -> str:
        """Format a single example using the template."""
        # Handle different dataset formats
        if "instruction" in example:
            # Alpaca-style format
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")

            return self.prompt_template.format(
                instruction=instruction,
                input=input_text,
                output=output,
            )
        elif "text" in example:
            # Raw text format
            return example["text"]
        elif "prompt" in example and "completion" in example:
            # OpenAI-style format
            return f"{example['prompt']}{example['completion']}"
        elif "messages" in example:
            # Chat format
            return self._format_chat_messages(example["messages"])
        else:
            raise ValueError(f"Unknown dataset format. Keys: {example.keys()}")

    def _format_chat_messages(self, messages: list) -> str:
        """Format chat messages into a single string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"### System:\n{content}")
            elif role == "user":
                formatted.append(f"### User:\n{content}")
            elif role == "assistant":
                formatted.append(f"### Assistant:\n{content}")
        return "\n\n".join(formatted)

    def tokenize(self, example: dict) -> dict:
        """Tokenize a single example."""
        text = self.format_instruction(example)

        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        # For causal LM, labels are the same as input_ids
        # Set padding token labels to -100 so they're ignored in loss
        labels = result["input_ids"].copy()
        if self.tokenizer.pad_token_id is not None:
            labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
        result["labels"] = labels

        return result

    def prepare_dataset(
        self,
        dataset: Dataset,
        num_proc: int = 4,
    ) -> Dataset:
        """Prepare dataset for training."""
        logger.info(f"Tokenizing {len(dataset)} examples...")

        tokenized = dataset.map(
            self.tokenize,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="Tokenizing",
        )

        return tokenized


class DPODataLoader:
    """Data loader for DPO (Direct Preference Optimization) datasets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for DPO training.

        Expected format:
        - prompt: The input prompt
        - chosen: The preferred response
        - rejected: The rejected response
        """
        required_columns = {"prompt", "chosen", "rejected"}
        if not required_columns.issubset(set(dataset.column_names)):
            raise ValueError(
                f"DPO dataset must have columns: {required_columns}. "
                f"Found: {dataset.column_names}"
            )

        return dataset


class PPODataLoader:
    """Data loader for PPO (Proximal Policy Optimization) datasets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for PPO training.

        Expected format:
        - query: The input prompt/query
        """
        if "query" not in dataset.column_names:
            # Try to convert from other formats
            if "prompt" in dataset.column_names:
                dataset = dataset.rename_column("prompt", "query")
            elif "text" in dataset.column_names:
                dataset = dataset.rename_column("text", "query")
            else:
                raise ValueError(
                    "PPO dataset must have 'query' column. "
                    f"Found: {dataset.column_names}"
                )

        return dataset
