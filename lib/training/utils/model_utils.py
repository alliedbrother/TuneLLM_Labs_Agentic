"""Model loading and saving utilities."""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def get_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
    padding_side: str = "right",
) -> PreTrainedTokenizer:
    """Load tokenizer for a model.

    Args:
        model_name: HuggingFace model name or path
        trust_remote_code: Whether to trust remote code
        padding_side: Padding side for the tokenizer

    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def load_base_model(
    model_name: str,
    quantization: Optional[str] = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
) -> PreTrainedModel:
    """Load base model for fine-tuning.

    Args:
        model_name: HuggingFace model name or path
        quantization: Quantization method ('4bit', '8bit', or None)
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        torch_dtype: Torch data type for model weights
        attn_implementation: Attention implementation ('flash_attention_2', 'sdpa', etc.)

    Returns:
        Loaded model
    """
    logger.info(f"Loading base model: {model_name}")

    # Determine torch dtype
    if torch_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

    # Configure quantization
    quantization_config = None
    if quantization == "4bit":
        logger.info("Using 4-bit quantization (QLoRA)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Build model kwargs
    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    return model


def load_model_for_inference(
    model_name: str,
    adapter_path: Optional[str] = None,
    quantization: Optional[str] = None,
    device_map: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model for inference, optionally with LoRA adapter.

    Args:
        model_name: Base model name or path
        adapter_path: Path to LoRA adapter (optional)
        quantization: Quantization method
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = get_tokenizer(model_name)
    model = load_base_model(
        model_name,
        quantization=quantization,
        device_map=device_map,
    )

    if adapter_path:
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info("Adapter merged with base model")

    model.eval()
    return model, tokenizer


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    save_full_model: bool = False,
) -> None:
    """Save model and tokenizer.

    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Output directory
        save_full_model: If True, merge LoRA and save full model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to: {output_dir}")

    # Check if it's a PEFT model
    is_peft = hasattr(model, "peft_config")

    if is_peft and not save_full_model:
        # Save only the adapter
        model.save_pretrained(output_dir)
        logger.info("Saved LoRA adapter")
    elif is_peft and save_full_model:
        # Merge and save full model
        logger.info("Merging adapter with base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        logger.info("Saved merged model")
    else:
        # Save regular model
        model.save_pretrained(output_dir, safe_serialization=True)
        logger.info("Saved model")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved tokenizer")


def get_model_size_gb(model: PreTrainedModel) -> float:
    """Get model size in GB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024**3)


def count_trainable_parameters(model: PreTrainedModel) -> Tuple[int, int, float]:
    """Count trainable and total parameters.

    Returns:
        Tuple of (trainable_params, all_params, trainable_percentage)
    """
    trainable_params = 0
    all_params = 0

    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0

    return trainable_params, all_params, trainable_percent
