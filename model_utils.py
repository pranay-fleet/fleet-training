"""Model loading and generation utilities for VL models."""

from typing import Dict, Any
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_qwen25_vl(
    model_id: str, 
    dtype: torch.dtype, 
    use_lora: bool, 
    lora_r: int = 16, 
    lora_alpha: int = 32, 
    lora_dropout: float = 0.05
):
    """
    Load a Qwen2.5-VL model + processor. Prefer native VL class if available.
    We keep it in FP16/BF16 per H100 capability (BF16 recommended).
    """
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    # Optional: add LoRA
    if use_lora:
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # adapt per model
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Tokenizer (some processors bundle it; still handy)
    try:
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    # Make sure special tokens exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, processor, tokenizer


@torch.no_grad()
def generate_tool_call(
    model,
    processor,
    tokenizer,
    batch: Dict[str, torch.Tensor],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
):
    """
    Runs model.generate on (images + text) and returns decoded strings (one per batch item).
    """
    # Build kwargs for generate: Qwen-VL typically expects pixel_values + input_ids/attention_mask.
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # Pixel values: processor-defined key (often "pixel_values" or similar for VL)
    pixel_values = batch.get("pixel_values", None)
    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device)

    # Some Qwen-VL models expect "images" rather than pixel_values—trust_remote_code often normalizes this.
    # We'll try the common signature:
    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **gen_kwargs,
        )
    except TypeError:
        # Fallback: drop pixel_values if model signature differs.
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts

