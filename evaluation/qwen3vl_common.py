#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import torch
from peft import PeftConfig, PeftModel


def apply_chat_template_compat(processor: Any, conversation: list[dict], add_generation_prompt: bool = True) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    image_token = getattr(processor, "image_token", "<image>")
    converted_messages: list[dict[str, str]] = []
    for msg in conversation:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if item.get("type") == "image":
                    parts.append(image_token)
                elif item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            content = "\n".join([p for p in parts if p]).strip()
        converted_messages.append({"role": msg["role"], "content": str(content)})
    if tokenizer is not None and image_token == "<IMG_CONTEXT>":
        return tokenizer.apply_chat_template(converted_messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:
        if tokenizer is None:
            raise
        return tokenizer.apply_chat_template(converted_messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def load_peft_vlm(
    adapter_path: str = "",
    base_model_path: str = "",
    device: str = "cuda:0",
    image_max_pixels: int = 262144,
    image_min_pixels: int = 50176,
) -> tuple[Any, torch.nn.Module, str]:
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    if not adapter_path and not base_model_path:
        raise ValueError("Provide base_model_path for a base-model audit or adapter_path for a fine-tuned model")
    if not base_model_path:
        base_model_path = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path

    cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    if not getattr(processor, "chat_template", None) and getattr(processor, "tokenizer", None) is not None:
        ct = getattr(processor.tokenizer, "chat_template", None)
        if ct:
            processor.chat_template = ct
    if hasattr(processor, "image_processor"):
        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = image_max_pixels
        if hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = image_min_pixels

    if model_type != "qwen3_vl":
        raise ValueError(
            f"This release package supports only Qwen3-VL checkpoints; got model_type={model_type!r}"
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    processor = Qwen3VLProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = image_max_pixels
        if hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = image_min_pixels
    backbone = "qwen3_vl"

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"
    return processor, model, backbone


def run_chat_generation_generic(model: torch.nn.Module, processor: Any, conversation: list[dict], images: list[Any], device: str, max_new_tokens: int) -> str:
    prompt = apply_chat_template_compat(processor, conversation, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, padding=True, return_tensors="pt")
    if "pixel_values" in inputs:
        try:
            model_dtype = next(model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        except Exception:
            pass
    inputs = inputs.to(device)
    eos_token_id = getattr(processor.tokenizer, 'eos_token_id', None) if getattr(processor, 'tokenizer', None) is not None else None
    pad_token_id = getattr(processor.tokenizer, 'pad_token_id', None) if getattr(processor, 'tokenizer', None) is not None else None
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    gen_ids = output_ids[:, input_len:]
    decoder = processor if hasattr(processor, 'batch_decode') else processor.tokenizer
    text = decoder.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return text.strip()
