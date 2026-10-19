from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import get_peft_model, LoraConfig, TaskType

from .base import ModelWrapper, ForwardOutput

class Qwen3VLWrapper(ModelWrapper):
    def __init__(self, model_path: str, lora_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = Qwen3VLProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Apply LoRA if config provided
        if lora_config:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False, 
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.05),
                target_modules=lora_config.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def forward_loss(self, batch: Dict[str, Any], mode: str) -> ForwardOutput:
        # batch 应该包含 processor 处理后的 input_ids, attention_mask, pixel_values 等
        
        if mode == "shortcut":
            # Dataset-level mean control, preserving the multimodal input format
            input_ids = batch["shortcut_input_ids"]
            attention_mask = batch["shortcut_attention_mask"]
            labels = batch["shortcut_labels"]
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=batch["shortcut_pixel_values"],
                image_grid_thw=batch["shortcut_image_grid_thw"],
                use_cache=False,
            )
        else:
            # Joint mode: standard forward
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            pixel_values = batch.get("pixel_values", None)
            image_grid_thw = batch.get("image_grid_thw", None)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
            )
            
        # Transformers returns the mean cross-entropy over final-answer tokens,
        # matching Eq. (1) in the final PDF.
        return ForwardOutput(loss=outputs.loss)

    def get_shared_params(self) -> Tuple[nn.Parameter, ...]:
        # Return trainable params (LoRA params)
        return tuple(p for p in self.model.parameters() if p.requires_grad)

def build_qwen3_wrapper(model_path: str, lora_cfg: Optional[Dict[str, Any]] = None):
    return Qwen3VLWrapper(model_path, lora_cfg)
