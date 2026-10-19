from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import torch
import torch.nn as nn

@dataclass
class ForwardOutput:
    loss: torch.Tensor          # scalar
    logits: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, Any]] = None

class ModelWrapper(nn.Module):
    """
    统一接口：
    - forward_loss(batch, mode="joint"|"shortcut") 返回 loss
    - get_shared_params() 返回 Θ_shared（通常是 LoRA 参数）
    """
    def __init__(self):
        super().__init__()

    def forward_loss(self, batch: Dict[str, Any], mode: str) -> ForwardOutput:
        """
        mode:
        - "joint": 正常前向传播 (text + image)
        - "shortcut": 保留多模态模板、将图像替换为 processor-mean 图像
        """
        raise NotImplementedError

    def get_shared_params(self) -> Tuple[nn.Parameter, ...]:
        """
        返回需要计算梯度的共享参数列表 (e.g. LoRA params)
        """
        raise NotImplementedError
