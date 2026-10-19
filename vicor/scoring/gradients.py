from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import torch
import torch.nn as nn

@dataclass
class GradStats:
    grad_vec: torch.Tensor      # [d] flattened (float32 on cpu)
    grad_norm: float
    loss: float


@dataclass
class CountSketchProjector:
    buckets: torch.Tensor
    signs: torch.Tensor
    sketch_dim: int
    seed: int

def _flatten_grads(params: Iterable[nn.Parameter]) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is None:
            # If no grad, treat as zero vector
            grads.append(torch.zeros_like(p, dtype=torch.float32).view(-1))
        else:
            grads.append(p.grad.detach().to(dtype=torch.float32).view(-1))
    return torch.cat(grads, dim=0)


def build_count_sketch_projector(
    input_dim: int,
    sketch_dim: int,
    seed: int = 17,
) -> CountSketchProjector:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be > 0, got {input_dim}")
    if sketch_dim <= 0:
        raise ValueError(f"sketch_dim must be > 0, got {sketch_dim}")

    idx = torch.arange(input_dim, dtype=torch.int64)
    seed1 = int(seed) * 1103515245 + 12345
    seed2 = int(seed) * 214013 + 2531011
    buckets = torch.remainder(idx * 6364136223846793005 + seed1, sketch_dim).to(torch.int64)
    sign_bits = torch.remainder(idx * 1442695040888963407 + seed2, 2).to(torch.int64)
    signs = sign_bits.mul(2).sub(1).to(torch.float32)
    return CountSketchProjector(
        buckets=buckets,
        signs=signs,
        sketch_dim=int(sketch_dim),
        seed=int(seed),
    )


def apply_count_sketch(
    vec: torch.Tensor,
    projector: CountSketchProjector,
    normalize: bool = True,
) -> torch.Tensor:
    if vec.ndim != 1:
        raise ValueError(f"vec must be 1-D, got shape {tuple(vec.shape)}")
    if vec.numel() != projector.buckets.numel():
        raise ValueError(
            f"vec size {vec.numel()} does not match projector input size {projector.buckets.numel()}"
        )

    sketch = torch.zeros(projector.sketch_dim, dtype=torch.float32)
    sketch.scatter_add_(0, projector.buckets, vec.to(torch.float32) * projector.signs)
    if normalize:
        norm = torch.linalg.vector_norm(sketch)
        if float(norm.item()) > 1e-8:
            sketch = sketch / norm
    return sketch

@torch.enable_grad()
def compute_grad_stats(
    model: nn.Module,
    shared_params: Tuple[nn.Parameter, ...],
    loss: torch.Tensor,
) -> GradStats:
    """
    计算给定 loss 对 shared_params 的梯度统计量。
    注意：此函数会执行 loss.backward()，请确保显存足够。
    建议在 eval 模式下调用，避免 dropout 带来的随机性。
    """
    # 1. Clear existing gradients
    for p in shared_params:
        if p.grad is not None:
            p.grad = None

    # 2. Backward pass
    # retain_graph=False (default) as we don't need it for second pass usually
    # unless called consecutively on same graph, but we usually re-forward.
    loss.backward()

    # 3. Collect gradients
    vec = _flatten_grads(shared_params).cpu()
    
    # 4. Compute norm
    norm = float(torch.linalg.vector_norm(vec).item())
    
    # 5. Detach loss for logging
    loss_val = float(loss.detach().cpu().item())

    return GradStats(grad_vec=vec, grad_norm=norm, loss=loss_val)
