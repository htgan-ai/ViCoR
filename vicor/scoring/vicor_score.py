from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class ViCoRResult:
    score: float
    rho: float              # 模态不平衡比 (E_shortcut + eps) / (E_delta + eps)
    nov: float              # 正交新颖性 1 - cos(g_delta, g_shortcut)
    E_delta: float          # 视觉增量能量 ||g_delta||_2
    E_shortcut: float       # 文本捷径能量 ||g_shortcut||_2
    norm_joint: float
    norm_shortcut: float
    norm_delta: float       # 原始 ||g_delta||_2 (未归一化)
    seq_len: int            # token 序列长度 L
    # 各候选得分（用于分析）
    score_delta: float = 0.0
    score_delta_log1p_rho: float = 0.0
    score_delta_nov: float = 0.0
    score_delta_log1p_rho_nov: float = 0.0
    score_mode: str = "delta_log1p_rho_nov"


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    if a.shape != b.shape:
        raise ValueError(f"cosine inputs must have identical shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("cosine inputs contain NaN or Inf")
    denom = float((torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + eps).item())
    value = float(torch.dot(a, b).item() / denom)
    if not math.isfinite(value):
        raise ValueError("cosine similarity is non-finite")
    return value


def vicor_score(
    g_joint: torch.Tensor,
    g_shortcut: torch.Tensor,
    seq_len: int = 1,
    eps: float = 1e-8,
    rho_clip: tuple[float, float] = (0.05, 20.0),  # [~0.05, ~20.0]
    E_delta_floor: float = 2.5e-5,
    score_mode: str = "delta_log1p_rho_nov",
) -> ViCoRResult:
    """
    ViCoR Score — 严格实现最终论文公式

    Step 1: 因果梯度解耦
        g_delta = g_joint - g_shortcut          # 差分，非正交投影
        g_delta 代表视觉模态引入的净更新量

    Step 2: 梯度范数诊断指标
        E_delta    = ||g_delta||_2
        E_shortcut = ||g_shortcut||_2
        rho = (E_shortcut + eps) / (E_delta + eps)  # 模态不平衡比
        nu  = 1 - cos(g_delta, g_shortcut)      # 正交新颖性（与v3的区别：对比g_delta而非g_joint）

    Step 3: 校准评分
        S_ViCoR = E_delta * log(1 + clip(rho)) * max(0, nu)

    修复说明（相比v3）:
    - [BUG FIX] g_delta 改为差分 g_joint - g_shortcut，而非Gram-Schmidt正交投影
    - scoring loss 已在 model wrapper 中按最终答案 token 平均，梯度范数直接对应论文 E_*。
    - [BUG FIX] nov 改为 cos(g_delta, g_shortcut)，语义更正确（视觉增量与捷径的正交性）
    - score 使用 E_delta，且不再对已经 token-平均的梯度重复除以 sqrt(L)
    - [NEW] E_delta_floor 阈值：视觉冗余样本直接score=0，避免噪声污染高分区
    """
    if g_joint.ndim != 1 or g_shortcut.ndim != 1:
        raise ValueError("ViCoR gradients must be one-dimensional flattened vectors")
    if g_joint.shape != g_shortcut.shape:
        raise ValueError(
            f"ViCoR gradients must have identical shapes, got {tuple(g_joint.shape)} and {tuple(g_shortcut.shape)}"
        )
    if not torch.isfinite(g_joint).all() or not torch.isfinite(g_shortcut).all():
        raise ValueError("ViCoR gradients contain NaN or Inf")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if not math.isfinite(float(E_delta_floor)) or E_delta_floor < 0:
        raise ValueError(f"E_delta_floor must be finite and non-negative, got {E_delta_floor}")
    if len(rho_clip) != 2 or not all(math.isfinite(float(x)) for x in rho_clip) or rho_clip[0] <= 0 or rho_clip[0] > rho_clip[1]:
        raise ValueError(f"rho_clip must be finite with 0 < min <= max, got {rho_clip}")
    L = max(1, seq_len)

    # ── Step 1: 差分解耦（论文: g_delta = g_joint - g_shortcut） ────────
    g_delta = g_joint - g_shortcut

    # ── Step 2: 计算论文 Eq. (4) 中的梯度范数 ─────────────────────────
    nj = float(torch.linalg.vector_norm(g_joint).item())
    ns = float(torch.linalg.vector_norm(g_shortcut).item())
    nd = float(torch.linalg.vector_norm(g_delta).item())

    # Eq. (1) already averages the final-answer token loss.  The final PDF
    # defines E_delta and E_shortcut as the corresponding gradient norms;
    # applying another length normalization would change the selector.
    E_delta    = nd
    E_shortcut = ns

    # Keep raw statistics even below the gate: Appendix C uses the raw ratio
    # and novelty in the diversity descriptor, independently of the score.
    rho = (E_shortcut + eps) / (E_delta + eps)
    rho_weight = float(max(rho_clip[0], min(rho, rho_clip[1])))

    # ── 正交新颖性 nu（论文: 1 - cos(g_delta, g_shortcut)） ────────────
    # nu≈0: 视觉增量与捷径共线 → 复读/加强语言先验（视觉冗余）
    # nu大: 视觉增量正交或反向于捷径 → 视觉纠错
    cos_ds = cosine_sim(g_delta, g_shortcut, eps=eps)
    nov = float(max(0.0, 1.0 - cos_ds))

    # ── Step 3: 校准评分（论文: S_ViCoR = E_delta * log(1+rho) * nu） ─────
    log1p_rho = math.log1p(rho_weight)

    score_delta               = E_delta
    score_delta_log1p_rho     = E_delta * log1p_rho
    score_delta_nov           = E_delta * nov
    score_delta_log1p_rho_nov = E_delta * log1p_rho * nov

    if not all(math.isfinite(value) for value in (rho, nov, score_delta, score_delta_log1p_rho, score_delta_nov, score_delta_log1p_rho_nov)):
        raise ValueError("ViCoR score statistics became non-finite")

    score_map = {
        "delta":               score_delta,
        "delta_log1p_rho":     score_delta_log1p_rho,
        "delta_nov":           score_delta_nov,
        "delta_log1p_rho_nov": score_delta_log1p_rho_nov,
    }
    if score_mode not in score_map:
        available = ", ".join(sorted(score_map.keys()))
        raise ValueError(f"Unknown score_mode={score_mode!r}. Available: {available}")

    if E_delta < E_delta_floor:
        score_delta = score_delta_log1p_rho = score_delta_nov = score_delta_log1p_rho_nov = 0.0
        score = 0.0
    else:
        score = float(score_map[score_mode])

    return ViCoRResult(
        score=score,
        rho=rho,
        nov=nov,
        E_delta=E_delta,
        E_shortcut=E_shortcut,
        norm_joint=nj,
        norm_shortcut=ns,
        norm_delta=nd,
        seq_len=L,
        score_delta=score_delta,
        score_delta_log1p_rho=score_delta_log1p_rho,
        score_delta_nov=score_delta_nov,
        score_delta_log1p_rho_nov=score_delta_log1p_rho_nov,
        score_mode=score_mode,
    )
