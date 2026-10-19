"""
ViCoR 子集选择脚本 v5

核心设计原则：
  1. 质量优先：ViCoR Score 是主排序轴，视觉必要性强的样本优先
  2. 多样性约束：在质量排序的基础上，通过 MMR（Maximal Marginal Relevance）
     避免语义重复，防止子集塌缩为同一类型的高分样本
  3. 长尾保护：显式保留长尾答案的配额，对抗多数类垄断
  4. 策略分层：质量筛选 → 分组均衡 → MMR 多样性重排，三层独立可控

关于训练步数对齐：
  本脚本只负责数据筛选，不直接控制训练。
  对齐策略详见 configs/，核心思路是"等效 token 预算"而非"乘倍 epoch"。
"""
import argparse
import hashlib
import json
import math
import os
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vicor.config import load_config


# ─────────────────────────────────────────────────────────────────────────────
# Argument & Config
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--score_col", type=str, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["topk", "group_capped_topk", "round_robin",
                 "vicor_mmr", "paper_vicor_mmr", "paper_group_vicor_mmr", "tail_vicor_mmr", "cluster_vicor_mmr", "cluster_floor_vicor_mmr", "adaptive_tail_cluster_vicor_mmr", "label_sparse_vicor_mmr", "label_sparse_tail_vicor_mmr", "label_floor_vicor_mmr"],
        help=(
            "topk: 纯 ViCoR Score 排序\n"
            "group_capped_topk: 分组约束 topk\n"
            "round_robin: 组间轮询\n"
            "vicor_mmr: ViCoR Score 质量 + MMR 多样性重排（推荐）\n"
            "paper_vicor_mmr: 最终论文的 S_rank 校准 + γK 候选池 + MMR\n"
            "paper_group_vicor_mmr: 兼容旧配置的 group floor/cap + MMR\n"
            "tail_vicor_mmr: 长尾保护 + ViCoR Score + MMR（长尾严重时使用）\n"
            "cluster_vicor_mmr: 先做特征空间分簇，再按簇分预算执行 ViCoR+MMR（借鉴 coverage 约束）\n"
            "cluster_floor_vicor_mmr: 先给分簇一个 coverage floor，再用全局 ViCoR+MMR 填满剩余预算（更温和）\n"
            "adaptive_tail_cluster_vicor_mmr: 单一通用方法，根据答案空间开放度自适应平衡 tail 保护与 coverage floor\n"
            "label_sparse_vicor_mmr: v4b 主干 + 标签对齐 + 分类式表达 + 低标签数偏好\n"
            "label_sparse_tail_vicor_mmr: tail_vicor_mmr 主干 + 标签对齐 + 分类式表达 + 低标签数偏好\n"
            "label_floor_vicor_mmr: v7a 主干 + benchmark 标签覆盖保底 + cluster-floor ViCoR+MMR"
        ),
    )
    parser.add_argument("--balance_col", type=str, default=None)
    parser.add_argument("--max_group_share", type=float, default=None)
    parser.add_argument("--min_group_samples", type=int, default=None)
    parser.add_argument("--tail_reserve_ratio", type=float, default=None)
    # MMR 参数
    parser.add_argument(
        "--mmr_lambda",
        type=float,
        default=None,
        help="MMR 权衡系数 λ ∈ [0,1]。λ=1.0 纯质量；λ=0.0 纯多样性；推荐 0.6~0.8",
    )
    parser.add_argument(
        "--mmr_candidate_ratio",
        type=float,
        default=None,
        help="MMR 候选池大小 = ratio * mmr_candidate_ratio（相对于最终选集）。"
             "默认 3.0，即在 top-60%% 中用 MMR 选 top-20%%",
    )
    parser.add_argument("--strict_balance_col", action="store_true")
    parser.add_argument("--no_strict_balance_col", action="store_true")
    parser.add_argument("--cluster_n_clusters", type=int, default=None)
    parser.add_argument("--cluster_min_quota", type=int, default=None)
    parser.add_argument("--cluster_score_alpha", type=float, default=None)
    parser.add_argument("--cluster_floor_ratio", type=float, default=None)
    parser.add_argument("--label_align_alpha", type=float, default=None)
    parser.add_argument("--directness_alpha", type=float, default=None)
    parser.add_argument("--cardinality_alpha", type=float, default=None)
    parser.add_argument("--target_cardinality", type=float, default=None)
    parser.add_argument("--label_floor_min_samples", type=int, default=None)
    return parser.parse_args()


def _resolve_settings(args, config: dict) -> dict:
    vicor_cfg = config.get("vicor", {})
    sel_cfg = vicor_cfg.get("selection", {})

    ratio             = args.ratio            if args.ratio            is not None else sel_cfg.get("ratio",             vicor_cfg.get("top_ratio", 0.2))
    score_col         = args.score_col        or sel_cfg.get("score_col",        "score")
    strategy          = args.strategy         or sel_cfg.get("strategy",         "vicor_mmr")
    balance_col       = args.balance_col      or sel_cfg.get("balance_col",      "")
    max_group_share   = args.max_group_share  if args.max_group_share  is not None else sel_cfg.get("max_group_share",  0.25)
    min_group_samples = args.min_group_samples if args.min_group_samples is not None else sel_cfg.get("min_group_samples", 3)
    tail_reserve_ratio = args.tail_reserve_ratio if args.tail_reserve_ratio is not None else sel_cfg.get("tail_reserve_ratio", 0.15)
    mmr_lambda        = args.mmr_lambda       if args.mmr_lambda       is not None else sel_cfg.get("mmr_lambda",       0.7)
    mmr_candidate_ratio = args.mmr_candidate_ratio if args.mmr_candidate_ratio is not None else sel_cfg.get("mmr_candidate_ratio", 3.0)
    cluster_n_clusters = args.cluster_n_clusters if args.cluster_n_clusters is not None else sel_cfg.get("cluster_n_clusters", 0)
    cluster_min_quota = args.cluster_min_quota if args.cluster_min_quota is not None else sel_cfg.get("cluster_min_quota", 1)
    cluster_score_alpha = args.cluster_score_alpha if args.cluster_score_alpha is not None else sel_cfg.get("cluster_score_alpha", 1.0)
    cluster_floor_ratio = args.cluster_floor_ratio if args.cluster_floor_ratio is not None else sel_cfg.get("cluster_floor_ratio", 0.25)
    label_align_alpha = args.label_align_alpha if args.label_align_alpha is not None else sel_cfg.get("label_align_alpha", 0.0)
    directness_alpha = args.directness_alpha if args.directness_alpha is not None else sel_cfg.get("directness_alpha", 0.0)
    cardinality_alpha = args.cardinality_alpha if args.cardinality_alpha is not None else sel_cfg.get("cardinality_alpha", 0.0)
    target_cardinality = args.target_cardinality if args.target_cardinality is not None else sel_cfg.get("target_cardinality", 0.0)
    label_floor_min_samples = args.label_floor_min_samples if args.label_floor_min_samples is not None else sel_cfg.get("label_floor_min_samples", 0)
    target_label_vocab = sel_cfg.get("target_label_vocab", [])
    target_label_aliases = sel_cfg.get("target_label_aliases", {})
    # The paper calls this softness parameter eta. Keep cardinality_slack as
    # a backward-compatible alias for older local score files/configs.
    cardinality_slack = sel_cfg.get("eta", sel_cfg.get("cardinality_slack", 1.0))
    beta_a = float(sel_cfg.get("beta_a", 0.70))
    beta_d = float(sel_cfg.get("beta_d", 0.85))

    strict = bool(sel_cfg.get("strict_balance_col", False))
    if args.strict_balance_col:
        strict = True
    if args.no_strict_balance_col:
        strict = False

    ratio              = float(ratio)
    max_group_share    = float(max_group_share)
    min_group_samples  = int(min_group_samples)
    tail_reserve_ratio = float(tail_reserve_ratio)
    mmr_lambda         = float(mmr_lambda)
    mmr_candidate_ratio = float(mmr_candidate_ratio)
    cluster_n_clusters = int(cluster_n_clusters)
    cluster_min_quota = int(cluster_min_quota)
    cluster_score_alpha = float(cluster_score_alpha)
    cluster_floor_ratio = float(cluster_floor_ratio)
    label_align_alpha = float(label_align_alpha)
    directness_alpha = float(directness_alpha)
    cardinality_alpha = float(cardinality_alpha)
    target_cardinality = float(target_cardinality)
    label_floor_min_samples = int(label_floor_min_samples)
    cardinality_slack = float(cardinality_slack)

    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0,1], got {ratio}")
    if not (0.0 < max_group_share <= 1.0):
        raise ValueError(f"max_group_share must be in (0,1], got {max_group_share}")
    if min_group_samples < 0:
        raise ValueError(f"min_group_samples must be >=0, got {min_group_samples}")
    if not (0.0 <= tail_reserve_ratio <= 0.5):
        raise ValueError(f"tail_reserve_ratio must be in [0,0.5], got {tail_reserve_ratio}")
    if not (0.0 <= mmr_lambda <= 1.0):
        raise ValueError(f"mmr_lambda must be in [0,1], got {mmr_lambda}")
    if mmr_candidate_ratio < 1.0:
        raise ValueError(f"mmr_candidate_ratio must be >=1.0, got {mmr_candidate_ratio}")
    if cluster_n_clusters < 0:
        raise ValueError(f"cluster_n_clusters must be >=0, got {cluster_n_clusters}")
    if cluster_min_quota < 0:
        raise ValueError(f"cluster_min_quota must be >=0, got {cluster_min_quota}")
    if cluster_score_alpha < 0.0:
        raise ValueError(f"cluster_score_alpha must be >=0, got {cluster_score_alpha}")
    if not (0.0 <= cluster_floor_ratio <= 1.0):
        raise ValueError(f"cluster_floor_ratio must be in [0,1], got {cluster_floor_ratio}")
    if label_align_alpha < 0.0 or directness_alpha < 0.0 or cardinality_alpha < 0.0:
        raise ValueError("label_align_alpha/directness_alpha/cardinality_alpha must be >= 0")
    if target_cardinality < 0.0:
        raise ValueError(f"target_cardinality must be >=0, got {target_cardinality}")
    if label_floor_min_samples < 0:
        raise ValueError(f"label_floor_min_samples must be >=0, got {label_floor_min_samples}")
    if cardinality_slack <= 0.0:
        raise ValueError(f"cardinality_slack must be >0, got {cardinality_slack}")
    if not isinstance(target_label_aliases, dict):
        raise ValueError("target_label_aliases must be a mapping of canonical label -> alias list/string")
    if not (0.0 <= beta_a <= 1.0 and 0.0 <= beta_d <= 1.0):
        raise ValueError("beta_a and beta_d must be in [0,1]")

    return dict(
        ratio=ratio,
        score_col=score_col,
        strategy=strategy,
        balance_col=balance_col,
        max_group_share=max_group_share,
        min_group_samples=min_group_samples,
        tail_reserve_ratio=tail_reserve_ratio,
        mmr_lambda=mmr_lambda,
        mmr_candidate_ratio=mmr_candidate_ratio,
        cluster_n_clusters=cluster_n_clusters,
        cluster_min_quota=cluster_min_quota,
        cluster_score_alpha=cluster_score_alpha,
        cluster_floor_ratio=cluster_floor_ratio,
        label_align_alpha=label_align_alpha,
        directness_alpha=directness_alpha,
        cardinality_alpha=cardinality_alpha,
        target_cardinality=target_cardinality,
        label_floor_min_samples=label_floor_min_samples,
        target_label_vocab=target_label_vocab,
        target_label_aliases=target_label_aliases,
        cardinality_slack=cardinality_slack,
        beta_a=beta_a,
        beta_d=beta_d,
        strict_balance_col=strict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_group_value(value) -> str:
    if pd.isna(value):
        return "__MISSING__"
    text = str(value).strip()
    return text if text else "__EMPTY__"


def _value_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if not col or col not in df.columns:
        return {}
    counts = df[col].map(_normalize_group_value).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    从 parquet 的统计字段构建轻量特征向量，用于 MMR 多样性度量。

    设计原则：
    - 只使用"任务语义"特征，不使用 score 本身（避免多样性与质量循环依赖）
    - 特征覆盖：问题复杂度（q_len）、推理链长度（r_len）、答案长度（a_len）
    - 如果有 answer_norm，用答案词袋的哈希桶做粗粒度语义区分
    - 归一化到 [0,1]，各维度权重相等

    注意：这是轻量替代 embedding 的方案（无需跑模型），
    对于医学 VLM 场景，问题类型 + CoT 长度已能抓住主要的语义多样性维度。
    """
    feature_cols = []

    for col in ["q_len", "r_len", "a_len", "seq_len"]:
        if col in df.columns:
            vals = df[col].fillna(0).astype(float).values
            v_max = vals.max()
            feature_cols.append(vals / (v_max + 1e-8))

    # 答案词哈希桶：将 answer_norm 映射到 32 个桶的 one-hot，
    # 捕捉粗粒度答案类型多样性（yes/no/数字/实体名称等）
    if "answer_norm" in df.columns:
        n_buckets = 32
        bucket_ids = df["answer_norm"].fillna("").apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % n_buckets
        ).values
        one_hot = np.zeros((len(df), n_buckets), dtype=float)
        one_hot[np.arange(len(df)), bucket_ids] = 1.0
        feature_cols.append(one_hot)

    # E_delta 和 rho 作为"视觉难度"维度（仅用于多样性，不影响质量轴）
    for col in ["E_delta", "rho", "nov"]:
        if col in df.columns:
            vals = df[col].fillna(0).astype(float).values
            v_max = vals.max()
            feature_cols.append((vals / (v_max + 1e-8)).reshape(-1, 1))

    if not feature_cols:
        # 极端兜底：返回全 1 矩阵，退化为纯 topk
        return np.ones((len(df), 1))

    parts = []
    for f in feature_cols:
        if f.ndim == 1:
            parts.append(f.reshape(-1, 1))
        else:
            parts.append(f)
    return np.hstack(parts).astype(np.float32)


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    """行向量 L2 归一化，避免构建 O(N^2) 相似度矩阵。"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    return X / norms


_DEFAULT_LABEL_ALIASES: dict[str, list[str]] = {
    "no finding": [
        "no findings",
        "normal",
        "normal study",
        "unremarkable",
        "negative",
        "no acute disease",
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary process",
        "no acute cardiopulmonary disease",
    ],
    "lung opacity": [
        "opacity",
        "opacities",
        "airspace opacity",
        "airspace opacities",
        "pulmonary opacity",
        "pulmonary opacities",
        "parenchymal opacity",
        "hazy opacity",
        "hazy opacities",
    ],
    "cardiomegaly": [
        "cardiac enlargement",
        "enlarged heart",
        "enlarged cardiac silhouette",
    ],
    "atelectasis": [
        "atelectatic change",
        "atelectatic changes",
        "subsegmental atelectasis",
        "bibasal atelectasis",
        "bibasilar atelectasis",
    ],
    "pleural effusion": [
        "pleural effusions",
        "effusion",
        "effusions",
        "bilateral pleural effusions",
    ],
    "edema": [
        "pulmonary edema",
        "interstitial edema",
    ],
    "pneumonia": [
        "infectious pneumonia",
    ],
    "support devices": [
        "support device",
        "endotracheal tube",
        "enteric tube",
        "ng tube",
        "central venous catheter",
        "central line",
        "picc",
        "pacemaker",
    ],
    "fracture": [
        "fractures",
        "rib fracture",
        "rib fractures",
        "osseous fracture",
    ],
    "lung lesion": [
        "mass",
        "masses",
        "nodule",
        "nodules",
        "pulmonary nodule",
        "pulmonary nodules",
    ],
    "consolidation": [
        "consolidations",
        "airspace consolidation",
        "focal consolidation",
    ],
    "enlarged cardiomediastinum": [
        "cardiomediastinal enlargement",
        "enlarged cardiomediastinal silhouette",
        "widened mediastinum",
        "mediastinal widening",
    ],
    "pleural other": [
        "pleural thickening",
        "pleural plaque",
        "pleural plaques",
        "pleural abnormality",
    ],
}


def _normalize_selector_text(text: str) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL)
    s = s.replace("\n", " ")
    s = s.replace(";", ",")
    s = s.replace("/", ",")
    s = s.replace("|", ",")
    s = s.replace("&", " and ")
    for pat in [
        r"\bthe disease of this patient is\b",
        r"\bthe diseases of this patient are\b",
        r"\bthe diagnosis is\b",
        r"\bthe diagnoses are\b",
        r"\bthe finding is\b",
        r"\bthe findings are\b",
        r"\bthe findings include\b",
        r"\bthe answer is\b",
        r"\banswer[:\s]+",
        r"\bfindings[:\s]+",
        r"\bimpression[:\s]+",
        r"\bthere is\b",
        r"\bthere are\b",
        r"\bcxr shows\b",
        r"\bxray shows\b",
        r"\bimage shows\b",
        r"\bthe chest radiograph shows\b",
    ]:
        s = re.sub(pat, " ", s)
    s = re.sub(r"[^a-z0-9,\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,.:;")


def _normalize_alias_map(target_label_vocab: list[str], target_label_aliases: dict) -> dict[str, list[str]]:
    alias_map: dict[str, set[str]] = {}
    for raw_label in target_label_vocab:
        label = _normalize_selector_text(str(raw_label))
        if not label:
            continue
        alias_map.setdefault(label, set()).add(label)
        for alias in _DEFAULT_LABEL_ALIASES.get(label, []):
            norm_alias = _normalize_selector_text(alias)
            if norm_alias:
                alias_map[label].add(norm_alias)

    for raw_label, raw_aliases in target_label_aliases.items():
        label = _normalize_selector_text(str(raw_label))
        if not label:
            continue
        alias_map.setdefault(label, set()).add(label)
        if isinstance(raw_aliases, str):
            alias_iter = [raw_aliases]
        elif isinstance(raw_aliases, list):
            alias_iter = raw_aliases
        else:
            continue
        for alias in alias_iter:
            norm_alias = _normalize_selector_text(str(alias))
            if norm_alias:
                alias_map[label].add(norm_alias)

    return {
        label: sorted(aliases, key=lambda x: (-len(x), x))
        for label, aliases in alias_map.items()
    }


def _contains_alias(text: str, alias: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None


def _canonicalize_part(part: str, alias_map: dict[str, list[str]]) -> str:
    if not part:
        return ""
    for canonical, aliases in alias_map.items():
        if part == canonical or part in aliases:
            return canonical
    return part


def _build_answer_labels_series(
    df: pd.DataFrame,
    target_label_vocab: list[str],
    target_label_aliases: dict,
) -> tuple[pd.Series, dict[str, list[str]]]:
    alias_map = _normalize_alias_map(target_label_vocab, target_label_aliases)

    if "answer_norm" in df.columns:
        raw_answers = df["answer_norm"]
        if "answer" in df.columns:
            raw_answers = raw_answers.fillna(df["answer"])
        raw_answers = raw_answers.fillna("")
    elif "answer" in df.columns:
        raw_answers = df["answer"].fillna("")
    else:
        empty = pd.Series([[] for _ in range(len(df))], index=df.index)
        return empty, alias_map

    labels_series = raw_answers.astype(str).apply(lambda x: _parse_answer_labels(x, alias_map))
    return labels_series, alias_map


def _parse_answer_labels(text: str, alias_map: Optional[dict[str, list[str]]] = None) -> list[str]:
    s = _normalize_selector_text(text)
    if not s:
        return []

    alias_map = alias_map or {}
    labels: list[str] = []
    # Clinical finding lists use more separators than commas and ``and``;
    # in particular CXR answers commonly say ``A with B``.  Split those
    # connectors first, then recover every canonical alias mentioned inside a
    # free-text segment instead of selecting only its longest match.
    parts = [p.strip() for p in re.split(r",|;|/|\band\b|\bwith\b|\bplus\b|、|，|；|\+", s) if p.strip()]
    for part in parts:
        # Preserve every parsed mention: m_i in Eq. (9)/(11) is a mention
        # count, so repeated findings must not be silently deduplicated.
        label = _canonicalize_part(part, alias_map)
        if label == part:
            # Handle free-text segments such as ``evidence of cardiomegaly``
            # and retain multiple non-overlapping findings in source order.
            matches = []
            for canonical, aliases in alias_map.items():
                for alias in aliases:
                    if not alias:
                        continue
                    for match in re.finditer(
                        rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", part
                    ):
                        matches.append((match.start(), -len(alias), canonical, match.end()))
            selected: list[tuple[int, int, str, int]] = []
            for candidate in sorted(matches):
                if any(candidate[0] < previous[3] and candidate[3] > previous[0] for previous in selected):
                    continue
                selected.append(candidate)
            if selected:
                labels.extend(item[2] for item in sorted(selected, key=lambda item: item[0]))
                continue
        if label:
            labels.append(label)

    return labels


def _compute_v6_selector_score(
    df: pd.DataFrame,
    base_score_col: str,
    target_label_vocab: list[str],
    target_label_aliases: dict,
    label_align_alpha: float,
    directness_alpha: float,
    cardinality_alpha: float,
    target_cardinality: float,
    cardinality_slack: float,
    beta_a: float = 0.70,
    beta_d: float = 0.85,
) -> pd.Series:
    base = df[base_score_col].astype(float).copy()

    labels_series, alias_map = _build_answer_labels_series(df, target_label_vocab, target_label_aliases)
    card = labels_series.apply(len).astype(float)

    vocab = set(alias_map.keys()) if alias_map else {_normalize_selector_text(x) for x in target_label_vocab if str(x).strip()}
    if vocab:
        hit_count = labels_series.apply(lambda xs: sum(1 for x in xs if x in vocab)).astype(float)
        align_precision = hit_count / np.maximum(card, 1.0)
        if target_cardinality > 0:
            align_support = np.minimum(hit_count, target_cardinality) / target_cardinality
        else:
            align_support = (hit_count > 0).astype(float)
        align = (beta_a * align_precision + (1.0 - beta_a) * align_support).astype(float)
    else:
        align = pd.Series(np.zeros(len(df), dtype=float), index=df.index)

    a_len = df['a_len'].astype(float) if 'a_len' in df.columns else pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    r_len = df['r_len'].astype(float) if 'r_len' in df.columns else pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    def percentile_rank(values: pd.Series) -> pd.Series:
        if len(values) <= 1:
            return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
        return (values.rank(method='average') - 1.0) / float(len(values) - 1)

    # Eq. (10) defines percentile ranks on [0, 1] within the full training pool.
    a_rank = percentile_rank(a_len)
    r_rank = percentile_rank(r_len)
    directness = 1.0 - (beta_d * a_rank + (1.0 - beta_d) * r_rank)

    if target_cardinality > 0:
        card_excess = np.maximum(card - target_cardinality, 0.0)
        card_pen = 1.0 / (1.0 + (card_excess / max(cardinality_slack, 1e-6)))
    else:
        card_pen = pd.Series(np.ones(len(df), dtype=float), index=df.index)

    bonus = 1.0 + label_align_alpha * align + directness_alpha * directness + cardinality_alpha * (card_pen - 1.0)
    return base * bonus


def _kmeans_assign(X: np.ndarray, n_clusters: int, n_iters: int = 20, seed: int = 42) -> np.ndarray:
    """轻量 k-means（余弦近似 via 归一化点积），避免引入额外依赖。"""
    n = len(X)
    if n == 0:
        return np.zeros((0,), dtype=int)
    if n_clusters <= 1 or n <= 1:
        return np.zeros((n,), dtype=int)
    n_clusters = min(n_clusters, n)

    Xn = _normalize_rows(X.astype(np.float32, copy=False))
    rng = np.random.default_rng(seed)

    # kmeans++-like init on cosine distance
    centers_idx = [int(rng.integers(0, n))]
    max_sim = Xn @ Xn[centers_idx[0]]
    for _ in range(1, n_clusters):
        dist = 1.0 - max_sim
        next_idx = int(np.argmax(dist))
        centers_idx.append(next_idx)
        max_sim = np.maximum(max_sim, Xn @ Xn[next_idx])

    centers = Xn[np.array(centers_idx)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iters):
        sims = Xn @ centers.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if not np.any(mask):
                centers[k] = Xn[int(rng.integers(0, n))]
            else:
                c = Xn[mask].mean(axis=0)
                norm = np.linalg.norm(c)
                centers[k] = c / (norm + 1e-8)
    return labels


def _allocate_cluster_budgets(
    labels: np.ndarray,
    scores_norm: np.ndarray,
    k: int,
    min_quota: int,
    score_alpha: float,
) -> dict[int, int]:
    clusters = sorted(set(int(x) for x in labels.tolist()))
    cluster_rows: dict[int, np.ndarray] = {c: np.where(labels == c)[0] for c in clusters}
    # utility-aware cluster weight: size * (mean_score ^ alpha)
    weights = {}
    for c, rows in cluster_rows.items():
        mean_score = float(scores_norm[rows].mean()) if len(rows) else 0.0
        weights[c] = len(rows) * ((mean_score + 1e-6) ** max(score_alpha, 0.0))

    budgets = {c: 0 for c in clusters}
    remaining = k

    # floor allocation to strongest clusters first
    if min_quota > 0 and remaining > 0:
        ordered = sorted(clusters, key=lambda c: (-weights[c], c))
        for c in ordered:
            if remaining <= 0:
                break
            take = min(min_quota, len(cluster_rows[c]), remaining)
            budgets[c] += take
            remaining -= take

    # proportional allocation over remaining budget
    if remaining > 0:
        total_weight = sum(weights.values())
        if total_weight <= 0:
            ordered = sorted(clusters, key=lambda c: (-len(cluster_rows[c]), c))
            idx = 0
            while remaining > 0 and ordered:
                c = ordered[idx % len(ordered)]
                if budgets[c] < len(cluster_rows[c]):
                    budgets[c] += 1
                    remaining -= 1
                idx += 1
        else:
            quotas = {c: remaining * weights[c] / total_weight for c in clusters}
            for c in clusters:
                add = min(len(cluster_rows[c]) - budgets[c], int(math.floor(quotas[c])))
                if add > 0:
                    budgets[c] += add
                    remaining -= add
            # distribute leftovers by largest fractional residual
            while remaining > 0:
                candidates = [
                    c for c in clusters if budgets[c] < len(cluster_rows[c])
                ]
                if not candidates:
                    break
                c = max(candidates, key=lambda x: (quotas[x] - math.floor(quotas[x]), weights[x]))
                budgets[c] += 1
                remaining -= 1

    return budgets


def _select_cluster_vicor_mmr(
    df: pd.DataFrame,
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    mmr_lambda: float,
    mmr_candidate_ratio: float,
    feat_matrix: np.ndarray,
    cluster_n_clusters: int,
    cluster_min_quota: int,
    cluster_score_alpha: float,
) -> pd.DataFrame:
    """Coverage-aware adaptive selection: allocate budget by cluster, then use ViCoR+MMR."""
    candidate_k = min(len(df), max(k, int(math.ceil(k * mmr_candidate_ratio))))
    candidate_indices = list(df_sorted.head(candidate_k).index)
    if len(candidate_indices) <= k:
        return df.loc[candidate_indices].copy().sort_values(by=score_col, ascending=False, kind='mergesort')

    cand_positions = {idx: i for i, idx in enumerate(df.index)}
    cand_rows = [cand_positions[idx] for idx in candidate_indices]
    X_cand = feat_matrix[cand_rows]

    n_clusters = cluster_n_clusters if cluster_n_clusters > 0 else min(k, max(4, int(round(math.sqrt(len(candidate_indices))))))
    labels = _kmeans_assign(X_cand, n_clusters=n_clusters)

    scores_raw = df.loc[candidate_indices, score_col].values.astype(float)
    s_min, s_max = scores_raw.min(), scores_raw.max()
    if s_max - s_min < 1e-12:
        scores_norm = np.ones_like(scores_raw)
    else:
        scores_norm = (scores_raw - s_min) / (s_max - s_min)

    budgets = _allocate_cluster_budgets(
        labels=labels,
        scores_norm=scores_norm,
        k=k,
        min_quota=cluster_min_quota,
        score_alpha=cluster_score_alpha,
    )

    selected: list[int] = []
    for cluster_id, quota in sorted(budgets.items(), key=lambda kv: (-kv[1], kv[0])):
        if quota <= 0:
            continue
        local_candidates = [candidate_indices[i] for i in np.where(labels == cluster_id)[0].tolist()]
        local_selected = _select_mmr(
            df=df,
            candidate_indices=local_candidates,
            k=min(quota, len(local_candidates)),
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            feat_matrix=feat_matrix,
        )
        selected.extend(local_selected)

    # fallback fill if cluster budgets underfill due to caps/rounding
    if len(selected) < k:
        selected_set = set(selected)
        remainder = [idx for idx in candidate_indices if idx not in selected_set]
        if remainder:
            fill = _select_mmr(
                df=df,
                candidate_indices=remainder,
                k=min(k - len(selected), len(remainder)),
                score_col=score_col,
                mmr_lambda=mmr_lambda,
                feat_matrix=feat_matrix,
            )
            selected.extend(fill)

    selected = selected[:k]
    selected_df = df.loc[selected].copy()
    return selected_df.sort_values(by=score_col, ascending=False, kind='mergesort')


def _select_cluster_floor_vicor_mmr(
    df: pd.DataFrame,
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    mmr_lambda: float,
    mmr_candidate_ratio: float,
    feat_matrix: np.ndarray,
    cluster_n_clusters: int,
    cluster_min_quota: int,
    cluster_score_alpha: float,
    cluster_floor_ratio: float,
) -> pd.DataFrame:
    """更温和的 coverage 借鉴：先按簇保底，再用全局 ViCoR+MMR 填满剩余预算。"""
    candidate_k = min(len(df), max(k, int(math.ceil(k * mmr_candidate_ratio))))
    candidate_indices = list(df_sorted.head(candidate_k).index)
    if len(candidate_indices) <= k:
        return df.loc[candidate_indices].copy().sort_values(by=score_col, ascending=False, kind='mergesort')

    cand_positions = {idx: i for i, idx in enumerate(df.index)}
    cand_rows = [cand_positions[idx] for idx in candidate_indices]
    X_cand = feat_matrix[cand_rows]

    n_clusters = cluster_n_clusters if cluster_n_clusters > 0 else min(k, max(4, int(round(math.sqrt(len(candidate_indices))))))
    labels = _kmeans_assign(X_cand, n_clusters=n_clusters)

    scores_raw = df.loc[candidate_indices, score_col].values.astype(float)
    s_min, s_max = scores_raw.min(), scores_raw.max()
    if s_max - s_min < 1e-12:
        scores_norm = np.ones_like(scores_raw)
    else:
        scores_norm = (scores_raw - s_min) / (s_max - s_min)

    floor_k = min(k, max(0, int(round(k * cluster_floor_ratio))))
    floor_budgets = _allocate_cluster_budgets(
        labels=labels,
        scores_norm=scores_norm,
        k=floor_k,
        min_quota=cluster_min_quota,
        score_alpha=cluster_score_alpha,
    )

    selected: list[int] = []
    for cluster_id, quota in sorted(floor_budgets.items(), key=lambda kv: (-kv[1], kv[0])):
        if quota <= 0:
            continue
        local_candidates = [candidate_indices[i] for i in np.where(labels == cluster_id)[0].tolist()]
        local_selected = _select_mmr(
            df=df,
            candidate_indices=local_candidates,
            k=min(quota, len(local_candidates)),
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            feat_matrix=feat_matrix,
        )
        selected.extend(local_selected)

    selected_set = set(selected)
    remainder = [idx for idx in candidate_indices if idx not in selected_set]
    if len(selected) < k and remainder:
        fill = _select_mmr(
            df=df,
            candidate_indices=remainder,
            k=min(k - len(selected), len(remainder)),
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            feat_matrix=feat_matrix,
        )
        selected.extend(fill)

    selected = selected[:k]
    selected_df = df.loc[selected].copy()
    return selected_df.sort_values(by=score_col, ascending=False, kind='mergesort')


def _slice_feat_matrix_by_index(
    df: pd.DataFrame,
    subset_index: list[int],
    feat_matrix: np.ndarray,
) -> np.ndarray:
    if not subset_index:
        return np.zeros((0, feat_matrix.shape[1]), dtype=feat_matrix.dtype)
    pos_map = {idx: i for i, idx in enumerate(df.index)}
    rows = [pos_map[idx] for idx in subset_index]
    return feat_matrix[rows]


def _select_label_floor_vicor_mmr(
    df: pd.DataFrame,
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    mmr_lambda: float,
    mmr_candidate_ratio: float,
    feat_matrix: np.ndarray,
    cluster_n_clusters: int,
    cluster_min_quota: int,
    cluster_score_alpha: float,
    cluster_floor_ratio: float,
    labels_series: pd.Series,
    target_label_vocab: list[str],
    label_floor_min_samples: int,
) -> pd.DataFrame:
    if label_floor_min_samples <= 0 or not target_label_vocab:
        return _select_cluster_floor_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            mmr_candidate_ratio=mmr_candidate_ratio,
            feat_matrix=feat_matrix,
            cluster_n_clusters=cluster_n_clusters,
            cluster_min_quota=cluster_min_quota,
            cluster_score_alpha=cluster_score_alpha,
            cluster_floor_ratio=cluster_floor_ratio,
        )

    normalized_labels: list[str] = []
    seen_labels: set[str] = set()
    for raw_label in target_label_vocab:
        label = _normalize_selector_text(str(raw_label))
        if label and label not in seen_labels:
            seen_labels.add(label)
            normalized_labels.append(label)

    label_to_indices: dict[str, list[int]] = {label: [] for label in normalized_labels}
    labels_lookup = labels_series.to_dict()
    for idx in df_sorted.index.tolist():
        for label in labels_lookup.get(idx, []):
            if label in label_to_indices:
                label_to_indices[label].append(idx)

    selected: list[int] = []
    selected_set: set[int] = set()

    present_labels = [label for label in normalized_labels if label_to_indices.get(label)]
    present_labels.sort(key=lambda label: (len(label_to_indices[label]), label))

    for label in present_labels:
        if len(selected) >= k:
            break
        remaining_candidates = [idx for idx in label_to_indices[label] if idx not in selected_set]
        quota = min(label_floor_min_samples, len(remaining_candidates), k - len(selected))
        if quota <= 0:
            continue
        candidate_k = min(
            len(remaining_candidates),
            max(quota, int(math.ceil(quota * mmr_candidate_ratio))),
        )
        local_selected = _select_mmr(
            df=df,
            candidate_indices=remaining_candidates[:candidate_k],
            k=quota,
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            feat_matrix=feat_matrix,
        )
        for idx in local_selected:
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)

    if len(selected) >= k:
        selected_df = df.loc[selected[:k]].copy()
        return selected_df.sort_values(by=score_col, ascending=False, kind='mergesort')

    remainder_indices = [idx for idx in df.index.tolist() if idx not in selected_set]
    if not remainder_indices:
        selected_df = df.loc[selected].copy()
        return selected_df.sort_values(by=score_col, ascending=False, kind='mergesort')

    remainder_df = df.loc[remainder_indices].copy()
    remainder_sorted = remainder_df.sort_values(by=score_col, ascending=False, kind='mergesort')
    remainder_feat_matrix = _slice_feat_matrix_by_index(df, remainder_indices, feat_matrix)
    fill_df = _select_cluster_floor_vicor_mmr(
        df=remainder_df,
        df_sorted=remainder_sorted,
        k=min(k - len(selected), len(remainder_df)),
        score_col=score_col,
        mmr_lambda=mmr_lambda,
        mmr_candidate_ratio=mmr_candidate_ratio,
        feat_matrix=remainder_feat_matrix,
        cluster_n_clusters=cluster_n_clusters,
        cluster_min_quota=cluster_min_quota,
        cluster_score_alpha=cluster_score_alpha,
        cluster_floor_ratio=cluster_floor_ratio,
    )
    all_selected = selected + fill_df.index.tolist()
    selected_df = df.loc[all_selected].copy()
    return selected_df.sort_values(by=score_col, ascending=False, kind='mergesort')


def _count_target_labels_from_series(
    labels_series: pd.Series,
    target_label_vocab: list[str],
) -> dict[str, int]:
    vocab = {
        _normalize_selector_text(str(label))
        for label in target_label_vocab
        if _normalize_selector_text(str(label))
    }
    counts: defaultdict[str, int] = defaultdict(int)
    for labels in labels_series.tolist():
        for label in labels:
            if label in vocab:
                counts[label] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _enforce_group_caps(
    df: pd.DataFrame,
    selected_df: pd.DataFrame,
    k: int,
    score_col: str,
    balance_col: str,
    max_group_share: float,
) -> pd.DataFrame:
    """Repair selections so the configured group cap is actually enforced.

    The paper's constrained stage promises group-wise upper caps.  Several
    quality/MMR selectors do not need a group column inside their selection step, so this final
    deterministic repair applies the same cap to every strategy.  If the pool
    contains only one group, the cap is vacuous and all rows may be selected.
    """
    if not balance_col or balance_col not in df.columns or balance_col not in selected_df.columns:
        return selected_df
    if df[balance_col].nunique(dropna=False) <= 1:
        return selected_df
    cap = max(1, int(math.ceil(k * max_group_share)))
    selected = selected_df.copy()
    counts = selected[balance_col].map(_normalize_group_value).value_counts().to_dict()
    overflow = {group: int(count - cap) for group, count in counts.items() if count > cap}
    if not overflow:
        return selected

    selected = selected.sort_values(by=score_col, ascending=False, kind="mergesort")
    drop_indices: list[object] = []
    for group, excess in overflow.items():
        candidates = selected[selected[balance_col].map(_normalize_group_value) == group].sort_values(
            by=score_col, ascending=True, kind="mergesort"
        )
        drop_indices.extend(candidates.head(excess).index.tolist())
    selected = selected.drop(index=drop_indices)

    used = set(selected.index)
    remaining = df.drop(index=list(used), errors="ignore").copy()
    remaining["__group_norm__"] = remaining[balance_col].map(_normalize_group_value)
    counts = selected[balance_col].map(_normalize_group_value).value_counts().to_dict()
    additions: list[object] = []
    for idx, row in remaining.sort_values(by=score_col, ascending=False, kind="mergesort").iterrows():
        group = row["__group_norm__"]
        if counts.get(group, 0) >= cap:
            continue
        additions.append(idx)
        counts[group] = counts.get(group, 0) + 1
        if len(selected) + len(additions) >= k:
            break
    repaired = pd.concat([selected, df.loc[additions]], axis=0)
    return repaired.sort_values(by=score_col, ascending=False, kind="mergesort")


def _select_paper_group_vicor_mmr(
    df: pd.DataFrame,
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    balance_col: str,
    min_group_samples: int,
    max_group_share: float,
    mmr_lambda: float,
    mmr_candidate_ratio: float,
    feat_matrix: np.ndarray,
) -> pd.DataFrame:
    """Implement the paper's group floors/caps around pure ViCoR scores."""
    if not balance_col or balance_col not in df.columns:
        raise ValueError("paper_group_vicor_mmr requires a materialized balance_col")
    candidate_k = min(len(df), max(k, int(math.ceil(k * mmr_candidate_ratio))))
    candidate_indices = list(df_sorted.head(candidate_k).index)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in candidate_indices:
        groups[_normalize_group_value(df.at[idx, balance_col])].append(idx)
    cap = max(1, int(math.ceil(k * max_group_share)))
    if min_group_samples * len(groups) > k:
        raise ValueError("paper group floors are infeasible for the requested budget")
    if cap * len(groups) < k:
        raise ValueError("paper group caps are infeasible for the requested budget")

    selected: list[int] = []
    counts: dict[str, int] = defaultdict(int)
    # First satisfy every group floor with a within-group MMR pass.
    for group in sorted(groups):
        local = groups[group]
        take = min(min_group_samples, len(local), k - len(selected))
        if take <= 0:
            continue
        chosen = _select_mmr(df, local, take, score_col, mmr_lambda, feat_matrix)
        selected.extend(chosen)
        counts[group] += len(chosen)

    # Fill the remaining budget by global ViCoR+MMR while respecting caps.
    while len(selected) < k:
        remaining = [idx for idx in candidate_indices if idx not in set(selected) and counts[_normalize_group_value(df.at[idx, balance_col])] < cap]
        if not remaining:
            break
        take = min(k - len(selected), len(remaining))
        chosen = _select_mmr(df, remaining, take, score_col, mmr_lambda, feat_matrix)
        if not chosen:
            break
        selected.extend(chosen)
        for idx in chosen:
            counts[_normalize_group_value(df.at[idx, balance_col])] += 1
    if len(selected) != k:
        raise RuntimeError(f"paper_group_vicor_mmr selected {len(selected)}/{k}; adjust group floors/caps")
    return df.loc[selected].sort_values(by=score_col, ascending=False, kind="mergesort")


def stage1_rank_signature(
    base_score_col: str,
    target_label_vocab: list,
    target_label_aliases: dict,
    label_align_alpha: float,
    directness_alpha: float,
    cardinality_alpha: float,
    target_cardinality: float,
    cardinality_slack: float,
    beta_a: float,
    beta_d: float,
) -> str:
    """Stable identifier of the settings a Stage-I ``stage1_rank_score`` column was computed with."""
    import hashlib
    payload = {
        "base_score_col": str(base_score_col),
        "vocab": list(target_label_vocab or []),
        "aliases": target_label_aliases or {},
        "label_align_alpha": float(label_align_alpha),
        "directness_alpha": float(directness_alpha),
        "cardinality_alpha": float(cardinality_alpha),
        "target_cardinality": float(target_cardinality),
        "cardinality_slack": float(cardinality_slack),
        "beta_a": float(beta_a),
        "beta_d": float(beta_d),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _select_paper_vicor_mmr(
    df: pd.DataFrame,
    k: int,
    base_score_col: str,
    mmr_lambda: float,
    mmr_candidate_ratio: float,
    feat_matrix: np.ndarray,
    target_label_vocab: list[str],
    target_label_aliases: dict,
    label_align_alpha: float,
    directness_alpha: float,
    cardinality_alpha: float,
    target_cardinality: float,
    cardinality_slack: float,
    beta_a: float,
    beta_d: float,
) -> pd.DataFrame:
    """Final-paper Eq. (12)-(13): calibrated S_rank followed by γK MMR."""
    ranked = df.copy()
    # Stage-I already computes S_rank on the full source pool before applying
    # the gamma cap. Preserve that full-pool calibration for the final MMR;
    # recomputing percentile ranks on the truncated candidate pool would alter
    # Eq. (10)-(13) and change the paper's candidate ordering.
    expected_signature = stage1_rank_signature(
        base_score_col=base_score_col,
        target_label_vocab=target_label_vocab,
        target_label_aliases=target_label_aliases,
        label_align_alpha=label_align_alpha,
        directness_alpha=directness_alpha,
        cardinality_alpha=cardinality_alpha,
        target_cardinality=target_cardinality,
        cardinality_slack=cardinality_slack,
        beta_a=beta_a,
        beta_d=beta_d,
    )
    cached_signatures = (
        set(ranked["stage1_rank_signature"].dropna().astype(str).unique().tolist())
        if "stage1_rank_signature" in ranked.columns else set()
    )
    if "stage1_rank_score" in ranked.columns and cached_signatures == {expected_signature}:
        rank_col = "stage1_rank_score"
    else:
        if "stage1_rank_score" in ranked.columns:
            print(
                "[select] cached stage1_rank_score was computed with different calibration settings "
                f"(found {sorted(cached_signatures) or ['<no signature>']}, expected {expected_signature}); recomputing S_rank"
            )
        rank_col = "__paper_rank_score__"
        ranked[rank_col] = _compute_v6_selector_score(
            df=ranked,
            base_score_col=base_score_col,
            target_label_vocab=target_label_vocab,
            target_label_aliases=target_label_aliases,
            label_align_alpha=label_align_alpha,
            directness_alpha=directness_alpha,
            cardinality_alpha=cardinality_alpha,
            target_cardinality=target_cardinality,
            cardinality_slack=cardinality_slack,
            beta_a=beta_a,
            beta_d=beta_d,
        )
    ranked_sorted = ranked.sort_values(by=rank_col, ascending=False, kind="mergesort")
    candidate_k = min(len(ranked), max(k, int(math.ceil(k * mmr_candidate_ratio))))
    candidate_indices = list(ranked_sorted.head(candidate_k).index)
    candidates = ranked.loc[candidate_indices].copy()
    chosen = _select_mmr(
        df=candidates,
        candidate_indices=candidate_indices,
        k=k,
        score_col=rank_col,
        mmr_lambda=mmr_lambda,
        feat_matrix=_build_feature_matrix(candidates),
    )
    selected = ranked.loc[chosen].copy()
    selected["base_score_for_report"] = selected[base_score_col]
    selected["selector_score_vicor"] = selected[rank_col]
    return selected.sort_values(by=rank_col, ascending=False, kind="mergesort")


def _compute_answer_openness_stats(df: pd.DataFrame) -> dict[str, float | int | str | None]:
    ans_col = None
    for cand in ["answer_norm", "answer"]:
        if cand in df.columns:
            ans_col = cand
            break

    tail_rate_all = float(df["is_tail"].mean()) if "is_tail" in df.columns and len(df) > 0 else 0.0
    if ans_col is None or len(df) == 0:
        return {
            "answer_col": ans_col,
            "unique_answers_all": None,
            "top1_ratio_all": None,
            "tail_rate_all": tail_rate_all,
            "unique_signal": 0.5,
            "concentration_signal": 0.5,
            "tail_signal": min(1.0, max(0.0, tail_rate_all)),
            "open_scale": 0.5,
        }

    ans_counts = df[ans_col].value_counts(dropna=False)
    total = max(1, len(df))
    unique_answers_all = int(ans_counts.shape[0])
    top1_ratio_all = float(ans_counts.iloc[0] / total) if len(ans_counts) else 1.0

    low_anchor = math.log1p(min(total, 16))
    high_anchor = math.log1p(total)
    if high_anchor - low_anchor < 1e-8:
        unique_signal = 0.0
    else:
        unique_signal = (math.log1p(unique_answers_all) - low_anchor) / (high_anchor - low_anchor)
    unique_signal = float(min(1.0, max(0.0, unique_signal)))

    concentration_signal = 1.0 - ((top1_ratio_all - 0.03) / 0.30)
    concentration_signal = float(min(1.0, max(0.0, concentration_signal)))

    tail_signal = float(min(1.0, max(0.0, (tail_rate_all - 0.08) / 0.35)))
    open_scale = float(min(1.0, max(0.0, 0.45 * unique_signal + 0.35 * concentration_signal + 0.20 * tail_signal)))

    return {
        "answer_col": ans_col,
        "unique_answers_all": unique_answers_all,
        "top1_ratio_all": top1_ratio_all,
        "tail_rate_all": tail_rate_all,
        "unique_signal": unique_signal,
        "concentration_signal": concentration_signal,
        "tail_signal": tail_signal,
        "open_scale": open_scale,
    }


def _derive_adaptive_selector_meta(
    df: pd.DataFrame,
    base_tail_reserve_ratio: float,
    base_cluster_floor_ratio: float,
) -> dict[str, float | int | str | None]:
    stats = _compute_answer_openness_stats(df)
    open_scale = _meta_float(stats, "open_scale", 0.5)
    target_tail_ratio = 0.05 + 0.08 * open_scale
    target_cluster_floor = 0.34 - 0.22 * open_scale

    tail_ratio_used = float(min(0.18, max(0.03, 0.5 * float(base_tail_reserve_ratio) + 0.5 * target_tail_ratio)))
    cluster_floor_ratio_used = float(min(0.35, max(0.08, 0.5 * float(base_cluster_floor_ratio) + 0.5 * target_cluster_floor)))

    stats.update({
        "tail_ratio_used": tail_ratio_used,
        "cluster_floor_ratio_used": cluster_floor_ratio_used,
    })
    return stats


def _meta_float(meta: dict[str, float | int | str | None], key: str, default: float = 0.0) -> float:
    value = meta.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _select_adaptive_tail_cluster_vicor_mmr(
    df: pd.DataFrame,
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    base_tail_reserve_ratio: float,
    mmr_lambda: float,
    mmr_candidate_ratio: float,
    feat_matrix: np.ndarray,
    cluster_n_clusters: int,
    cluster_min_quota: int,
    cluster_score_alpha: float,
    base_cluster_floor_ratio: float,
    adaptive_meta: dict[str, float | int | str | None],
) -> pd.DataFrame:
    if "is_tail" not in df_sorted.columns:
        return _select_cluster_floor_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            mmr_candidate_ratio=mmr_candidate_ratio,
            feat_matrix=feat_matrix,
            cluster_n_clusters=cluster_n_clusters,
            cluster_min_quota=cluster_min_quota,
            cluster_score_alpha=cluster_score_alpha,
            cluster_floor_ratio=_meta_float(adaptive_meta, "cluster_floor_ratio_used", base_cluster_floor_ratio),
        )

    tail_ratio_used = _meta_float(adaptive_meta, "tail_ratio_used", base_tail_reserve_ratio)
    cluster_floor_ratio_used = _meta_float(adaptive_meta, "cluster_floor_ratio_used", base_cluster_floor_ratio)

    tail_quota = int(math.ceil(k * tail_ratio_used))
    tail_sorted = df_sorted[df_sorted["is_tail"] == True]
    non_tail_sorted = df_sorted[df_sorted["is_tail"] == False]

    if len(tail_sorted) == 0 or tail_quota <= 0:
        tail_selected_idx: list[int] = []
    elif len(tail_sorted) <= tail_quota:
        tail_selected_idx = list(tail_sorted.index)
    else:
        tail_cand_k = min(len(tail_sorted), max(tail_quota, int(math.ceil(tail_quota * mmr_candidate_ratio))))
        tail_selected_idx = _select_mmr(
            df=df,
            candidate_indices=list(tail_sorted.head(tail_cand_k).index),
            k=tail_quota,
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            feat_matrix=feat_matrix,
        )

    selected_set = set(tail_selected_idx)
    fill_quota = k - len(tail_selected_idx)
    remainder_idx = [idx for idx in df.index.tolist() if idx not in selected_set]
    if fill_quota <= 0 or not remainder_idx:
        selected_df = df.loc[tail_selected_idx].copy()
        return selected_df.sort_values(by=score_col, ascending=False, kind="mergesort")

    remainder_df = df.loc[remainder_idx].copy()
    remainder_sorted = remainder_df.sort_values(by=score_col, ascending=False, kind="mergesort")
    remainder_feat_matrix = _slice_feat_matrix_by_index(df, remainder_idx, feat_matrix)
    fill_df = _select_cluster_floor_vicor_mmr(
        df=remainder_df,
        df_sorted=remainder_sorted,
        k=min(fill_quota, len(remainder_df)),
        score_col=score_col,
        mmr_lambda=mmr_lambda,
        mmr_candidate_ratio=mmr_candidate_ratio,
        feat_matrix=remainder_feat_matrix,
        cluster_n_clusters=cluster_n_clusters,
        cluster_min_quota=cluster_min_quota,
        cluster_score_alpha=cluster_score_alpha,
        cluster_floor_ratio=cluster_floor_ratio_used,
    )
    selected_df = df.loc[tail_selected_idx + fill_df.index.tolist()].copy()
    return selected_df.sort_values(by=score_col, ascending=False, kind="mergesort")


# ─────────────────────────────────────────────────────────────────────────────
# 核心选择策略
# ─────────────────────────────────────────────────────────────────────────────

def _select_mmr(
    df: pd.DataFrame,
    candidate_indices: list[int],
    k: int,
    score_col: str,
    mmr_lambda: float,
    feat_matrix: np.ndarray,
) -> list[int]:
    """
    Maximal Marginal Relevance (MMR) 选择算法。

    MMR 公式：
        argmax_{d ∈ C\\S} [ λ · Score(d)
                           - (1-λ) · max_{s ∈ S} Sim(d, s) ]

    其中：
        C  = 候选集（已按 ViCoR Score 预筛到前 candidate_ratio 比例）
        S  = 已选集（初始为空，每轮迭代扩充）
        Score(d) = ViCoR Score，归一化到 [0,1]
        Sim(d,s) = 余弦相似度（基于任务语义特征向量）
        λ  = 质量-多样性权衡系数（λ→1 纯质量，λ→0 纯多样性）

    设计要点：
    - 候选集限制为 top-(ratio * candidate_ratio) 个，保证基础质量下限
    - 特征向量不含 score，避免多样性与质量的循环依赖
    - 每轮 O(|C|) 更新 max_sim，总复杂度 O(k·|C|)，对10万量级可行
    """
    if k <= 0 or not candidate_indices:
        return []

    cand = list(candidate_indices)
    if len(cand) <= k:
        return cand

    # 归一化 score 到 [0,1]
    scores_raw = df.loc[cand, score_col].values.astype(float)
    s_min, s_max = scores_raw.min(), scores_raw.max()
    if s_max - s_min < 1e-12:
        scores_norm = np.ones_like(scores_raw)
    else:
        scores_norm = (scores_raw - s_min) / (s_max - s_min)
    score_map = {idx: scores_norm[i] for i, idx in enumerate(cand)}

    # 提取候选集的特征向量
    cand_positions = {idx: i for i, idx in enumerate(df.index)}
    cand_feat_rows = [cand_positions[idx] for idx in cand]
    X_cand = feat_matrix[cand_feat_rows]  # [|C|, D]

    # 仅做行归一化：后续每轮与新选样本做一次向量化点积更新 max_sim。
    # 避免显式构建 [|C|,|C|] 的相似度矩阵（内存 O(C^2)）。
    X_cand = _normalize_rows(X_cand).astype(np.float32, copy=False)

    selected_global: list[int] = []  # 已选样本的全局 df index
    C = len(cand)
    # 布尔掩码：True = 仍在候选集中（替代 list.remove，避免 O(C) scan）
    remaining_mask = np.ones(C, dtype=bool)

    # 归一化 score 向量（与 cand 顺序对齐）
    scores_vec = np.array([score_map[idx] for idx in cand], dtype=float)

    # 初始：选 ViCoR Score 最高的样本作为种子
    seed_local = int(np.argmax(scores_raw))
    selected_global.append(cand[seed_local])
    remaining_mask[seed_local] = False

    # max_sim[i] = max_{s ∈ S} Sim(cand[i], s)
    max_sim = X_cand @ X_cand[seed_local]  # [C]

    while len(selected_global) < k and remaining_mask.any():
        # 向量化 MMR：一次性计算所有候选的 MMR 值
        mmr_vals = mmr_lambda * scores_vec - (1.0 - mmr_lambda) * max_sim
        mmr_vals[~remaining_mask] = -1e9  # 已选 / 已排除的置为最低

        best_local = int(np.argmax(mmr_vals))
        selected_global.append(cand[best_local])
        remaining_mask[best_local] = False

        # 原地更新 max_sim（与新选样本做一次向量化相似度计算）
        sim_to_best = X_cand @ X_cand[best_local]
        np.maximum(max_sim, sim_to_best, out=max_sim)

    return selected_global


def _select_vicor_mmr(
    df: pd.DataFrame,
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    mmr_lambda: float,
    mmr_candidate_ratio: float,
    feat_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    ViCoR + MMR 联合选择：质量筛选 → 多样性重排。

    流程：
    1. 从全量数据中取前 k * candidate_ratio 个样本作为候选池
       （保证候选池内每个样本的 ViCoR Score 都在前 candidate_ratio 比例内）
    2. 在候选池内运行 MMR，迭代选出 k 个既高质量又多样的样本

    参数：
        mmr_lambda: λ，越大越偏质量，越小越偏多样性
        mmr_candidate_ratio: 候选池 = k * ratio，通常取 2.5~4.0
    """
    candidate_k = min(len(df), max(k, int(math.ceil(k * mmr_candidate_ratio))))
    candidate_indices = list(df_sorted.head(candidate_k).index)

    selected_global = _select_mmr(
        df=df,
        candidate_indices=candidate_indices,
        k=k,
        score_col=score_col,
        mmr_lambda=mmr_lambda,
        feat_matrix=feat_matrix,
    )

    selected_df = df.loc[selected_global].copy()
    # 重排：先按 ViCoR Score 降序输出，保持与 topk 一致的可读性
    return selected_df.sort_values(by=score_col, ascending=False, kind="mergesort")


def _select_tail_vicor_mmr(
    df: pd.DataFrame,
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    tail_reserve_ratio: float,
    mmr_lambda: float,
    mmr_candidate_ratio: float,
    feat_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    长尾保护 + ViCoR + MMR 三层联合选择。

    流程：
    1. 长尾池（is_tail==True）和非长尾池分别独立运行 ViCoR+MMR
    2. 长尾池分配 tail_reserve_ratio * k 个配额
    3. 非长尾池分配剩余配额
    4. 合并后按 score 降序输出

    核心优势（相比旧版 tail_aware_topk）：
    - 旧版：长尾/非长尾池内各自纯 topk，仍然可能语义塌缩
    - 新版：长尾/非长尾池内各自独立运行 MMR，两池内部都保证多样性
    - 对长尾严重的 GEMeX（开放域答案几乎全是 tail）效果更好
    """
    if "is_tail" not in df_sorted.columns:
        raise KeyError(
            "[P0-2] strategy=tail_vicor_mmr requires 'is_tail' column, but it is missing. "
            "Run 01_score_vicor.py first, which adds 'is_tail' based on answer_freq."
        )

    tail_quota = int(math.ceil(k * tail_reserve_ratio))
    main_quota = k - tail_quota

    tail_sorted     = df_sorted[df_sorted["is_tail"] == True]
    non_tail_sorted = df_sorted[df_sorted["is_tail"] == False]

    # ── 长尾池：ViCoR + MMR ──
    actual_tail_pool = len(tail_sorted)
    if actual_tail_pool == 0:
        tail_selected_idx = []
        main_quota = k
    elif actual_tail_pool <= tail_quota:
        # 长尾样本不足配额，全选，剩余补给非长尾
        tail_selected_idx = list(tail_sorted.index)
        bonus = tail_quota - len(tail_selected_idx)
        main_quota += bonus
        print(f"[Info] tail pool ({actual_tail_pool}) < quota ({tail_quota}), "
              f"bonus {bonus} → non-tail.")
    else:
        # 长尾池内独立运行 MMR
        tail_cand_k = min(actual_tail_pool, max(tail_quota, int(math.ceil(tail_quota * mmr_candidate_ratio))))
        tail_cand_idx = list(tail_sorted.head(tail_cand_k).index)
        tail_selected_idx = _select_mmr(
            df=df,
            candidate_indices=tail_cand_idx,
            k=tail_quota,
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            feat_matrix=feat_matrix,
        )

    # ── 非长尾池：ViCoR + MMR ──
    tail_idx_set = set(tail_selected_idx)
    non_tail_sorted_clean = non_tail_sorted[~non_tail_sorted.index.isin(tail_idx_set)]
    actual_non_tail = len(non_tail_sorted_clean)

    if actual_non_tail == 0:
        non_tail_selected_idx = []
    elif actual_non_tail <= main_quota:
        non_tail_selected_idx = list(non_tail_sorted_clean.index)
        print(f"[Info] non-tail pool ({actual_non_tail}) < quota ({main_quota}), using all.")
    else:
        non_tail_cand_k = min(actual_non_tail, max(main_quota, int(math.ceil(main_quota * mmr_candidate_ratio))))
        non_tail_cand_idx = list(non_tail_sorted_clean.head(non_tail_cand_k).index)
        non_tail_selected_idx = _select_mmr(
            df=df,
            candidate_indices=non_tail_cand_idx,
            k=main_quota,
            score_col=score_col,
            mmr_lambda=mmr_lambda,
            feat_matrix=feat_matrix,
        )

    all_selected = tail_selected_idx + non_tail_selected_idx
    selected_df = df.loc[all_selected].copy()
    return selected_df.sort_values(by=score_col, ascending=False, kind="mergesort")


def _select_topk(df_sorted: pd.DataFrame, k: int) -> pd.DataFrame:
    return df_sorted.head(k)


def _select_group_capped_topk(
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    balance_col: str,
    max_group_share: float,
    min_group_samples: int,
) -> pd.DataFrame:
    """
    分组约束 topk：每组最多占 max_group_share 比例。
    注意：此策略内部无 MMR，如需多样性请用 vicor_mmr 或 tail_vicor_mmr。
    """
    cap = max(1, int(math.ceil(k * max_group_share)))

    group_of_idx: dict[int, str] = {}
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, gv in df_sorted[balance_col].items():
        g = _normalize_group_value(gv)
        group_of_idx[idx] = g
        grouped_indices[g].append(idx)

    group_order: list[str] = []
    seen: set[str] = set()
    for idx in df_sorted.index:
        g = group_of_idx[idx]
        if g not in seen:
            seen.add(g)
            group_order.append(g)

    selected: list[int] = []
    selected_set: set[int] = set()
    group_counts: dict[str, int] = defaultdict(int)

    # Phase 1: 每组保底 min_group_samples
    if min_group_samples > 0:
        for g in group_order:
            if len(selected) >= k:
                break
            taken = 0
            for idx in grouped_indices[g]:
                if len(selected) >= k:
                    break
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                group_counts[g] += 1
                taken += 1
                if taken >= min_group_samples:
                    break

    # Phase 2: 贪心填充（受 cap 约束）
    for idx in df_sorted.index:
        if len(selected) >= k:
            break
        if idx in selected_set:
            continue
        g = group_of_idx[idx]
        if group_counts[g] >= cap:
            continue
        selected.append(idx)
        selected_set.add(idx)
        group_counts[g] += 1

    # Phase 3: 无法凑满时报错（P0-3 护栏）
    if len(selected) < k:
        raise RuntimeError(
            f"[P0-3] group_capped_topk: Phase1+Phase2 only yielded {len(selected)}/{k} samples. "
            f"The group constraints are too tight. "
            f"Increase ratio, relax max_group_share, or lower min_group_samples."
        )

    return df_sorted.loc[selected].sort_values(by=score_col, ascending=False, kind="mergesort")


def _select_round_robin(
    df_sorted: pd.DataFrame,
    k: int,
    score_col: str,
    balance_col: str,
) -> pd.DataFrame:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, value in df_sorted[balance_col].items():
        group_to_indices[_normalize_group_value(value)].append(idx)

    group_order: list[str] = []
    seen: set[str] = set()
    for idx in df_sorted.index:
        g = _normalize_group_value(df_sorted.at[idx, balance_col])
        if g not in seen:
            seen.add(g)
            group_order.append(g)

    pointers: dict[str, int] = defaultdict(int)
    selected: list[int] = []
    selected_set: set[int] = set()

    while len(selected) < k:
        progressed = False
        for g in group_order:
            idx_list = group_to_indices[g]
            p = pointers[g]
            while p < len(idx_list) and idx_list[p] in selected_set:
                p += 1
            pointers[g] = p
            if p >= len(idx_list):
                continue
            idx = idx_list[p]
            pointers[g] += 1
            selected.append(idx)
            selected_set.add(idx)
            progressed = True
            if len(selected) >= k:
                break
        if not progressed:
            break

    for idx in df_sorted.index:
        if len(selected) >= k:
            break
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)

    return df_sorted.loc[selected].sort_values(by=score_col, ascending=False, kind="mergesort")


def _validate_group_constraints(df, k, balance_col, max_group_share, min_group_samples):
    if not balance_col or balance_col not in df.columns:
        raise ValueError(f"Invalid balance_col={balance_col!r}")
    n_groups = int(df[balance_col].map(_normalize_group_value).nunique(dropna=False))
    if n_groups <= 0:
        raise ValueError("No groups found.")
    cap = max(1, int(math.ceil(k * max_group_share)))
    if min_group_samples * n_groups > k:
        raise ValueError(f"Infeasible: min_group_samples({min_group_samples}) * n_groups({n_groups}) > k({k})")
    if cap < min_group_samples:
        raise ValueError(f"Infeasible: cap({cap}) < min_group_samples({min_group_samples})")
    # P0-4: 总容量必须能装下 k 个样本
    if cap * n_groups < k:
        raise ValueError(
            f"[P0-4] Infeasible: cap({cap}) * n_groups({n_groups}) = {cap * n_groups} < k({k}). "
            f"Increase max_group_share or ratio."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 诊断输出
# ─────────────────────────────────────────────────────────────────────────────

def _print_diagnostics(df_all: pd.DataFrame, selected_df: pd.DataFrame, score_col: str) -> None:
    print(f"\n{'='*60}")
    print(f"Selection Diagnostics  ({len(selected_df)} / {len(df_all)} samples)")
    print(f"{'='*60}")

    if "is_tail" in df_all.columns and "is_tail" in selected_df.columns:
        total_tail    = int(df_all["is_tail"].sum())
        selected_tail = int(selected_df["is_tail"].sum())
        cover         = selected_tail / total_tail if total_tail > 0 else 0.0
        print(f"  Long-tail coverage : {selected_tail}/{total_tail} = {cover:.2%}")
        print(f"  Tail rate  (all)   : {total_tail/len(df_all):.2%}")
        print(f"  Tail rate  (sel)   : {selected_tail/len(selected_df):.2%}")

    if score_col in selected_df.columns:
        s = selected_df[score_col]
        print(f"  Score (selected)   : mean={s.mean():.4e}  std={s.std():.4e}  "
              f"min={s.min():.4e}  max={s.max():.4e}")
        zero_n = int((s == 0).sum())
        print(f"  Zero-score samples : {zero_n} ({zero_n/len(selected_df):.2%})")

    for col, label in [("answer_freq_bin", "answer_freq_bin"), ("r_len_bin", "r_len_bin"), ("rho_bin", "rho_bin")]:
        if col in selected_df.columns:
            dist = selected_df[col].value_counts().sort_index()
            print(f"  {label:18s}: {dict(dist)}")

    if "E_delta" in selected_df.columns:
        e = selected_df["E_delta"]
        print(f"  E_delta (selected) : mean={e.mean():.4e}  median={e.median():.4e}")

    # P0-5b: 答案多样性验收指标
    ans_col = None
    for cand in ["answer_norm", "answer"]:
        if cand in selected_df.columns:
            ans_col = cand
            break
    if ans_col is not None:
        ans_counts = selected_df[ans_col].value_counts()
        unique_ans = int(ans_counts.shape[0])
        top1_ratio = float(ans_counts.iloc[0]) / len(selected_df) if len(selected_df) > 0 else 0.0
        print(f"  unique_answers_sel  : {unique_ans}  (验收: 越高越好)")
        print(f"  top1_ratio          : {top1_ratio:.2%}  (验收: 越低越好, 应显著低于旧版)")
        if ans_col in df_all.columns:
            all_counts = df_all[ans_col].value_counts()
            unique_ans_all = int(all_counts.shape[0])
            top1_ratio_all = float(all_counts.iloc[0]) / len(df_all) if len(df_all) > 0 else 0.0
            print(f"  unique_answers_all  : {unique_ans_all}")
            print(f"  top1_ratio_all      : {top1_ratio_all:.2%}")

    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)
    s      = _resolve_settings(args, config)

    print(f"Loading scores from {args.scores}...")
    df = pd.read_parquet(args.scores)
    if len(df) == 0:
        raise ValueError("scores parquet is empty")

    score_col = s["score_col"]
    if score_col not in df.columns:
        raise KeyError(
            f"[P0-1] score_col={score_col!r} not found in parquet. "
            f"Available columns: {list(df.columns)}. "
            f"Check that 01_score_vicor.py wrote the correct score column, "
            f"or set score_col in your config to match."
        )

    if "lora_init_seed" in df.columns:
        seeds = sorted({int(x) for x in df["lora_init_seed"].dropna().unique().tolist()})
        if len(seeds) > 1:
            raise ValueError(
                f"scores parquet mixes LoRA initialization seeds {seeds}; every shard must be "
                "scored with the same vicor.seed so residual norms share one parameter space"
            )

    total = len(df)
    # When Stage-I uses a gamma cap, the scores parquet contains only the
    # top-gamma-K candidate pool.  The final selector must still return K
    # examples measured against the original pool size, rather than applying
    # the ratio a second time to the reduced candidate pool.
    stage1_cfg = config.get("stage1", {}) or {}
    budget_path = Path(str(config.get("out_dir", "."))) / "stage1_selection_budget.json"
    budget = None
    try:
        gamma = float(stage1_cfg.get("candidate_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        gamma = 0.0
    # Stage-I always writes a source-pool budget, including gamma=0 where no
    # optional cap is applied.  Read it whenever this run was produced by a
    # Stage-I config so K is never recomputed from the filtered candidate
    # table.  Requiring the config section prevents an old budget file in a
    # normal (non-Stage-I) output directory from changing its semantics.
    if "stage1" in config and budget_path.exists():
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        budget_gamma = float(budget.get("gamma", gamma))
        if not math.isclose(budget_gamma, gamma, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Stage-I budget gamma={budget_gamma} disagrees with config candidate_ratio={gamma}; "
                "rerun scoring before selection"
            )
    if budget is not None:
        budget_ratio = float(budget.get("final_selection_ratio", s["ratio"]))
        if args.ratio is not None and not math.isclose(float(args.ratio), budget_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "--ratio differs from the Stage-I scoring budget; rerun scoring with the matching final ratio"
            )
        expected_candidate_pool = int(budget.get("candidate_pool_size", total))
        if expected_candidate_pool != total:
            raise ValueError(
                f"Stage-I budget expects {expected_candidate_pool} scored candidates, got {total}; "
                "use the matching scores parquet and config"
            )
        k = max(1, min(total, int(budget["target_k"])))
    else:
        k = max(1, min(total, int(round(total * s["ratio"]))))
    strategy     = s["strategy"]
    balance_col  = s["balance_col"]
    has_bal      = bool(balance_col) and balance_col in df.columns

    if strategy in ("group_capped_topk", "round_robin") and not has_bal:
        if s["strict_balance_col"]:
            raise ValueError(f"strategy={strategy} requires balance_col={balance_col!r}")
        print(f"[Warn] {strategy} needs balance_col; fallback to vicor_mmr.")
        strategy = "vicor_mmr"

    if strategy == "group_capped_topk":
        _validate_group_constraints(df, k, balance_col, s["max_group_share"], s["min_group_samples"])

    print(f"Strategy={strategy}  ratio={s['ratio']:.2f}  k={k}/{total}  score={score_col}")
    print(f"MMR λ={s['mmr_lambda']}  candidate_ratio={s['mmr_candidate_ratio']}")
    if strategy == "cluster_vicor_mmr":
        print(
            f"Cluster selector: n_clusters={s['cluster_n_clusters'] or 'auto'}  "
            f"min_quota={s['cluster_min_quota']}  score_alpha={s['cluster_score_alpha']}"
        )
    if strategy == "cluster_floor_vicor_mmr":
        print(
            f"Cluster-floor selector: n_clusters={s['cluster_n_clusters'] or 'auto'}  "
            f"min_quota={s['cluster_min_quota']}  score_alpha={s['cluster_score_alpha']}  "
            f"floor_ratio={s['cluster_floor_ratio']}"
        )
    adaptive_meta = None
    if strategy == "adaptive_tail_cluster_vicor_mmr":
        adaptive_meta = _derive_adaptive_selector_meta(
            df=df,
            base_tail_reserve_ratio=s["tail_reserve_ratio"],
            base_cluster_floor_ratio=s["cluster_floor_ratio"],
        )
        print(
            f"Adaptive selector: open_scale={_meta_float(adaptive_meta, 'open_scale', 0.5):.3f}  "
            f"unique_signal={_meta_float(adaptive_meta, 'unique_signal', 0.5):.3f}  "
            f"concentration_signal={_meta_float(adaptive_meta, 'concentration_signal', 0.5):.3f}  "
            f"tail_signal={_meta_float(adaptive_meta, 'tail_signal', 0.5):.3f}"
        )
        print(
            f"Adaptive selector: tail_ratio_used={_meta_float(adaptive_meta, 'tail_ratio_used', s['tail_reserve_ratio']):.3f}  "
            f"cluster_floor_ratio_used={_meta_float(adaptive_meta, 'cluster_floor_ratio_used', s['cluster_floor_ratio']):.3f}"
        )
    if strategy in ("label_sparse_vicor_mmr", "label_sparse_tail_vicor_mmr", "label_floor_vicor_mmr"):
        print(
            f"V6 selector: label_align_alpha={s['label_align_alpha']}  "
            f"directness_alpha={s['directness_alpha']}  cardinality_alpha={s['cardinality_alpha']}  "
            f"target_cardinality={s['target_cardinality']}  cardinality_slack={s['cardinality_slack']}  "
            f"target_vocab={len(s['target_label_vocab'])}  alias_labels={len(s['target_label_aliases'])}"
        )
    if strategy == "label_floor_vicor_mmr":
        print(f"Label-floor coverage: min_samples_per_label={s['label_floor_min_samples']}")

    df_sorted = df.sort_values(by=score_col, ascending=False, kind="mergesort")

    # 构建特征矩阵（MMR 需要；对 topk / group_capped_topk 无害）
    print("Building feature matrix for MMR...")
    feat_matrix = _build_feature_matrix(df)   # df 保持原始 index 顺序

    target_label_counts_all = None
    target_label_counts_selected = None

    if strategy == "topk":
        selected_df = _select_topk(df_sorted, k)

    elif strategy == "group_capped_topk":
        selected_df = _select_group_capped_topk(
            df_sorted, k, score_col, balance_col,
            s["max_group_share"], s["min_group_samples"],
        )

    elif strategy == "round_robin":
        selected_df = _select_round_robin(df_sorted, k, score_col, balance_col)

    elif strategy == "paper_vicor_mmr":
        selected_df = _select_paper_vicor_mmr(
            df=df,
            k=k,
            base_score_col=score_col,
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
            target_label_vocab=s["target_label_vocab"],
            target_label_aliases=s["target_label_aliases"],
            label_align_alpha=s["label_align_alpha"],
            directness_alpha=s["directness_alpha"],
            cardinality_alpha=s["cardinality_alpha"],
            target_cardinality=s["target_cardinality"],
            cardinality_slack=s["cardinality_slack"],
            beta_a=s["beta_a"],
            beta_d=s["beta_d"],
        )
        report_score_col = "selector_score_vicor"

    elif strategy == "paper_group_vicor_mmr":
        selected_df = _select_paper_group_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=score_col,
            balance_col=balance_col,
            min_group_samples=s["min_group_samples"],
            max_group_share=s["max_group_share"],
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
        )

    elif strategy == "vicor_mmr":
        selected_df = _select_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=score_col,
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
        )

    elif strategy == "tail_vicor_mmr":
        selected_df = _select_tail_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=score_col,
            tail_reserve_ratio=s["tail_reserve_ratio"],
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
        )

    elif strategy == "cluster_vicor_mmr":
        selected_df = _select_cluster_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=score_col,
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
            cluster_n_clusters=s["cluster_n_clusters"],
            cluster_min_quota=s["cluster_min_quota"],
            cluster_score_alpha=s["cluster_score_alpha"],
        )

    elif strategy == "cluster_floor_vicor_mmr":
        selected_df = _select_cluster_floor_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=score_col,
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
            cluster_n_clusters=s["cluster_n_clusters"],
            cluster_min_quota=s["cluster_min_quota"],
            cluster_score_alpha=s["cluster_score_alpha"],
            cluster_floor_ratio=s["cluster_floor_ratio"],
        )

    elif strategy == "adaptive_tail_cluster_vicor_mmr":
        selected_df = _select_adaptive_tail_cluster_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=score_col,
            base_tail_reserve_ratio=s["tail_reserve_ratio"],
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
            cluster_n_clusters=s["cluster_n_clusters"],
            cluster_min_quota=s["cluster_min_quota"],
            cluster_score_alpha=s["cluster_score_alpha"],
            base_cluster_floor_ratio=s["cluster_floor_ratio"],
            adaptive_meta=adaptive_meta or _derive_adaptive_selector_meta(
                df=df,
                base_tail_reserve_ratio=s["tail_reserve_ratio"],
                base_cluster_floor_ratio=s["cluster_floor_ratio"],
            ),
        )

    elif strategy == "label_sparse_vicor_mmr":
        selector_score_col = '__selector_score_v6__'
        df = df.copy()
        df[selector_score_col] = _compute_v6_selector_score(
            df=df,
            base_score_col=score_col,
            target_label_vocab=s["target_label_vocab"],
            target_label_aliases=s["target_label_aliases"],
            label_align_alpha=s["label_align_alpha"],
            directness_alpha=s["directness_alpha"],
            cardinality_alpha=s["cardinality_alpha"],
            target_cardinality=s["target_cardinality"],
            cardinality_slack=s["cardinality_slack"],
        )
        df_sorted = df.sort_values(by=selector_score_col, ascending=False, kind='mergesort')
        selected_df = _select_cluster_floor_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=selector_score_col,
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
            cluster_n_clusters=s["cluster_n_clusters"],
            cluster_min_quota=s["cluster_min_quota"],
            cluster_score_alpha=s["cluster_score_alpha"],
            cluster_floor_ratio=s["cluster_floor_ratio"],
        )
        selected_df['base_score_for_report'] = selected_df[score_col]
        selected_df['selector_score_v6'] = selected_df[selector_score_col]
        report_score_col = selector_score_col

    elif strategy == "label_sparse_tail_vicor_mmr":
        selector_score_col = '__selector_score_v6__'
        df = df.copy()
        df[selector_score_col] = _compute_v6_selector_score(
            df=df,
            base_score_col=score_col,
            target_label_vocab=s["target_label_vocab"],
            target_label_aliases=s["target_label_aliases"],
            label_align_alpha=s["label_align_alpha"],
            directness_alpha=s["directness_alpha"],
            cardinality_alpha=s["cardinality_alpha"],
            target_cardinality=s["target_cardinality"],
            cardinality_slack=s["cardinality_slack"],
        )
        df_sorted = df.sort_values(by=selector_score_col, ascending=False, kind='mergesort')
        selected_df = _select_tail_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=selector_score_col,
            tail_reserve_ratio=s["tail_reserve_ratio"],
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
        )
        selected_df['base_score_for_report'] = selected_df[score_col]
        selected_df['selector_score_v6'] = selected_df[selector_score_col]
        report_score_col = selector_score_col

    elif strategy == "label_floor_vicor_mmr":
        selector_score_col = '__selector_score_v6__'
        df = df.copy()
        labels_series, _ = _build_answer_labels_series(df, s["target_label_vocab"], s["target_label_aliases"])
        df[selector_score_col] = _compute_v6_selector_score(
            df=df,
            base_score_col=score_col,
            target_label_vocab=s["target_label_vocab"],
            target_label_aliases=s["target_label_aliases"],
            label_align_alpha=s["label_align_alpha"],
            directness_alpha=s["directness_alpha"],
            cardinality_alpha=s["cardinality_alpha"],
            target_cardinality=s["target_cardinality"],
            cardinality_slack=s["cardinality_slack"],
        )
        df_sorted = df.sort_values(by=selector_score_col, ascending=False, kind='mergesort')
        selected_df = _select_label_floor_vicor_mmr(
            df=df,
            df_sorted=df_sorted,
            k=k,
            score_col=selector_score_col,
            mmr_lambda=s["mmr_lambda"],
            mmr_candidate_ratio=s["mmr_candidate_ratio"],
            feat_matrix=feat_matrix,
            cluster_n_clusters=s["cluster_n_clusters"],
            cluster_min_quota=s["cluster_min_quota"],
            cluster_score_alpha=s["cluster_score_alpha"],
            cluster_floor_ratio=s["cluster_floor_ratio"],
            labels_series=labels_series,
            target_label_vocab=s["target_label_vocab"],
            label_floor_min_samples=s["label_floor_min_samples"],
        )
        selected_df['base_score_for_report'] = selected_df[score_col]
        selected_df['selector_score_v6'] = selected_df[selector_score_col]
        report_score_col = selector_score_col
        target_label_counts_all = _count_target_labels_from_series(labels_series, s["target_label_vocab"])
        selected_labels_series, _ = _build_answer_labels_series(selected_df, s["target_label_vocab"], s["target_label_aliases"])
        target_label_counts_selected = _count_target_labels_from_series(
            selected_labels_series,
            s["target_label_vocab"],
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    report_score_col = locals().get("report_score_col", score_col)
    selected_df = _enforce_group_caps(
        df=df,
        selected_df=selected_df,
        k=k,
        score_col=report_score_col,
        balance_col=balance_col,
        max_group_share=s["max_group_share"],
    ).head(k).copy()
    selected_df["selection_rank"] = range(1, len(selected_df) + 1)
    selected_ids = selected_df["id"].tolist()

    # P0-5: 唯一性断言
    if len(set(selected_ids)) != len(selected_ids):
        dupes = len(selected_ids) - len(set(selected_ids))
        raise AssertionError(
            f"[P0-5] selected_ids contains {dupes} duplicate id(s). "
            f"This indicates a bug in the selection strategy."
        )
    if len(selected_ids) != k:
        raise RuntimeError(
            f"[P0-5] Expected {k} selected samples, got {len(selected_ids)}."
        )
    print(f"Final selection count: {len(selected_ids)}  (unique={len(set(selected_ids))})  ✓")

    report_score_col = locals().get('report_score_col', score_col)
    _print_diagnostics(df, selected_df, report_score_col)

    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "selected_ids.json"), "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, indent=2, ensure_ascii=False)
    selected_df.to_parquet(os.path.join(out_dir, "selection_details.parquet"), index=False)

    score_mode = str(df["score_mode"].iloc[0]) if "score_mode" in df.columns else ""
    summary = dict(
        scores_path=args.scores,
        strategy=strategy,
        ratio=s["ratio"],
        selected_k=len(selected_df),
        total_samples=total,
        source_pool_size=(int(budget["source_pool_size"]) if budget is not None else total),
        stage1_candidate_pool_size=(int(budget["candidate_pool_size"]) if budget is not None else total),
        score_col=score_col,
        report_score_col=report_score_col,
        score_mode=score_mode,
        mmr_lambda=s["mmr_lambda"],
        mmr_candidate_ratio=s["mmr_candidate_ratio"],
        cluster_n_clusters=s["cluster_n_clusters"],
        cluster_min_quota=s["cluster_min_quota"],
        cluster_score_alpha=s["cluster_score_alpha"],
        cluster_floor_ratio=s["cluster_floor_ratio"],
        label_align_alpha=s["label_align_alpha"],
        directness_alpha=s["directness_alpha"],
        cardinality_alpha=s["cardinality_alpha"],
        target_cardinality=s["target_cardinality"],
        label_floor_min_samples=s["label_floor_min_samples"],
        target_label_vocab=s["target_label_vocab"],
        target_label_aliases=s["target_label_aliases"],
        cardinality_slack=s["cardinality_slack"],
        tail_reserve_ratio=s["tail_reserve_ratio"],
        balance_col=balance_col,
        max_group_share=s["max_group_share"],
        min_group_samples=s["min_group_samples"],
        total_group_counts=_value_counts(df, balance_col),
        selected_group_counts=_value_counts(selected_df, balance_col),
        selected_score_min=float(selected_df[score_col].min()) if len(selected_df) else None,
        selected_score_max=float(selected_df[score_col].max()) if len(selected_df) else None,
        selected_score_mean=float(selected_df[score_col].mean()) if len(selected_df) else None,
        tail_in_all=int(df["is_tail"].sum()) if "is_tail" in df.columns else None,
        tail_in_selected=int(selected_df["is_tail"].sum()) if "is_tail" in selected_df.columns else None,
        adaptive_open_scale=_meta_float(adaptive_meta, "open_scale", 0.5) if adaptive_meta is not None else None,
        adaptive_unique_signal=_meta_float(adaptive_meta, "unique_signal", 0.5) if adaptive_meta is not None else None,
        adaptive_concentration_signal=_meta_float(adaptive_meta, "concentration_signal", 0.5) if adaptive_meta is not None else None,
        adaptive_tail_signal=_meta_float(adaptive_meta, "tail_signal", 0.5) if adaptive_meta is not None else None,
        adaptive_tail_ratio_used=_meta_float(adaptive_meta, "tail_ratio_used", s["tail_reserve_ratio"]) if adaptive_meta is not None else None,
        adaptive_cluster_floor_ratio_used=_meta_float(adaptive_meta, "cluster_floor_ratio_used", s["cluster_floor_ratio"]) if adaptive_meta is not None else None,
        # P0-5b: 答案多样性验收指标（无 fallback 即为健康）
        unique_answers_selected=(
            int(selected_df["answer_norm"].nunique())
            if "answer_norm" in selected_df.columns
            else int(selected_df["answer"].nunique())
            if "answer" in selected_df.columns
            else None
        ),
        target_label_counts_all=target_label_counts_all,
        target_label_counts_selected=target_label_counts_selected,
        top1_ratio=(
            float(selected_df["answer_norm"].value_counts().iloc[0] / len(selected_df))
            if "answer_norm" in selected_df.columns and len(selected_df) > 0
            else float(selected_df["answer"].value_counts().iloc[0] / len(selected_df))
            if "answer" in selected_df.columns and len(selected_df) > 0
            else None
        ),
        ids_unique=True,  # P0-5 assertion passed — no duplicates
    )
    with open(os.path.join(out_dir, "selection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved selected_ids.json / selection_details.parquet / selection_summary.json → {out_dir}")


if __name__ == "__main__":
    main()
