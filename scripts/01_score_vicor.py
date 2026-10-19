import argparse
import json
import hashlib
import importlib.util
import os
import sys
import re
import math
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vicor.config import load_config
from vicor.data.gemex import GEMeXDataset
from vicor.utils.misc import set_seed


def _normalize_answer_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _safe_qcut_bins(series: pd.Series, q: int, labels: list[str], default_label: str) -> pd.Series:
    """Quantile bins with robust fallback when values are degenerate."""
    if len(series) == 0:
        return pd.Series([], dtype="object")
    unique = series.nunique(dropna=False)
    if unique <= 1:
        return pd.Series([default_label] * len(series), index=series.index, dtype="object")
    q_eff = max(1, min(q, unique))
    cut = pd.qcut(series, q=q_eff, labels=False, duplicates="drop")
    n_bins = int(cut.max()) + 1 if len(cut) else 1
    label_pool = labels[:n_bins] if len(labels) >= n_bins else labels + [labels[-1]] * (n_bins - len(labels))
    mapped = cut.map(lambda x: label_pool[int(x)] if pd.notna(x) else default_label)
    return mapped.astype("object")


def _safe_cut_bins(series: pd.Series, bins: int, labels: list[str], default_label: str) -> pd.Series:
    """均匀分箱（非分位数），适合rho/nov等分布不规则的指标。"""
    if len(series) == 0:
        return pd.Series([], dtype="object")
    unique = series.nunique(dropna=False)
    if unique <= 1:
        return pd.Series([default_label] * len(series), index=series.index, dtype="object")
    bins_eff = max(1, min(bins, unique))
    try:
        cut = pd.cut(series, bins=bins_eff, labels=False, duplicates="drop")
    except Exception:
        return pd.Series([default_label] * len(series), index=series.index, dtype="object")
    n_bins = int(cut.max()) + 1 if cut.notna().any() else 1
    label_pool = labels[:n_bins] if len(labels) >= n_bins else labels + [labels[-1]] * (n_bins - len(labels))
    mapped = cut.map(lambda x: label_pool[int(x)] if pd.notna(x) else default_label)
    return mapped.astype("object")


def _get_seq_len(batch: dict, mode: str) -> int:
    """
    获取最终答案 token 数 L = |Y|，用于审计和记录。

    不能用 attention_mask.sum()，因为其中包含图像 token；这里的 L 只记录
    最终答案监督范围。论文 Eq. (1) 已在模型 loss 中做 token 平均，Eq. (4)
    不再对梯度范数做额外的 sqrt(L) 缩放。

    正确做法：统计 labels 中非 -100 的 token 数，即 loss 实际覆盖的 Y token 数。
    Collator 已将此值预计算为 label_seq_lens / shortcut_label_seq_lens。
    """
    if mode == "joint":
        seq_lens = batch.get("label_seq_lens", None)
    else:
        seq_lens = batch.get("shortcut_label_seq_lens", None)

    if seq_lens is not None and len(seq_lens) > 0:
        return max(1, int(seq_lens[0]))

    # 兜底：从labels直接统计（性能略低但语义正确）
    if mode == "joint":
        labels = batch.get("labels", None)
    else:
        labels = batch.get("shortcut_labels", None)

    if labels is not None:
        n_valid = int((labels[0] != -100).sum().item())
        return max(1, n_valid)

    return 1


def _apply_stage1_gamma_cap(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the optional gamma*K cap *after* all rows have been scored.

    Stage-I quality filtering must not subsample before ViCoR scores exist.
    ``candidate_ratio`` is therefore interpreted as gamma and applied to the
    configured final selection ratio after this complete scoring pass.
    """
    stage1 = config.get("stage1", {}) or {}
    gamma = float(stage1.get("candidate_ratio", 0.0) or 0.0)
    if not math.isfinite(gamma):
        raise ValueError(f"stage1.candidate_ratio (gamma) must be finite, got {gamma}")
    if gamma < 0.0 or (0.0 < gamma < 1.0):
        raise ValueError(
            "stage1.candidate_ratio (gamma) must be 0 (disabled) or >= 1.0; "
            f"got {gamma}. A gamma below one cannot provide the final K examples."
        )
    # Ordinary runs have no Stage-I filtering and should not emit or consume
    # a stale source-pool budget.  A generated Stage-I config always contains
    # the ``stage1`` section, even when its gamma cap is explicitly zero.
    if gamma == 0.0 and "stage1" not in config:
        return df
    selection = config.get("vicor", {}).get("selection", {}) or {}
    final_ratio = float(selection.get("ratio", config.get("vicor", {}).get("top_ratio", 0.2)))
    if not math.isfinite(final_ratio) or final_ratio <= 0.0:
        raise ValueError(f"Final selection ratio must be positive for gamma cap, got {final_ratio}")
    # ``df`` may already be a filtered Stage-I candidate table.  The final
    # budget is defined against the fixed Appendix-B source pool, so carry the
    # preflight size through the config instead of silently using len(df).
    source_pool_size_raw = config.get("_source_pool_size")
    if source_pool_size_raw is None:
        # ``training_pool_size`` is optional for custom configs, but a
        # generated Stage-I config still records the unfiltered source file.
        # Count that file here so gamma=0 cannot silently switch the budget
        # denominator to the filtered candidate table.
        source_jsonl = stage1.get("source_jsonl")
        if source_jsonl:
            source_path = Path(str(source_jsonl)).expanduser()
            if not source_path.exists():
                raise FileNotFoundError(f"Stage-I source JSONL does not exist: {source_path}")
            source_pool_size_raw = _count_nonempty_jsonl_rows(source_path)
    source_pool_size = int(source_pool_size_raw if source_pool_size_raw is not None else len(df))
    if source_pool_size <= 0:
        raise ValueError(f"Stage-I source pool size must be positive, got {source_pool_size}")
    if len(df) > source_pool_size:
        raise ValueError(
            f"Scored candidate rows ({len(df)}) exceed the configured source pool "
            f"size ({source_pool_size})"
        )
    # The final K is always defined against the fixed source pool, including
    # gamma=0 (the default, where Stage-I quality filtering is enabled but the
    # optional score cap is disabled).  Persist this budget before returning so
    # the selector cannot accidentally recompute K from the filtered table.
    target_k = max(1, min(source_pool_size, int(round(source_pool_size * final_ratio))))
    num_shards = int(config.get("_num_shards", 1))
    shard_id = int(config.get("_shard_id", 0))
    if num_shards == 1 and len(df) < target_k:
        raise ValueError(
            f"Stage-I filtering left {len(df)} scored candidates, fewer than the "
            f"required final K={target_k} from the source pool of {source_pool_size}"
        )
    out_dir = str(config.get("out_dir", "."))
    os.makedirs(out_dir, exist_ok=True)
    if gamma == 0.0:
        budget = {
            "source_pool_size": source_pool_size,
            "final_selection_ratio": final_ratio,
            "target_k": target_k,
            "gamma": 0.0,
            "protocol": "source_pool_budget_only_no_gamma_cap",
        }
        if num_shards == 1:
            budget.update({"requested_candidate_pool_size": len(df), "candidate_pool_size": len(df)})
            budget_path = os.path.join(out_dir, "stage1_selection_budget.json")
        else:
            # A shard only knows its own rows.  The global candidate pool size is
            # written by scripts/merge_score_shards.py after it has verified that
            # the shards cover the complete table.
            budget.update({"shard_id": shard_id, "num_shards": num_shards, "shard_candidate_rows": len(df), "partial": True})
            budget_path = os.path.join(out_dir, f"stage1_selection_budget_shard{shard_id}.json")
        with open(budget_path, "w", encoding="utf-8") as handle:
            json.dump(budget, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return df
    budget_path = os.path.join(out_dir, "stage1_selection_budget.json")
    if int(config.get("_num_shards", 1)) != 1:
        raise ValueError("Stage-I gamma cap requires a single complete scoring pass; merge all shards before applying candidate_ratio")
    if df.empty:
        return df
    # Eq. (12)-(13) define the candidate ordering with the calibrated
    # supervision-aware S_rank score.  Reuse the exact selector implementation
    # before applying the optional gamma cap, rather than ranking by raw S_vis.
    selector_path = Path(__file__).with_name("02_select_subset.py")
    spec = importlib.util.spec_from_file_location("vicor_selector_for_stage1", selector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load calibrated selector from {selector_path}")
    selector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector_module)
    selection = config.get("vicor", {}).get("selection", {}) or {}
    # Use the configured ViCoR base score.  The release configs use
    # ``score_delta_log1p_rho_nov``; hard-coding the generic ``score`` here
    # would make a Stage-I cap silently rank a different objective from the
    # final selector.
    base_score_col = str(selection.get("score_col", "score") or "score")
    if base_score_col not in df.columns:
        raise KeyError(
            f"Stage-I calibration score_col={base_score_col!r} is not present in "
            f"the scoring table. Available columns: {list(df.columns)}"
        )
    rank_scores = selector_module._compute_v6_selector_score(
        df=df,
        base_score_col=base_score_col,
        target_label_vocab=selection.get("target_label_vocab", []),
        target_label_aliases=selection.get("target_label_aliases", {}),
        label_align_alpha=float(selection.get("label_align_alpha", 0.0)),
        directness_alpha=float(selection.get("directness_alpha", 0.0)),
        cardinality_alpha=float(selection.get("cardinality_alpha", 0.0)),
        target_cardinality=float(selection.get("target_cardinality", 0.0)),
        cardinality_slack=float(selection.get("eta", selection.get("cardinality_slack", 1.0))),
        beta_a=float(selection.get("beta_a", 0.70)),
        beta_d=float(selection.get("beta_d", 0.85)),
    )
    df = df.copy()
    df["stage1_rank_score"] = rank_scores.astype(float)
    # Record the calibration settings the ranking was computed with, so the
    # selector reuses it only under identical settings (see 02_select_subset.py).
    df["stage1_rank_signature"] = selector_module.stage1_rank_signature(
        base_score_col=base_score_col,
        target_label_vocab=selection.get("target_label_vocab", []),
        target_label_aliases=selection.get("target_label_aliases", {}),
        label_align_alpha=float(selection.get("label_align_alpha", 0.0)),
        directness_alpha=float(selection.get("directness_alpha", 0.0)),
        cardinality_alpha=float(selection.get("cardinality_alpha", 0.0)),
        target_cardinality=float(selection.get("target_cardinality", 0.0)),
        cardinality_slack=float(selection.get("eta", selection.get("cardinality_slack", 1.0))),
        beta_a=float(selection.get("beta_a", 0.70)),
        beta_d=float(selection.get("beta_d", 0.85)),
    )
    if not df["stage1_rank_score"].map(math.isfinite).all():
        raise ValueError("Stage-I calibrated rank score contains NaN or Inf")
    # Paper protocol: first determine the final K from the *source* pool,
    # then retain min(gamma*K, N) candidates.  Do not round/ceil N*ratio*gamma
    # directly, as that changes the budget when Stage-I removed rows.
    requested_target = min(source_pool_size, max(1, int(math.ceil(gamma * target_k))))
    # A quality filter may leave fewer rows than the requested gamma*K cap. In
    # that case retain every surviving row, while recording the actual budget
    # consumed by the candidate table so the final selector can validate it.
    target = min(requested_target, len(df))
    budget = {
        "source_pool_size": source_pool_size,
        "final_selection_ratio": final_ratio,
        "target_k": target_k,
        "gamma": gamma,
        "requested_candidate_pool_size": requested_target,
        "candidate_pool_size": target,
        "protocol": "score_all_then_keep_top_gamma_k",
    }
    with open(budget_path, "w", encoding="utf-8") as handle:
        json.dump(budget, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if target >= len(df):
        return df
    ranked = (
        df.assign(_stage1_id_sort=df["id"].astype(str))
        .sort_values(["stage1_rank_score", "_stage1_id_sort"], ascending=[False, True], kind="mergesort")
        .drop(columns=["_stage1_id_sort"])
    )
    kept_indices = set(ranked.head(target).index.tolist())
    dropped = df.loc[~df.index.isin(kept_indices)].copy()
    result = ranked.head(target).sort_index().copy()

    # Keep the original Stage-I audit truthful and emit a score-aware manifest
    # for sharded/parallel runs where the source audit may be updated later.
    manifest_path = str(stage1.get("audit_manifest", ""))
    if manifest_path and os.path.exists(manifest_path):
        audit_rows = []
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    audit_rows.append(json.loads(line))
        dropped_positions = {int(row.get("candidate_index")) for _, row in dropped.iterrows() if pd.notna(row.get("candidate_index"))}
        score_by_candidate = {
            int(row["candidate_index"]): row
            for _, row in df.iterrows()
            if pd.notna(row.get("candidate_index"))
        }
        for item in audit_rows:
            candidate_index = item.get("candidate_index")
            if candidate_index is not None and int(candidate_index) in score_by_candidate:
                scored_row = score_by_candidate[int(candidate_index)]
                item["vicor_score"] = float(scored_row[base_score_col])
                item["score"] = float(scored_row[base_score_col])
                item["rank_score"] = float(scored_row["stage1_rank_score"])
                if int(candidate_index) in dropped_positions:
                    item["status"] = "gamma_cap"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            for item in audit_rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    cap_manifest = os.path.join(str(config.get("out_dir", ".")), "stage1_scored_cap_manifest.jsonl")
    os.makedirs(os.path.dirname(cap_manifest) or ".", exist_ok=True)
    with open(cap_manifest, "w", encoding="utf-8") as handle:
        for _, row in df.sort_index().iterrows():
            row_id = row.get("id")
            if hasattr(row_id, "item"):
                row_id = row_id.item()
            handle.write(json.dumps({
                "candidate_index": int(row.get("candidate_index", row.name)),
                "id": row_id,
                "score": float(row[base_score_col]),
                "rank_score": float(row["stage1_rank_score"]),
                "status": "kept" if row.name in kept_indices else "gamma_cap",
                "gamma": gamma,
                "final_selection_ratio": final_ratio,
                "target": target,
            }, ensure_ascii=False) + "\n")
    print(f"Stage-I gamma cap applied after scoring: {target}/{len(df)} rows retained (gamma={gamma:g})")
    return result


def _format_cot(sample: dict) -> dict:
    reasoning = sample.get("reasoning", "")
    answer = sample.get("answer", "")
    if reasoning:
        sample["answer_cot"] = f"<think>{reasoning}</think>{answer}"
    else:
        sample["answer_cot"] = sample.get("answer", "")
    return sample


def _get_image_availability_stats(ds) -> tuple[int, int] | None:
    """
    Return (existing_images, total_samples) if dataset exposes per-item image availability,
    otherwise return None.
    """
    data = getattr(ds, "data", None)
    if not isinstance(data, list) or len(data) == 0:
        return None

    exist_flags = []
    for item in data:
        if isinstance(item, dict) and "_image_exists" in item:
            exist_flags.append(bool(item["_image_exists"]))

    if not exist_flags:
        return None

    existing = int(sum(exist_flags))
    total = int(len(exist_flags))
    return existing, total


def _run_image_preflight(ds, config: dict) -> None:
    """
    Optional hard guard to prevent scoring on datasets with too many missing images.

    Config keys under `data`:
      - min_image_exists_ratio: float in [0, 1], optional
      - min_image_exists_count: int >= 0, optional
    """
    data_cfg = config.get("data", {})
    min_ratio_cfg = data_cfg.get("min_image_exists_ratio", None)
    min_count_cfg = data_cfg.get("min_image_exists_count", None)

    if min_ratio_cfg is None and min_count_cfg is None:
        return

    stats = _get_image_availability_stats(ds)
    if stats is None:
        print("[Preflight] Image availability stats unavailable for this dataset; skipping checks.")
        return

    existing, total = stats
    ratio = (existing / total) if total > 0 else 0.0
    print(f"[Preflight] image availability: {existing}/{total} ({ratio:.2%})")

    if min_ratio_cfg is not None:
        min_ratio = float(min_ratio_cfg)
        if min_ratio < 0.0 or min_ratio > 1.0:
            raise ValueError(f"data.min_image_exists_ratio must be within [0,1], got {min_ratio}")
        if ratio < min_ratio:
            raise RuntimeError(
                f"Image availability preflight failed: ratio={ratio:.2%} < required {min_ratio:.2%}. "
                "Fix dataset image paths or lower threshold deliberately."
            )

    if min_count_cfg is not None:
        min_count = int(min_count_cfg)
        if min_count < 0:
            raise ValueError(f"data.min_image_exists_count must be >= 0, got {min_count}")
        if existing < min_count:
            raise RuntimeError(
                f"Image availability preflight failed: existing={existing} < required {min_count}. "
                "Fix dataset image paths or lower threshold deliberately."
            )


def _count_nonempty_jsonl_rows(path: str | os.PathLike[str]) -> int:
    """Count JSONL records without loading the full pool into memory."""
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _validate_training_pool_size(config: dict, ds_len: int) -> None:
    """Enforce the fixed Appendix-B pool size declared by the run config.

    A Stage-I candidate file is intentionally smaller after deterministic
    denoising/deduplication.  In that case validate the *source* JSONL against
    ``training_pool_size`` and report the filtered candidate count; otherwise
    require the loaded JSONL itself to have the configured size.
    """
    data_cfg = config.get("data", {}) or {}
    expected_raw = data_cfg.get("training_pool_size", config.get("training_pool_size"))
    if expected_raw is None:
        return
    try:
        expected = int(expected_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training_pool_size must be an integer, got {expected_raw!r}") from exc
    if expected <= 0:
        raise ValueError(f"training_pool_size must be positive, got {expected}")

    stage1_cfg = config.get("stage1", {}) or {}
    source_jsonl = stage1_cfg.get("source_jsonl")
    if source_jsonl:
        source_path = Path(str(source_jsonl)).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(
                f"Stage-I source JSONL declared by stage1.source_jsonl does not exist: {source_path}"
            )
        source_count = _count_nonempty_jsonl_rows(source_path)
        if source_count != expected:
            raise ValueError(
                f"Configured training_pool_size={expected}, but Stage-I source JSONL "
                f"contains {source_count} records: {source_path}. "
                "Use the fixed paper pool or update the configuration deliberately."
            )
        if ds_len > expected:
            raise ValueError(
                f"Stage-I candidate JSONL contains {ds_len} records, exceeding the "
                f"source training_pool_size={expected}"
            )
        # Preserve the fixed Appendix-B denominator for the later Stage-I
        # budget.  ``ds_len`` can be a filtered candidate count.
        config["_source_pool_size"] = source_count
        print(
            f"[Preflight] training pool source size verified: {source_count}; "
            f"Stage-I candidate rows after filtering: {ds_len}"
        )
        return

    if ds_len != expected:
        raise ValueError(
            f"Configured training_pool_size={expected}, but loaded JSONL contains {ds_len} records. "
            "This run would not match the fixed paper training pool."
        )
    config["_source_pool_size"] = expected
    print(f"[Preflight] training pool size verified: {ds_len}")

def load_dataset(config):
    dataset_name = config.get("dataset_name", "").lower()
    data_root = config.get("data_root")
    use_cot = config.get("data", {}).get("use_cot", True)
    jsonl_name = config.get("data", {}).get("jsonl_name", "data.jsonl")
    strict_image_paths = bool(config.get("data", {}).get("strict_image_paths", False))
    skip_image_exists_scan = bool(config.get("data", {}).get("skip_image_exists_scan", True))
    force_mean_image = bool(config.get("data", {}).get("force_mean_image", False))
    image_open_timeout_sec = float(config.get("data", {}).get("image_open_timeout_sec", 0.0))
    image_roots_cfg = config.get("data", {}).get("image_roots", [])
    if isinstance(image_roots_cfg, str):
        image_roots = [image_roots_cfg]
    elif isinstance(image_roots_cfg, list):
        image_roots = [str(p) for p in image_roots_cfg if str(p).strip()]
    else:
        raise ValueError("data.image_roots must be a string or a list of strings")

    if dataset_name in {"gemex", "s-chain", "schain", "cxr-counterfact", "cxr_counterfact", "custom_jsonl"}:
        ds = GEMeXDataset(
            data_root=data_root,
            jsonl_name=jsonl_name,
            use_cot=use_cot,
            strict_image_paths=strict_image_paths,
            image_roots=image_roots,
            skip_image_exists_scan=skip_image_exists_scan,
            force_mean_image=force_mean_image,
            image_open_timeout_sec=image_open_timeout_sec,
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "The release supports GEMeX, S-Chain, CXR-CounterFact, and custom JSONL inputs."
        )

    if dataset_name in {"gemex", "s-chain", "schain", "cxr-counterfact", "cxr_counterfact", "custom_jsonl"} and use_cot:
        for i in range(len(ds)):
            if "answer_cot" not in ds.data[i]:
                ds.data[i] = _format_cot(ds.data[i])

    _run_image_preflight(ds, config)
    _validate_training_pool_size(config, len(ds))

    return dataset_name, ds


def build_data_loader(config, dataset_name: str, ds):
    # Lazy import heavy processor/collator dependencies so `--preflight_only`
    # can run without full training stack installed.
    from vicor.data.processors import get_processor
    from vicor.data.collators import build_collator

    processor = get_processor(config["model_name"], config["model_ckpt"])
    collator = build_collator(
        config["model_name"],
        dataset_name,
        processor,
        data_cfg=config.get("data", {}),
    )
    # Appendix E: compute a source-specific per-channel mean after the actual
    # backbone preprocessing. Always use the full training pool, not a shard.
    from vicor.data.collators import channel_view
    source = ds.dataset if isinstance(ds, Subset) else ds
    total = torch.zeros(3, dtype=torch.float64)
    count = 0
    for item in tqdm(source, desc="Computing training-source visual mean"):
        pixels = collator([item])["pixel_values"]
        view = channel_view(pixels, config["model_name"]).to(torch.float64)
        total += view.sum(dim=(0, 2))
        count += view.shape[0] * view.shape[2]
    if count == 0:
        raise ValueError("Cannot compute shortcut control from an empty training pool")
    collator.shortcut_mean = (total / count).tolist()
    out = config["out_dir"]
    os.makedirs(out, exist_ok=True)
    source_path = os.path.join(config["data_root"], config.get("data", {}).get("jsonl_name", "data.jsonl"))
    with open(source_path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()
    # Identical content on every shard; each shard writes its own file.
    mean_manifest = {"model_name": config["model_name"], "model_ckpt": config["model_ckpt"],
                     "source_sha256": digest, "source_samples": len(source), "elements_per_channel": count,
                     "channel_mean": collator.shortcut_mean, "data_config": config.get("data", {})}
    manifest_name = "shortcut_mean_" + str(config.get("_shard_id", 0)) + ".json"
    with open(os.path.join(out, manifest_name), "w", encoding="utf-8") as handle:
        json.dump(mean_manifest, handle, indent=2)
    return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)



def build_shard_indices(total_size: int, num_shards: int, shard_id: int, max_samples: int | None = None) -> list[int]:
    if total_size < 0:
        raise ValueError(f"total_size must be >= 0, got {total_size}")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id {shard_id} for num_shards {num_shards}")

    indices = list(range(shard_id, total_size, num_shards))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive when provided, got {max_samples}")
        indices = indices[:max_samples]
    return indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id [0..num_shards-1]")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of scored samples after shard filtering",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Only run dataset/image preflight checks and exit without loading model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    config["_shard_id"] = args.shard_id
    config["_num_shards"] = args.num_shards

    vicor_cfg = config.get("vicor", {})
    score_mode = vicor_cfg.get("score_mode", "delta_log1p_rho_nov")
    E_delta_floor = float(vicor_cfg.get("E_delta_floor", 2.5e-5))
    # The paper scores gradients over the parameters that fine-tuning will
    # update (Sec. 3.1).  PEFT draws LoRA-A at random and sets LoRA-B to zero,
    # so the scored gradient lives in a projection determined by that draw.
    # Every scoring run and every shard must therefore use one fixed seed;
    # otherwise residual norms are not comparable across samples.
    lora_init_seed = int(vicor_cfg.get("seed", config.get("seed", 17)))
    print(f"ViCoR score_mode: {score_mode}, E_delta_floor: {E_delta_floor}, seed: {lora_init_seed}")

    # 0. Load Dataset + Preflight (before any heavy model dependency)
    dataset_name, ds = load_dataset(config)
    print(f"Loaded dataset '{dataset_name}' with {len(ds)} samples.")

    if args.preflight_only:
        print("Preflight check passed. Exiting due to --preflight_only.")
        return

    # Lazy import model/scoring stack for full scoring run.
    from vicor.models.qwen3_vl import build_qwen3_wrapper
    from vicor.scoring.gradients import build_count_sketch_projector, apply_count_sketch, compute_grad_stats
    from vicor.scoring.vicor_score import vicor_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model (seed first: the LoRA-A initialization defines the scored parameter space)
    print(f"Loading model: {config['model_name']} from {config['model_ckpt']}")
    if config['model_name'] != "qwen3_vl":
        raise ValueError(f"Unsupported model: {config['model_name']}")
    set_seed(lora_init_seed)
    model = build_qwen3_wrapper(config['model_ckpt'], config.get('lora', None))

    if not getattr(model, "skip_to_device", False):
        model.to(device)
    model.eval()

    # 2. Get Shared Params (LoRA)
    shared_params = model.get_shared_params()
    if not shared_params:
        raise ValueError("No shared params found! Check LoRA config.")
    print(f"Number of shared params tensors: {len(shared_params)}")

    grad_sketch_dim = int(vicor_cfg.get("grad_sketch_dim", 0))
    grad_sketch_seed = int(vicor_cfg.get("grad_sketch_seed", 17))
    if grad_sketch_dim < 0:
        raise ValueError(f"vicor.grad_sketch_dim must be >= 0, got {grad_sketch_dim}")
    grad_sketch_projector = None
    if grad_sketch_dim > 0:
        print(f"Gradient sketch enabled: dim={grad_sketch_dim}, seed={grad_sketch_seed}")

    # 3. Build Data Loader (shard BEFORE collator/model compute)
    num_shards = max(1, args.num_shards)
    shard_id = args.shard_id
    shard_indices = build_shard_indices(
        total_size=len(ds),
        num_shards=num_shards,
        shard_id=shard_id,
        max_samples=args.max_samples,
    )
    print(
        f"Shard setup: shard_id={shard_id}/{num_shards}, "
        f"assigned_samples={len(shard_indices)}, total_dataset={len(ds)}"
    )
    dataset = build_data_loader(config, dataset_name, Subset(ds, shard_indices))

    records = []
    scored_count = 0
    oom_skipped = 0

    # 4. Scoring Loop
    print("Starting scoring...")
    for local_idx, ex in enumerate(tqdm(dataset, total=len(shard_indices))):
        global_idx = shard_indices[local_idx] if local_idx < len(shard_indices) else local_idx
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ex.items()}

        # 记录最终答案监督长度；评分本身不再做额外长度缩放。
        seq_len_joint    = _get_seq_len(batch, mode="joint")
        seq_len_shortcut = _get_seq_len(batch, mode="shortcut")
        # 仅记录监督 token 数 L=|Y|；论文的 loss 已按 token 平均，
        # 梯度范数不再做额外的长度或 sqrt(L) 缩放。
        seq_len = max(1, seq_len_joint)

        sample_id       = batch["ids"][0] if "ids" in batch else "unknown"
        sample_type     = batch["types"][0] if "types" in batch and batch["types"] else ""
        sample_question = batch["meta_questions"][0] if "meta_questions" in batch and batch["meta_questions"] else ""
        sample_answer   = batch["meta_answers"][0] if "meta_answers" in batch and batch["meta_answers"] else ""
        sample_reasoning = batch["meta_reasonings"][0] if "meta_reasonings" in batch and batch["meta_reasonings"] else ""

        answer_norm = _normalize_answer_text(sample_answer)
        q_len = len(str(sample_question).split())
        a_len = len(str(sample_answer).split())
        r_len = len(str(sample_reasoning).split())

        try:
            # --- Joint Pass ---
            model.zero_grad()
            out_j = model.forward_loss(batch, mode="joint")
            gs_j = compute_grad_stats(model, shared_params, out_j.loss)

            # --- Shortcut Pass ---
            model.zero_grad()
            out_s = model.forward_loss(batch, mode="shortcut")
            gs_s = compute_grad_stats(model, shared_params, out_s.loss)

            # --- Calculate Score (paper: residual + direct gradient norms) ---
            res = vicor_score(
                gs_j.grad_vec,
                gs_s.grad_vec,
                seq_len=seq_len,
                eps=vicor_cfg.get('eps', 1e-8),
                rho_clip=tuple(vicor_cfg.get('rho_clip', [0.05, 20.0])),
                E_delta_floor=E_delta_floor,
                score_mode=score_mode,
            )

            sketch_record = {}
            if grad_sketch_dim > 0:
                if grad_sketch_projector is None:
                    grad_sketch_projector = build_count_sketch_projector(
                        input_dim=int(gs_j.grad_vec.numel()),
                        sketch_dim=grad_sketch_dim,
                        seed=grad_sketch_seed,
                    )
                grad_delta = gs_j.grad_vec - gs_s.grad_vec
                delta_sketch = apply_count_sketch(grad_delta, grad_sketch_projector, normalize=True)
                shortcut_sketch = apply_count_sketch(gs_s.grad_vec, grad_sketch_projector, normalize=True)
                sketch_record = {
                    **{f"delta_sketch_{i:02d}": float(delta_sketch[i].item()) for i in range(grad_sketch_dim)},
                    **{f"shortcut_sketch_{i:02d}": float(shortcut_sketch[i].item()) for i in range(grad_sketch_dim)},
                }

            records.append({
                "id": sample_id,
                "candidate_index": int(global_idx),
                "type": sample_type,
                "question": sample_question,
                "answer": sample_answer,
                "reasoning": sample_reasoning,
                "answer_norm": answer_norm,
                "q_len": q_len,
                "a_len": a_len,
                "r_len": r_len,
                "seq_len": int(batch["attention_mask"][0].sum().item()),
                "answer_token_count": seq_len,
                "seq_len_shortcut": seq_len_shortcut,
                "score": res.score,
                "score_mode": res.score_mode,
                "score_delta": res.score_delta,
                "score_delta_log1p_rho": res.score_delta_log1p_rho,
                "score_delta_nov": res.score_delta_nov,
                "score_delta_log1p_rho_nov": res.score_delta_log1p_rho_nov,
                # Eq. (4) gradient-norm statistics
                "E_delta": res.E_delta,
                "E_shortcut": res.E_shortcut,
                "rho": res.rho,
                "nov": res.nov,
                "loss_joint": gs_j.loss,
                "loss_shortcut": gs_s.loss,
                "norm_joint": res.norm_joint,
                "norm_shortcut": res.norm_shortcut,
                "norm_delta": res.norm_delta,
                "lora_init_seed": lora_init_seed,
                **sketch_record,
            })
        except torch.OutOfMemoryError as e:
            raise RuntimeError(f"Scoring OOM at id={sample_id}; no samples may be silently removed from the fixed paper pool") from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(f"Scoring OOM at id={sample_id}; no samples may be silently removed from the fixed paper pool") from e
            raise

        scored_count += 1

    # 5. Save Results
    if records:
        df = pd.DataFrame(records)
        df = _apply_stage1_gamma_cap(df, config)

        # ── 多维多样性分组（解决长尾问题）────────────────────────────────
        # 维度1: 答案频率分箱（保护稀有答案/长尾样本）
        ans_freq = df["answer_norm"].value_counts(dropna=False)
        df["answer_freq"] = df["answer_norm"].map(ans_freq).fillna(1).astype(int)
        df["group_size"] = df.groupby("answer_norm")["id"].transform("count").astype(int)
        df["group_rank"] = (
            df.groupby("answer_norm")["score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        df["answer_freq_bin"] = _safe_qcut_bins(
            df["answer_freq"],
            q=5,
            labels=["tail", "low", "mid", "high", "head"],
            default_label="mid",
        )

        # 维度2: 推理链长度分箱（平衡短/长CoT）
        df["r_len_bin"] = _safe_qcut_bins(
            df["r_len"],
            q=4,
            labels=["r_s", "r_m", "r_l", "r_xl"],
            default_label="r_m",
        )

        # 维度3: 问题长度分箱（平衡简单/复杂问题）
        df["q_len_bin"] = _safe_qcut_bins(
            df["q_len"],
            q=3,
            labels=["q_short", "q_mid", "q_long"],
            default_label="q_mid",
        )

        # 维度4: 答案长度分箱
        df["a_len_bin"] = _safe_qcut_bins(
            df["a_len"],
            q=3,
            labels=["a_short", "a_mid", "a_long"],
            default_label="a_mid",
        )

        # 诊断用分箱（不纳入分组键，避免对score本身的维度强制均衡导致高分区被稀释）
        df["E_delta_bin"] = _safe_cut_bins(
            df["E_delta"],
            bins=4,
            labels=["e_low", "e_mid", "e_high", "e_vhigh"],
            default_label="e_mid",
        )
        df["rho_bin"] = _safe_cut_bins(
            df["rho"],
            bins=3,
            labels=["rho_low", "rho_mid", "rho_high"],
            default_label="rho_mid",
        )

        # ── 组合分组键：基于任务语义维度（而非score维度）────────────────
        # 关键原则：分组键只用任务语义特征，不用 E_delta_bin/rho_bin 等score的函数。
        # 纳入score函数维度会强制把高分样本分散到各bin，稀释高质量核心集。
        #
        # group_key: 主分组（3维：答案频率 × CoT长度 × 问题长度）
        # 用于 group_capped_topk，确保每种"任务类型"都有代表，防止单一常见答案垄断
        df["group_key"] = (
            df["answer_freq_bin"].astype(str)
            + "|" + df["r_len_bin"].astype(str)
            + "|" + df["q_len_bin"].astype(str)
        )

        # group_key_coarse: 粗粒度分组（2维，用于round_robin，控制组数防止碎片化）
        df["group_key_coarse"] = (
            df["answer_freq_bin"].astype(str)
            + "|" + df["r_len_bin"].astype(str)
        )

        # ── 长尾标记（answer_freq == 1 或 freq_bin == tail） ────────────
        df["is_tail"] = (df["answer_freq"] == 1) | (df["answer_freq_bin"] == "tail")

        out_name = "scores.parquet" if num_shards == 1 else f"scores_shard{shard_id}.parquet"
        out_path = os.path.join(config['out_dir'], out_name)
        os.makedirs(config['out_dir'], exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved scores to {out_path}")
        print(f"OOM-skipped samples: {oom_skipped}")

        # Scoring manifest: everything needed to check that shards and reruns
        # share one scoring protocol (seed, control image, score settings).
        try:
            import transformers as _tf
            import peft as _peft
            versions = {"torch": torch.__version__, "transformers": _tf.__version__, "peft": _peft.__version__}
        except Exception:  # pragma: no cover - version metadata is informational
            versions = {"torch": torch.__version__}
        manifest = {
            "config": os.path.abspath(args.config),
            "model_ckpt": config["model_ckpt"],
            "lora": config.get("lora", {}),
            "lora_init_seed": lora_init_seed,
            "score_mode": score_mode,
            "E_delta_floor": E_delta_floor,
            "rho_clip": list(vicor_cfg.get("rho_clip", [0.05, 20.0])),
            "eps": float(vicor_cfg.get("eps", 1e-8)),
            "num_shards": num_shards,
            "shard_id": shard_id,
            "scored_rows": int(len(df)),
            "dataset_rows": int(len(ds)),
            "shard_rows": int(len(shard_indices)),
            "oom_skipped": int(oom_skipped),
            "shortcut_mean_manifest": "shortcut_mean_" + str(shard_id) + ".json",
            "versions": versions,
        }
        manifest_path = os.path.join(config["out_dir"], f"scoring_manifest_{shard_id}.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"Saved scoring manifest to {manifest_path}")

        # 打印分布诊断信息
        print(f"\n=== Score Distribution Diagnostics ===")
        print(f"Total samples: {len(df)}")
        print(f"Tail samples (is_tail=True): {df['is_tail'].sum()} ({df['is_tail'].mean():.2%})")
        print(f"Zero-score samples (visual redundant): {(df['score'] == 0).sum()} ({(df['score'] == 0).mean():.2%})")
        print(f"Score stats: mean={df['score'].mean():.4e}, std={df['score'].std():.4e}, "
              f"p25={df['score'].quantile(0.25):.4e}, p75={df['score'].quantile(0.75):.4e}")
        print(f"E_delta stats: mean={df['E_delta'].mean():.4e}, p50={df['E_delta'].median():.4e}")
        print(f"rho stats: mean={df['rho'].mean():.4f}, p50={df['rho'].median():.4f}")
        print(f"nov stats: mean={df['nov'].mean():.4f}, p50={df['nov'].median():.4f}")
        print(f"group_key unique count: {df['group_key'].nunique()}")
        print(f"group_key_coarse unique count: {df['group_key_coarse'].nunique()}")
        print(f"answer_freq_bin distribution:\n{df['answer_freq_bin'].value_counts().to_string()}")
    else:
        print("No records to save (dataset empty).")


if __name__ == "__main__":
    main()
