#!/usr/bin/env python3
"""Merge sharded ViCoR score tables into one ``scores.parquet``.

Shards are only comparable when they were produced by the same scoring
protocol: identical LoRA initialization seed, identical shortcut (mean-image)
control, identical score settings.  This script verifies those invariants
from the per-shard manifests before concatenating.

Usage:
    python scripts/merge_score_shards.py --out_dir outputs/gemex_vicor/gemex_qwen3vl_vicor
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, help="Directory containing scores_shard*.parquet")
    parser.add_argument("--output", default="", help="Merged parquet path (default: <out_dir>/scores.parquet)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    shards = sorted(out_dir.glob("scores_shard*.parquet"), key=lambda p: int(p.stem.split("shard")[-1]))
    if not shards:
        raise FileNotFoundError(f"No scores_shard*.parquet files in {out_dir}")

    manifests = []
    for shard in shards:
        shard_id = int(shard.stem.split("shard")[-1])
        manifest_path = out_dir / f"scoring_manifest_{shard_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing {manifest_path}; rerun scoring with the current release")
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    reference = manifests[0]
    for key in ("lora_init_seed", "score_mode", "E_delta_floor", "rho_clip", "eps", "model_ckpt", "lora", "num_shards"):
        values = [m.get(key) for m in manifests]
        if any(v != reference.get(key) for v in values):
            raise ValueError(f"Shards disagree on {key}: {values}")
    if len(shards) != int(reference["num_shards"]):
        raise ValueError(f"Expected {reference['num_shards']} shards, found {len(shards)}")
    if any(float(m.get("E_delta_floor", 0.0)) != float(reference.get("E_delta_floor", 0.0)) for m in manifests):
        raise ValueError("Shards disagree on E_delta_floor")

    means = []
    for m in manifests:
        mean_path = out_dir / str(m["shortcut_mean_manifest"])
        if not mean_path.exists():
            raise FileNotFoundError(f"Missing shortcut-mean manifest {mean_path}")
        means.append(json.loads(mean_path.read_text(encoding="utf-8")))
    for mean in means[1:]:
        if mean.get("source_sha256") != means[0].get("source_sha256"):
            raise ValueError("Shards were computed from different training-pool files")
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(mean["channel_mean"], means[0]["channel_mean"])):
            raise ValueError("Shards used different shortcut-control channel means")

    frames = [pd.read_parquet(shard) for shard in shards]
    columns = list(frames[0].columns)
    for shard, frame in zip(shards[1:], frames[1:]):
        if list(frame.columns) != columns:
            raise ValueError(f"{shard} has different columns from {shards[0]}")
    merged = pd.concat(frames, ignore_index=True)
    if merged["candidate_index"].duplicated().any():
        raise ValueError("Duplicate candidate_index values across shards")
    merged = merged.sort_values("candidate_index", kind="mergesort").reset_index(drop=True)

    # Complete coverage: every row of the scored table must come from exactly
    # one shard, and every candidate index of the dataset must be present
    # unless the shard reported it as skipped.
    dataset_rows = reference.get("dataset_rows")
    if dataset_rows is not None:
        dataset_rows = int(dataset_rows)
        if any(int(m.get("dataset_rows", -1)) != dataset_rows for m in manifests):
            raise ValueError("Shards disagree on dataset_rows")
        skipped = sum(int(m.get("oom_skipped", 0)) for m in manifests)
        shard_rows = sum(int(m.get("shard_rows", m.get("scored_rows", 0))) for m in manifests)
        if shard_rows != dataset_rows:
            raise ValueError(f"Shard assignments cover {shard_rows} rows but the dataset has {dataset_rows}")
        if len(merged) + skipped != dataset_rows:
            raise ValueError(f"Merged rows ({len(merged)}) plus skipped rows ({skipped}) do not equal dataset_rows ({dataset_rows})")
        indices = set(int(x) for x in merged["candidate_index"].tolist())
        if min(indices) < 0 or max(indices) >= dataset_rows:
            raise ValueError("candidate_index outside the dataset range")
        missing = sorted(set(range(dataset_rows)) - indices)
        if missing and skipped == 0:
            raise ValueError(f"Shards do not cover candidate indices {missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        skipped = 0

    output = Path(args.output) if args.output else out_dir / "scores.parquet"
    merged.to_parquet(output, index=False)

    # Global Stage-I budget assembled from the per-shard partial budgets.
    shard_budgets = sorted(out_dir.glob("stage1_selection_budget_shard*.json"))
    if shard_budgets:
        budgets = [json.loads(p.read_text(encoding="utf-8")) for p in shard_budgets]
        if len(budgets) != len(shards):
            raise ValueError(f"Found {len(budgets)} shard budgets for {len(shards)} shards")
        for key in ("source_pool_size", "final_selection_ratio", "target_k", "gamma", "protocol"):
            if any(b.get(key) != budgets[0].get(key) for b in budgets):
                raise ValueError(f"Shard budgets disagree on {key}")
        if sum(int(b.get("shard_candidate_rows", 0)) for b in budgets) != len(merged):
            raise ValueError("Shard candidate rows do not add up to the merged table")
        target_k = int(budgets[0]["target_k"])
        if len(merged) < target_k:
            raise ValueError(f"Merged candidate table has {len(merged)} rows, fewer than the required final K={target_k}")
        global_budget = {k: budgets[0][k] for k in ("source_pool_size", "final_selection_ratio", "target_k", "gamma", "protocol")}
        global_budget.update({"requested_candidate_pool_size": int(len(merged)), "candidate_pool_size": int(len(merged)),
                              "merged_from_shards": len(budgets), "skipped_rows": int(skipped)})
        (out_dir / "stage1_selection_budget.json").write_text(json.dumps(global_budget, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {"shards": len(shards), "rows": int(len(merged)), "skipped_rows": int(skipped), "lora_init_seed": reference["lora_init_seed"], "output": str(output)}
    (out_dir / "scoring_manifest_merged.json").write_text(json.dumps({**reference, "merged": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
