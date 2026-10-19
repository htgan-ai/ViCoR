#!/usr/bin/env python3
"""Run GEMeX/Qwen3-VL leave-one-component-out ViCoR ablations.

The script reuses an existing scoring parquet and never recomputes gradients.
Each invocation is intended to run one variant on one GPU:

  CUDA_VISIBLE_DEVICES=0 python scripts/run_gemex_component_ablation.py \
    --variant wo_counterfactual_residual \
    --scores /path/to/gemex_qwen3vl_scores.parquet \
    --base-config configs/gemex_qwen3vl_vicor.yaml \
    --work-root outputs/component_ablation_gemex_qwen3vl \
    --model-name-or-path /path/to/Qwen3-VL-4B-Instruct \
    --data-root /path/to/GEMeX \
    --image-root /path/to/GEMeX/images \
    --eval-root /path/to/eval
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vicor.config import load_config


FULL_SCORE_COL = "vicor_full_rank_score"
DIVERSITY_TOPK_SCORE_COL = "score_wo_diversity_reranking"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    display_name: str
    score_col: str
    selector_strategy: str
    description: str


VARIANTS: dict[str, VariantSpec] = {
    "wo_counterfactual_residual": VariantSpec(
        key="wo_counterfactual_residual",
        display_name="w/o Counterfactual residual",
        score_col="score_wo_counterfactual_residual",
        selector_strategy="label_sparse_tail_vicor_mmr",
        description="Use joint gradient energy instead of counterfactual residual energy.",
    ),
    "wo_shortcut_weighting": VariantSpec(
        key="wo_shortcut_weighting",
        display_name="w/o Shortcut weighting",
        score_col="score_wo_shortcut_weighting",
        selector_strategy="label_sparse_tail_vicor_mmr",
        description="Remove shortcut weighting w_s, keeping residual energy and directional correction.",
    ),
    "wo_directional_correction": VariantSpec(
        key="wo_directional_correction",
        display_name="w/o Directional correction",
        score_col="score_wo_directional_correction",
        selector_strategy="label_sparse_tail_vicor_mmr",
        description="Remove directional correction w_d, keeping residual energy and shortcut weighting.",
    ),
    "wo_residual_thresholding": VariantSpec(
        key="wo_residual_thresholding",
        display_name="w/o Residual thresholding",
        score_col="score_wo_residual_thresholding",
        selector_strategy="label_sparse_tail_vicor_mmr",
        description="Remove the residual-energy indicator gate.",
    ),
    "wo_supervision_calibration": VariantSpec(
        key="wo_supervision_calibration",
        display_name="w/o Supervision calibration",
        score_col=FULL_SCORE_COL,
        selector_strategy="paper_vicor_mmr",
        description="Keep diversity reranking but disable label/supervision calibration in the selector.",
    ),
    "wo_diversity_reranking": VariantSpec(
        key="wo_diversity_reranking",
        display_name="w/o Diversity reranking",
        score_col=DIVERSITY_TOPK_SCORE_COL,
        selector_strategy="topk",
        description="Select Top-K by calibrated full ranking score without MMR/diversity reranking.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--scores", required=True, help="Existing GEMeX/Qwen scoring parquet")
    parser.add_argument("--base-config", default="configs/gemex_qwen3vl_vicor.yaml")
    parser.add_argument("--work-root", default="outputs/component_ablation_gemex_qwen3vl")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-root", default="", help="Override config data_root")
    parser.add_argument("--image-root", action="append", default=[], help="Override/add image root; may be repeated")
    parser.add_argument("--dataset-jsonl-name", default="", help="Override data.jsonl_name")
    parser.add_argument("--eval-root", default=os.environ.get("EVAL_ROOT", ""))
    parser.add_argument("--smoke-eval-script", default="", help="Script for GMAI/MedBLINK/OmniMedVQA")
    parser.add_argument("--drvd-eval-script", default="", help="Script for DrVD")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--llamafactory-cli", default=os.environ.get("LLAMAFACTORY_CLI", "llamafactory-cli"))
    parser.add_argument("--export-script", default="", help="Prefer absolute/symlink export script")
    parser.add_argument("--dataset-key", default="", help="LLaMA-Factory dataset key; default is variant-specific")
    parser.add_argument(
        "--omvqa-max-samples",
        type=int,
        default=0,
        help="Maximum samples for the single-disease smoke/evaluation benchmarks; 0 evaluates the full benchmark.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--chex-max-new-tokens", type=int, default=32)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None, log_path: Path | None = None, dry_run: bool = False) -> None:
    rendered = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"[cmd] {rendered}", flush=True)
    if dry_run:
        return
    if log_path is None:
        subprocess.run([str(part) for part in cmd], check=True, env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {rendered}\n")
        log_file.flush()
        subprocess.run([str(part) for part in cmd], check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)


def require_columns(df: Any, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required scoring columns: {missing}. Available columns: {list(df.columns)}")


def safe_positive(series: Any) -> Any:
    return series.fillna(0.0).astype(float).clip(lower=0.0)


def load_v6_selector(selector_script: Path) -> Any:
    spec = importlib.util.spec_from_file_location("select_subset_for_ablation", selector_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import selector script: {selector_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compute_v6_selector_score


def assert_full_source_pool(scores_path: Path, df: Any, base_config: dict[str, Any]) -> int:
    """Reject score tables that do not cover the complete source pool.

    The ablation selects K = ratio x |source pool|.  A table truncated by the
    Stage-I gamma cap (or any partial table) would silently shrink K, so the
    entry point requires the full-pool scores produced by scripts/01_score_vicor.py
    (merged across shards when scoring was sharded).
    """
    total = int(len(df))
    problems: list[str] = []
    data_cfg = base_config.get("data", {}) or {}
    expected = data_cfg.get("training_pool_size", base_config.get("training_pool_size"))
    if expected is not None and int(expected) != total:
        problems.append(f"the base config declares training_pool_size={int(expected)} but the table has {total} rows")
    budget_path = scores_path.parent / "stage1_selection_budget.json"
    if budget_path.exists():
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        source = int(budget.get("source_pool_size", total))
        candidates = int(budget.get("candidate_pool_size", total))
        if candidates < source or total < source:
            problems.append(
                f"the Stage-I budget next to the scores keeps {candidates} of {source} source-pool rows "
                f"(gamma={budget.get('gamma')}); a gamma-capped table is not a full-pool table"
            )
    merged_manifest = scores_path.parent / "scoring_manifest_merged.json"
    single_manifest = scores_path.parent / "scoring_manifest_0.json"
    if merged_manifest.exists():
        manifest = json.loads(merged_manifest.read_text(encoding="utf-8"))
        rows = manifest.get("merged", {}).get("rows")
        if rows is not None and int(rows) != total:
            problems.append(f"the merged scoring manifest reports {int(rows)} rows but the table has {total}")
    elif single_manifest.exists():
        manifest = json.loads(single_manifest.read_text(encoding="utf-8"))
        if int(manifest.get("num_shards", 1)) != 1:
            problems.append("the scores come from a sharded run; merge the shards with scripts/merge_score_shards.py first")
        dataset_rows = manifest.get("dataset_rows")
        if dataset_rows is not None and int(dataset_rows) - int(manifest.get("oom_skipped", 0)) != total:
            problems.append(f"the scoring manifest reports dataset_rows={int(dataset_rows)} but the table has {total}")
    if problems:
        raise ValueError(
            "Component ablation requires the complete source-pool score table "
            f"({scores_path}): " + "; ".join(problems)
        )
    return total


def add_ablation_scores(
    scores_path: Path,
    output_path: Path,
    *,
    base_config: dict[str, Any],
    selector_script: Path,
    e_delta_floor: float,
) -> None:
    import pandas as pd

    df = pd.read_parquet(scores_path)
    require_columns(df, ["E_delta", "rho", "nov", "norm_joint", "seq_len"])
    assert_full_source_pool(scores_path, df, base_config)

    e_delta = safe_positive(df["E_delta"])
    rho = safe_positive(df["rho"])
    nov = safe_positive(df["nov"])
    # Eq. (4) uses the norm of the gradient of the token-averaged loss;
    # there is no additional sqrt(sequence_length) normalization.
    e_joint = safe_positive(df["norm_joint"])
    residual_gate = (e_delta >= float(e_delta_floor)).astype(float)
    joint_gate = (e_joint >= float(e_delta_floor)).astype(float)
    rho_min, rho_max = [float(x) for x in base_config.get("vicor", {}).get("rho_clip", [0.05, 20.0])]
    shortcut_weight = rho.map(lambda value: math.log1p(min(rho_max, max(rho_min, float(value)))))

    df[FULL_SCORE_COL] = residual_gate * e_delta * shortcut_weight * nov
    df["score_wo_counterfactual_residual"] = joint_gate * e_joint
    df["score_wo_shortcut_weighting"] = residual_gate * e_delta * nov
    df["score_wo_directional_correction"] = residual_gate * e_delta * shortcut_weight
    df["score_wo_residual_thresholding"] = e_delta * shortcut_weight * nov

    selector_cfg = base_config.get("vicor", {}).get("selection", {})
    compute_v6 = load_v6_selector(selector_script)
    df[DIVERSITY_TOPK_SCORE_COL] = compute_v6(
        df=df,
        base_score_col=FULL_SCORE_COL,
        target_label_vocab=selector_cfg.get("target_label_vocab", []),
        target_label_aliases=selector_cfg.get("target_label_aliases", {}),
        label_align_alpha=float(selector_cfg.get("label_align_alpha", 0.0)),
        directness_alpha=float(selector_cfg.get("directness_alpha", 0.0)),
        cardinality_alpha=float(selector_cfg.get("cardinality_alpha", 0.0)),
        target_cardinality=float(selector_cfg.get("target_cardinality", 0.0)),
        cardinality_slack=float(selector_cfg.get("eta", selector_cfg.get("cardinality_slack", 1.0))),
        beta_a=float(selector_cfg.get("beta_a", 0.70)),
        beta_d=float(selector_cfg.get("beta_d", 0.85)),
    )

    df["component_ablation_score_source"] = str(scores_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def expand_config_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_config_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_config_values(item) for key, item in value.items()}
    return value


def effective_selector_strategy(base_strategy: str, variant: VariantSpec) -> str:
    if variant.key == "wo_diversity_reranking":
        return "topk"
    if variant.key == "wo_supervision_calibration":
        # Keep the paper MMR construction; the calibration coefficients are
        # zeroed in write_variant_config below.
        return base_strategy
    return base_strategy or variant.selector_strategy


def write_variant_config(
    base_config: dict[str, Any],
    output_path: Path,
    variant: VariantSpec,
    variant_dir: Path,
    *,
    data_root: str,
    image_roots: list[str],
    dataset_jsonl_name: str,
    model_name_or_path: str,
) -> tuple[dict[str, Any], str]:
    import yaml

    config = expand_config_values(dict(base_config))
    config["out_dir"] = str(variant_dir / "selection")
    config["model_ckpt"] = model_name_or_path
    if data_root:
        config["data_root"] = data_root
    data_cfg = config.setdefault("data", {})
    data_cfg["skip_image_exists_scan"] = True
    if image_roots:
        data_cfg["image_roots"] = image_roots
    if dataset_jsonl_name:
        data_cfg["jsonl_name"] = dataset_jsonl_name
    data_cfg.setdefault("force_mean_image", False)

    config.setdefault("vicor", {}).setdefault("selection", {})
    base_strategy = str(config["vicor"]["selection"].get("strategy", variant.selector_strategy))
    strategy = effective_selector_strategy(base_strategy, variant)
    config["vicor"]["selection"]["score_col"] = variant.score_col
    config["vicor"]["selection"]["strategy"] = strategy
    if variant.key == "wo_supervision_calibration":
        selection = config["vicor"]["selection"]
        selection["label_align_alpha"] = 0.0
        selection["directness_alpha"] = 0.0
        selection["cardinality_alpha"] = 0.0
        selection["target_label_vocab"] = []
        selection["target_label_aliases"] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return config, strategy


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_eval_root_args(cmd: list[str], eval_root: str, kind: str) -> None:
    if not eval_root:
        return
    root = Path(eval_root)
    if kind == "smoke":
        cmd.extend(
            [
                "--gmai_val_tsv",
                str(root / "GMAI-MMBench" / "GMAI_mm_bench_VAL.tsv"),
                "--medblink_jsonl",
                str(root / "MedBLINK" / "metadata.jsonl"),
                "--medblink_root",
                str(root / "MedBLINK"),
                "--omni_open_dir",
                str(root / "OmniMedVQA" / "QA_information" / "Open-access"),
                "--omni_root",
                str(root / "OmniMedVQA"),
            ]
        )
    elif kind == "drvd":
        cmd.extend(
            [
                "--drvd_root",
                str(root / "DrVD-Bench"),
                "--drvd_visual_jsonl",
                str(root / "DrVD-Bench" / "visual_evidence_qa.jsonl"),
            ]
        )


def main() -> int:
    args = parse_args()
    variant = VARIANTS[args.variant]
    repo_root = Path(__file__).resolve().parents[1]
    work_root = Path(args.work_root).resolve()
    variant_dir = work_root / variant.key
    logs_dir = variant_dir / "logs"
    state_path = variant_dir / "state.json"
    augmented_scores = variant_dir / "scores" / "gemex_qwen3vl_component_ablation_scores.parquet"
    variant_config = variant_dir / "config.yaml"
    llama_data_dir = variant_dir / "llama_factory_data"
    adapter_dir = variant_dir / "qwen3vl_lora"
    eval_dir = variant_dir / "eval"
    dataset_key = args.dataset_key or f"gemex_component_{variant.key}"

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    write_state(
        state_path,
        {
            "variant": variant.key,
            "display_name": variant.display_name,
            "description": variant.description,
            "started_at": started_at,
            "status": "starting",
        },
    )

    import yaml

    base_config_path = Path(args.base_config)
    base_config = load_config(base_config_path)
    vicor_cfg = base_config.get("vicor", {})
    e_delta_floor = float(vicor_cfg.get("E_delta_floor", 2.5e-5))

    add_ablation_scores(
        Path(args.scores),
        augmented_scores,
        base_config=base_config,
        selector_script=repo_root / "scripts" / "02_select_subset.py",
        e_delta_floor=e_delta_floor,
    )
    _, selector_strategy = write_variant_config(
        base_config,
        variant_config,
        variant,
        variant_dir,
        data_root=args.data_root,
        image_roots=list(args.image_root),
        dataset_jsonl_name=args.dataset_jsonl_name,
        model_name_or_path=args.model_name_or_path,
    )

    write_state(
        state_path,
        {
            "variant": variant.key,
            "display_name": variant.display_name,
            "started_at": started_at,
            "status": "selecting_subset",
            "score_col": variant.score_col,
            "strategy": selector_strategy,
            "scores": str(augmented_scores),
            "config": str(variant_config),
        },
    )

    run(
        [
            args.python,
            "-u",
            str(repo_root / "scripts" / "02_select_subset.py"),
            "--scores",
            str(augmented_scores),
            "--config",
            str(variant_config),
            "--ratio",
            "0.2",
            "--score_col",
            variant.score_col,
            "--strategy",
            selector_strategy,
        ],
        log_path=logs_dir / "select.log",
        dry_run=args.dry_run,
    )

    export_script = Path(args.export_script) if args.export_script else repo_root / "scripts" / "03_export_to_llama_factory_absolute.py"
    if not export_script.exists():
        export_script = repo_root / "scripts" / "03_export_to_llama_factory_symlink.py"

    write_state(state_path, {"variant": variant.key, "started_at": started_at, "status": "exporting"})
    run(
        [
            args.python,
            "-u",
            str(export_script),
            "--config",
            str(variant_config),
            "--subset",
            str(variant_dir / "selection" / "selected_ids.json"),
            "--output_dir",
            str(llama_data_dir),
            "--dataset_name",
            dataset_key,
        ],
        log_path=logs_dir / "export.log",
        dry_run=args.dry_run,
    )

    if not args.skip_train:
        write_state(state_path, {"variant": variant.key, "started_at": started_at, "status": "training"})
        train_env = os.environ.copy()
        train_env.update(
            {
                "MODEL_NAME_OR_PATH": args.model_name_or_path,
                "DATASET_NAME": dataset_key,
                "DATASET_DIR": str(llama_data_dir),
                "OUTPUT_DIR": str(adapter_dir),
                "LLAMAFACTORY_CLI": args.llamafactory_cli,
                "PYTHONUNBUFFERED": "1",
            }
        )
        run(
            ["bash", str(repo_root / "training" / "train_qwen3vl_llamafactory_lora.sh")],
            env=train_env,
            log_path=logs_dir / "train.log",
            dry_run=args.dry_run,
        )

    if not args.skip_eval:
        write_state(state_path, {"variant": variant.key, "started_at": started_at, "status": "evaluating"})
        eval_env = os.environ.copy()
        if args.eval_root:
            eval_env["EVAL_ROOT"] = args.eval_root

        smoke_eval_script = Path(args.smoke_eval_script) if args.smoke_eval_script else repo_root / "evaluation" / "evaluate_single_disease_qwen3vl.py"
        smoke_cmd = [
            args.python,
            "-u",
            str(smoke_eval_script),
            "--adapter_path",
            str(adapter_dir),
            "--base_model_path",
            args.model_name_or_path,
            "--output_dir",
            str(eval_dir / "core4"),
            "--benchmarks",
            "gmai_val",
            "medblink_val",
            "omnimedvqa_open",
            "--max_samples",
            str(args.omvqa_max_samples),
            "--device",
            "cuda:0",
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--image_max_pixels",
            str(args.image_max_pixels),
            "--mcq_style",
            "strict",
        ]
        append_eval_root_args(smoke_cmd, args.eval_root, "smoke")
        run(smoke_cmd, env=eval_env, log_path=logs_dir / "eval_single.log", dry_run=args.dry_run)

        drvd_eval_script = Path(args.drvd_eval_script) if args.drvd_eval_script else repo_root / "evaluation" / "evaluate_drvd_qwen3vl.py"
        drvd_cmd = [
            args.python,
            "-u",
            str(drvd_eval_script),
            "--adapter_path",
            str(adapter_dir),
            "--base_model_path",
            args.model_name_or_path,
            "--output_dir",
            str(eval_dir / "drvd"),
            "--benchmarks",
            "drvd_visual_evidence_qa",
            "drvd_independent_qa",
            "drvd_joint_qa",
            "drvd_multi_turn_qa",
            "drvd_report_generation",
            "--max_samples",
            "0",
            "--device",
            "cuda:0",
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--image_max_pixels",
            str(args.image_max_pixels),
        ]
        append_eval_root_args(drvd_cmd, args.eval_root, "drvd")
        run(drvd_cmd, env=eval_env, log_path=logs_dir / "eval_drvd.log", dry_run=args.dry_run)

        if not args.eval_root:
            raise ValueError("--eval-root or EVAL_ROOT is required for CheXbench evaluation")
        run(
            [
                args.python,
                "-u",
                str(repo_root / "evaluation" / "evaluate_chexbench_qwen3vl.py"),
                "--adapter_path",
                str(adapter_dir),
                "--base_model_path",
                args.model_name_or_path,
                "--output_dir",
                str(eval_dir / "chexbench"),
                "--multiple_disease_json",
                str(Path(args.eval_root) / "Multiple_Disease_Identification" / "Multiple_Disease_Identification.json"),
                "--multiple_disease_root",
                str(Path(args.eval_root) / "Multiple_Disease_Identification" / "images"),
                "--max_samples",
                "0",
                "--max_new_tokens",
                str(args.chex_max_new_tokens),
                "--image_max_pixels",
                str(args.image_max_pixels),
            ],
            env=eval_env,
            log_path=logs_dir / "eval_chexbench.log",
            dry_run=args.dry_run,
        )
        run(
            [
                args.python,
                "-u",
                str(repo_root / "evaluation" / "aggregate_benchmark_results.py"),
                "--evaluation_dir",
                str(eval_dir),
                "--output",
                str(eval_dir / "aggregate_summary.json"),
            ],
            env=eval_env,
            log_path=logs_dir / "eval_aggregate.log",
            dry_run=args.dry_run,
        )

    write_state(
        state_path,
        {
            "variant": variant.key,
            "display_name": variant.display_name,
            "started_at": started_at,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed" if not args.dry_run else "dry_run_completed",
            "adapter_dir": str(adapter_dir),
            "eval_dir": str(eval_dir),
        },
    )
    print(f"[{variant.key}] pipeline finished; state={state_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
