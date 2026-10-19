#!/usr/bin/env python3
"""Run the complete Qwen3-VL evaluation chain used by the paper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("COMMAND: " + " ".join(cmd) + "\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluation failed with exit code {proc.returncode}; see {log_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--eval_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--drvd_multiturn_jsonl", default=os.environ.get("DRVD_MULTITURN_JSONL", ""))
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--report_max_new_tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image_max_pixels", type=int, default=262144)
    p.add_argument("--image_min_pixels", type=int, default=50176)
    p.add_argument(
        "--deepseek_api_key",
        default=os.environ.get("DEEPSEEK_API_KEY", ""),
        help="DeepSeek key required for the official DrVD report-generation metric.",
    )
    p.add_argument(
        "--report_metric_mode",
        choices=["official", "raw"],
        default="official",
        help="Official DrVD keyword extraction, or explicit raw smoke-test mode.",
    )
    args = p.parse_args()

    root = Path(args.eval_root)
    out = Path(args.output_dir)
    evaluation_dir = Path(__file__).resolve().parent
    repo_root = evaluation_dir.parent
    out.mkdir(parents=True, exist_ok=True)

    core_cmd = [
        args.python_bin, str(evaluation_dir / "evaluate_single_disease_qwen3vl.py"),
        "--adapter_path", args.adapter_path, "--base_model_path", args.base_model_path,
        "--output_dir", str(out / "core4"), "--benchmarks", "gmai_val", "medblink_val", "omnimedvqa_open",
        "--gmai_val_tsv", str(root / "GMAI-MMBench/GMAI_mm_bench_VAL.tsv"),
        "--medblink_jsonl", str(root / "MedBLINK/metadata.jsonl"), "--medblink_root", str(root / "MedBLINK"),
        "--omni_open_dir", str(root / "OmniMedVQA/QA_information/Open-access"),
        "--omni_root", str(root / "OmniMedVQA"), "--max_samples", str(args.max_samples),
        "--max_new_tokens", str(args.max_new_tokens), "--device", args.device,
        "--image_max_pixels", str(args.image_max_pixels), "--image_min_pixels", str(args.image_min_pixels),
    ]
    drvd_root = root / "DrVD-Bench"
    multiturn_jsonl = args.drvd_multiturn_jsonl or str(drvd_root / "multi_turn_qa.jsonl")
    drvd_cmd = [
        args.python_bin, str(evaluation_dir / "evaluate_drvd_qwen3vl.py"),
        "--adapter_path", args.adapter_path, "--base_model_path", args.base_model_path,
        "--output_dir", str(out / "drvd"), "--drvd_root", str(drvd_root),
        "--drvd_visual_jsonl", str(drvd_root / "visual_evidence_qa.jsonl"),
        "--drvd_independent_jsonl", str(drvd_root / "independent_qa.jsonl"),
        "--drvd_joint_jsonl", str(drvd_root / "joint_qa.jsonl"),
        "--drvd_multiturn_jsonl", multiturn_jsonl,
        "--drvd_report_jsonl", str(drvd_root / "report_generation.jsonl"),
        "--max_samples", str(args.max_samples), "--max_new_tokens", str(args.max_new_tokens),
        "--report_max_new_tokens", str(args.report_max_new_tokens),
        "--device", args.device, "--image_max_pixels", str(args.image_max_pixels),
        "--image_min_pixels", str(args.image_min_pixels),
        "--report_metric_mode", args.report_metric_mode,
    ]
    drvd_env = os.environ.copy()
    if args.deepseek_api_key:
        # Keep secrets out of commands.json and run.log; the evaluator reads
        # this environment variable as its default.
        drvd_env["DEEPSEEK_API_KEY"] = args.deepseek_api_key
    chex_cmd = [
        args.python_bin, str(evaluation_dir / "evaluate_chexbench_qwen3vl.py"),
        "--adapter_path", args.adapter_path, "--base_model_path", args.base_model_path,
        "--output_dir", str(out / "chexbench"),
        "--multiple_disease_json", str(root / "Multiple_Disease_Identification/Multiple_Disease_Identification.json"),
        "--multiple_disease_root", str(root / "Multiple_Disease_Identification/images"),
        "--max_samples", str(args.max_samples), "--max_new_tokens", str(args.max_new_tokens),
        "--device", args.device, "--image_max_pixels", str(args.image_max_pixels),
        "--image_min_pixels", str(args.image_min_pixels),
    ]
    option_cmd = [
        args.python_bin, str(evaluation_dir / "evaluate_chexbench_option_scores_qwen3vl.py"),
        "--adapter_path", args.adapter_path, "--base_model_path", args.base_model_path,
        "--benchmark_json", str(root / "Multiple_Disease_Identification/Multiple_Disease_Identification.json"),
        "--image_root", str(root / "Multiple_Disease_Identification/images"),
        "--config", str(repo_root / "configs/cxr_multi_qwen3vl_vicor.yaml"),
        "--output", str(out / "chexbench_option_scores.json"),
        "--max_samples", str(args.max_samples), "--device", args.device,
        "--image_max_pixels", str(args.image_max_pixels), "--image_min_pixels", str(args.image_min_pixels),
    ]
    commands = {"core4": core_cmd, "drvd": drvd_cmd, "chexbench": chex_cmd, "chexbench_option_scores": option_cmd}
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    run(core_cmd, out / "core4/run.log")
    run(drvd_cmd, out / "drvd/run.log", env=drvd_env)
    run(chex_cmd, out / "chexbench/run.log")
    run(option_cmd, out / "chexbench_option_scores.log")

    aggregate_cmd = [
        args.python_bin, str(evaluation_dir / "aggregate_benchmark_results.py"),
        "--evaluation_dir", str(out), "--output", str(out / "aggregate_summary.json"),
    ]
    run(aggregate_cmd, out / "aggregate.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
