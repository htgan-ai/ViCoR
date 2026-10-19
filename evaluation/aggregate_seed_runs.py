#!/usr/bin/env python3
"""Aggregate point estimates from the independent seed evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n_seeds": len(values),
        "mean": mean(values) if values else 0.0,
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Evaluation output directories, one per seed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.run_dirs) != 3:
        raise ValueError(
            "Exactly three independent seed evaluation directories are required; "
            f"received {len(args.run_dirs)}"
        )
    resolved_dirs = [Path(run_dir).expanduser().resolve() for run_dir in args.run_dirs]
    if len(set(resolved_dirs)) != len(resolved_dirs):
        raise ValueError("Seed evaluation directories must be distinct")
    summaries = []
    for run_dir in resolved_dirs:
        path = run_dir / "aggregate_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing aggregate summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError(f"Aggregate summary must be a JSON object: {path}")
        summaries.append(summary)

    def mapping_keys(summary: dict[str, Any], field: str, path: Path) -> set[str]:
        value = summary.get(field)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: missing or invalid {field} mapping")
        return {str(key) for key in value}

    core_names = sorted(mapping_keys(summaries[0], "core_benchmarks", resolved_dirs[0]))
    drvd_names = sorted(mapping_keys(summaries[0], "drvd_subtasks", resolved_dirs[0]))
    drvd_level_names = sorted(mapping_keys(summaries[0], "drvd_level_scores", resolved_dirs[0]))
    if not core_names or not drvd_names or not drvd_level_names:
        raise ValueError("Seed summaries must contain non-empty core_benchmarks, drvd_subtasks, and drvd_level_scores")
    for index, (summary, run_dir) in enumerate(zip(summaries[1:], resolved_dirs[1:]), start=1):
        if mapping_keys(summary, "core_benchmarks", run_dir) != set(core_names):
            raise ValueError(f"core_benchmarks keys differ in seed summary: {run_dir}")
        if mapping_keys(summary, "drvd_subtasks", run_dir) != set(drvd_names):
            raise ValueError(f"drvd_subtasks keys differ in seed summary: {run_dir}")
        if mapping_keys(summary, "drvd_level_scores", run_dir) != set(drvd_level_names):
            raise ValueError(f"drvd_level_scores keys differ in seed summary: {run_dir}")

    report_metrics = ("bleu", "bert_f1")
    report = summaries[0].get("drvd_report_generation")
    if not isinstance(report, dict) or any(metric not in report for metric in report_metrics):
        raise ValueError(
            f"{resolved_dirs[0]}: drvd_report_generation must contain official "
            f"metrics {report_metrics}"
        )
    if report.get("metric_protocol") != "drvd_official_keyword_extraction":
        raise ValueError(
            f"{resolved_dirs[0]}: drvd_report_generation must use the official "
            "keyword-extraction protocol"
        )
    report_keys = set(report)
    report_modalities = set(report.get("by_modality", {})) if isinstance(report.get("by_modality", {}), dict) else set()
    for summary, run_dir in zip(summaries[1:], resolved_dirs[1:]):
        candidate_report = summary.get("drvd_report_generation")
        if (
            not isinstance(candidate_report, dict)
            or set(candidate_report) != report_keys
            or candidate_report.get("metric_protocol") != report.get("metric_protocol")
            or set(candidate_report.get("by_modality", {})) != report_modalities
        ):
            raise ValueError(
                f"drvd_report_generation keys differ in seed summary: {run_dir}"
            )

    chex_field = "chexbench_multiple_disease_identification"
    chex_names = mapping_keys(summaries[0], chex_field, resolved_dirs[0])
    if not {"accuracy", "macro_f1"}.issubset(chex_names):
        raise ValueError(f"{chex_field} must contain accuracy and macro_f1")
    for summary, run_dir in zip(summaries[1:], resolved_dirs[1:]):
        if mapping_keys(summary, chex_field, run_dir) != chex_names:
            raise ValueError(f"{chex_field} keys differ in seed summary: {run_dir}")

    option_summaries = [summary.get("chexbench_option_scores") for summary in summaries]
    if any(item is not None for item in option_summaries) and not all(isinstance(item, dict) for item in option_summaries):
        raise ValueError("chexbench_option_scores must be present for all three seeds or none")
    if all(isinstance(item, dict) for item in option_summaries):
        option_keys = set(option_summaries[0])
        expected_option_keys = {"choice_accuracy", "correct_option_probability", "near_miss_win_rate"}
        if not expected_option_keys.issubset(option_keys):
            raise ValueError("chexbench_option_scores is missing a required metric")
        for item, run_dir in zip(option_summaries[1:], resolved_dirs[1:]):
            if set(item) != option_keys:
                raise ValueError(f"chexbench_option_scores keys differ in seed summary: {run_dir}")

    output: dict[str, Any] = {
        "seeds": [str(path) for path in resolved_dirs],
        "core_benchmarks": {
            name: _stats([float(summary["core_benchmarks"][name]) for summary in summaries])
            for name in core_names
        },
        "drvd_subtasks": {
            name: _stats([float(summary["drvd_subtasks"][name]) for summary in summaries])
            for name in drvd_names
        },
        "drvd_level_scores": {
            name: _stats([float(summary["drvd_level_scores"][name]) for summary in summaries])
            for name in drvd_level_names
        },
        "drvd_average": _stats([float(summary["drvd_average"]) for summary in summaries]),
        "drvd_report_generation": {
            metric: _stats([
                float(summary["drvd_report_generation"][metric])
                for summary in summaries
            ])
            for metric in report_metrics
        },
        "cross_benchmark_average": _stats([float(summary["cross_benchmark_average"]) for summary in summaries]),
        "chexbench": {
            metric: _stats([
                float(summary["chexbench_multiple_disease_identification"][metric])
                for summary in summaries
            ])
            for metric in ("accuracy", "macro_f1")
        },
    }
    if all(isinstance(item, dict) for item in option_summaries):
        output["chexbench_option_scores"] = {
            metric: _stats([float(item[metric]) for item in option_summaries if item is not None])
            for metric in ("choice_accuracy", "correct_option_probability", "near_miss_win_rate")
        }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
