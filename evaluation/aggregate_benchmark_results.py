#!/usr/bin/env python3
"""Aggregate the paper's core, five-part DrVD, and CheXbench summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a list of summary records in {path}")
    return payload


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        benchmark = str(row.get("benchmark", "")).strip()
        if not benchmark:
            continue
        if benchmark in indexed:
            raise ValueError(f"Duplicate benchmark summary record: {benchmark!r}")
        indexed[benchmark] = row
    return indexed


def aggregate(evaluation_dir: Path) -> dict[str, Any]:
    core = _index(_load_list(evaluation_dir / "core4" / "smoke_summary.json"))
    drvd = _index(_load_list(evaluation_dir / "drvd" / "drvd_summary.json"))
    chex = _index(_load_list(evaluation_dir / "chexbench" / "smoke_summary.json"))

    required_core = {
        "omnimedvqa_open": "OMVQA",
        "gmai_val": "GMAI",
        "medblink_val": "MedBLINK",
    }
    missing = [name for name in required_core if name not in core]
    required_drvd_qa = {
        "drvd_visual_evidence_qa",
        "drvd_independent_qa",
        "drvd_joint_qa",
        "drvd_multi_turn_qa",
    }
    missing.extend(sorted(required_drvd_qa - set(drvd)))
    if "drvd_report_generation" not in drvd:
        missing.append("drvd_report_generation")
    if "multiple_disease_identification" not in chex:
        missing.append("multiple_disease_identification")
    if missing:
        raise ValueError("Missing required benchmark summaries: " + ", ".join(missing))

    core_scores = {label: float(core[key]["accuracy"]) for key, label in required_core.items()}
    # Report generation is a free-form task.  The official DrVD protocol
    # reports BLEU and BERTScore F1 and has no accuracy field.  Keep it
    # separate from the accuracy mean over the three objective QA subtasks.
    report_row = drvd["drvd_report_generation"]
    for metric in ("bleu", "bert_f1"):
        if metric not in report_row:
            raise ValueError(
                "drvd_report_generation must contain official report metric "
                f"'{metric}', not an exact-match accuracy"
            )
    report_scores = {
        "bleu": float(report_row["bleu"]),
        "bert_f1": float(report_row["bert_f1"]),
    }
    protocol = str(report_row.get("metric_protocol", ""))
    if protocol != "drvd_official_keyword_extraction":
        raise ValueError(
            "drvd_report_generation uses a non-official metric protocol "
            f"({protocol!r}); rerun with --report_metric_mode official before aggregation"
        )
    by_modality = report_row.get("by_modality")
    if not isinstance(by_modality, dict) or not by_modality:
        raise ValueError(
            "drvd_report_generation must contain non-empty modality-grouped BLEU/BERTScore results"
        )
    if str(report_row.get("modality_aggregation", "")) != "macro":
        raise ValueError(
            "drvd_report_generation must use the official modality macro aggregation"
        )
    drvd_scores = {key: float(drvd[key]["accuracy"]) for key in sorted(required_drvd_qa)}
    level_scores: dict[str, float] = {}
    level_counts: dict[str, Any] = {}
    for benchmark in sorted(required_drvd_qa):
        row = drvd[benchmark]
        levels = row.get("level_accuracies")
        if not isinstance(levels, dict) or not levels:
            raise ValueError(
                f"{benchmark} must contain non-empty level_accuracies; "
                "DrVD accuracy is averaged within levels/tasks before benchmark aggregation"
            )
        for level, value in levels.items():
            level_scores[f"{benchmark}:{level}"] = float(value)
        counts = row.get("level_counts", {})
        if isinstance(counts, dict):
            level_counts[benchmark] = counts
    drvd_average = mean(level_scores.values())
    cross_benchmark = {**core_scores, "DRVD": drvd_average}
    chex_row = chex["multiple_disease_identification"]
    option_path = evaluation_dir / "chexbench_option_scores.json"
    option_scores = json.loads(option_path.read_text(encoding="utf-8")) if option_path.exists() else None

    return {
        "core_benchmarks": core_scores,
        "drvd_subtasks": drvd_scores,
        "drvd_level_scores": level_scores,
        "drvd_level_counts": level_counts,
        "drvd_average_protocol": "macro_mean_over_all_drvd_levels_and_tasks",
        "drvd_average": round(drvd_average, 6),
        "drvd_report_generation": {
            **report_scores,
            "samples": int(report_row.get("metric_samples", report_row.get("samples", 0))),
            "metric_protocol": protocol,
            "modality_aggregation": report_row.get("modality_aggregation", "macro"),
            "by_modality": by_modality,
        },
        "cross_benchmark_average": round(mean(cross_benchmark.values()), 6),
        "chexbench_multiple_disease_identification": {
            "accuracy": float(chex_row["accuracy"]),
            "macro_f1": float(chex_row.get("macro_f1", 0.0)),
            "samples": int(chex_row.get("samples", 0)),
        },
        "chexbench_option_scores": option_scores,
        "source_files": {
            "core": str(evaluation_dir / "core4" / "smoke_summary.json"),
            "drvd": str(evaluation_dir / "drvd" / "drvd_summary.json"),
            "chexbench": str(evaluation_dir / "chexbench" / "smoke_summary.json"),
            "chexbench_option_scores": str(option_path) if option_path.exists() else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate(Path(args.evaluation_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
