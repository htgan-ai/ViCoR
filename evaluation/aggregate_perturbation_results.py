#!/usr/bin/env python3
"""Compute paired perturbation drops and bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prediction_map(payload: dict[str, Any], path: str) -> dict[str, dict[str, Any]]:
    """Load predictions without silently changing the paired sample set.

    The perturbation estimand is paired at the image/sample level.  A missing
    row or a duplicate ID therefore changes the estimand and must be reported
    before any intersection/filtering is performed.
    """
    raw_rows = payload.get("predictions")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{path} has no non-empty predictions list")
    rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} prediction {index} is not an object")
        if row.get("id") is None or not str(row.get("id")).strip():
            raise ValueError(f"{path} prediction {index} has an empty id")
        sample_id = str(row["id"])
        if sample_id in rows:
            raise ValueError(f"{path} contains duplicate prediction id {sample_id!r}")
        if not isinstance(row.get("correct"), bool):
            raise ValueError(
                f"{path} prediction {sample_id!r} has non-boolean correct="
                f"{row.get('correct')!r}"
            )
        benchmark = str(row.get("benchmark", "")).strip()
        if not benchmark:
            raise ValueError(f"{path} prediction {sample_id!r} has no benchmark")
        rows[sample_id] = {
            "correct": row["correct"],
            "benchmark": benchmark,
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition_a", required=True, help="JSON output for the original/ground-truth condition")
    parser.add_argument("--condition_b", required=True, help="JSON output for the perturbed/background condition")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--difference",
        choices=["a_minus_b", "b_minus_a"],
        default="a_minus_b",
        help=(
            "Accuracy difference to report. The default is correct_A - correct_B; "
            "use b_minus_a for the paper's evidence gap when A is the ground-truth "
            "mask and B is the background mask."
        ),
    )
    parser.add_argument("--shared_ids", default="", help="Optional newline-delimited IDs for a shared-correct subset")
    parser.add_argument(
        "--expected_shared_count",
        type=int,
        default=0,
        help="Require this many paired IDs after --shared_ids filtering (paper evidence audit: 362).",
    )
    args = parser.parse_args()
    left = _load(args.condition_a)
    right = _load(args.condition_b)
    left_rows = _prediction_map(left, args.condition_a)
    right_rows = _prediction_map(right, args.condition_b)
    left_ids, right_ids = set(left_rows), set(right_rows)
    if left_ids != right_ids:
        only_left = sorted(left_ids - right_ids)
        only_right = sorted(right_ids - left_ids)
        raise ValueError(
            "Condition prediction IDs must match exactly for paired aggregation; "
            f"only in condition_a={only_left[:10]} (n={len(only_left)}), "
            f"only in condition_b={only_right[:10]} (n={len(only_right)})"
        )
    benchmark_mismatches = [
        sample_id
        for sample_id in sorted(left_ids)
        if left_rows[sample_id]["benchmark"] != right_rows[sample_id]["benchmark"]
    ]
    if benchmark_mismatches:
        sample_id = benchmark_mismatches[0]
        raise ValueError(
            f"Benchmark assignment for paired ID {sample_id!r} differs between conditions: "
            f"{left_rows[sample_id]['benchmark']!r} vs {right_rows[sample_id]['benchmark']!r}"
        )
    ids = sorted(left_ids)
    if args.shared_ids:
        allowed = {line.strip() for line in Path(args.shared_ids).read_text(encoding="utf-8").splitlines() if line.strip()}
        ids = [sample_id for sample_id in ids if sample_id in allowed]
    if args.expected_shared_count and len(ids) != args.expected_shared_count:
        raise ValueError(
            f"Expected {args.expected_shared_count} paired IDs after shared-correct filtering, got {len(ids)}"
        )
    if args.difference == "a_minus_b":
        drops = [int(left_rows[sample_id]["correct"]) - int(right_rows[sample_id]["correct"]) for sample_id in ids]
    else:
        drops = [int(right_rows[sample_id]["correct"]) - int(left_rows[sample_id]["correct"]) for sample_id in ids]
    if not drops:
        raise ValueError("No paired prediction IDs remain after filtering")
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample_id, drop in zip(ids, drops):
        benchmark = left_rows[sample_id]["benchmark"] or right_rows[sample_id]["benchmark"] or "__all__"
        grouped[benchmark].append(drop)
    rng = random.Random(args.seed)
    means = []
    for _ in range(max(1, args.bootstrap_reps)):
        if len(grouped) > 1:
            # Preserve the benchmark-balanced estimand in every bootstrap
            # replicate instead of letting a larger benchmark dominate the CI.
            group_means = []
            for values in grouped.values():
                sample = [values[rng.randrange(len(values))] for _ in values]
                group_means.append(sum(sample) / len(sample))
            means.append(sum(group_means) / len(group_means))
        else:
            sample = [drops[rng.randrange(len(drops))] for _ in drops]
            means.append(sum(sample) / len(sample))
    means.sort()
    lower = means[max(0, int(0.025 * len(means)))]
    upper = means[min(len(means) - 1, int(0.975 * len(means)))]
    benchmark_drop = {name: sum(values) / len(values) for name, values in sorted(grouped.items())}
    balanced_drop = sum(benchmark_drop.values()) / len(benchmark_drop)
    result = {
        "n": len(drops),
        "mean_accuracy_drop": balanced_drop,
        "micro_accuracy_drop": sum(drops) / len(drops),
        "benchmark_accuracy_drop": benchmark_drop,
        "aggregation": "benchmark_balanced" if len(benchmark_drop) > 1 else "paired_mean",
        "bootstrap_95_ci": [lower, upper],
        "bootstrap_reps": max(1, args.bootstrap_reps),
        "seed": args.seed,
        "condition_a": str(Path(args.condition_a).resolve()),
        "condition_b": str(Path(args.condition_b).resolve()),
        "difference": args.difference,
        "estimand": "accuracy_A - accuracy_B" if args.difference == "a_minus_b" else "accuracy_B - accuracy_A",
        "shared_ids": str(Path(args.shared_ids).resolve()) if args.shared_ids else None,
        "expected_shared_count": args.expected_shared_count or None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
