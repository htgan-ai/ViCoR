#!/usr/bin/env python3
"""Build the paper's shared-correct ID list for the evidence-region audit.

Each input is a normal-image prediction JSON (the output produced by
``evaluate_visual_perturbations_qwen3vl.py --mode normal``) for one compared
model.  The emitted IDs are correct for every input model, so all region-level
drops are evaluated on one fixed set of examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_normal_predictions(path: Path) -> tuple[dict[str, str], set[str]]:
    """Load and validate one complete normal-condition result.

    This utility is used to define the fixed shared-correct population.  A
    duplicate ID, missing benchmark, or silently truncated result would make
    the intersection depend on file order, so fail before writing any IDs.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") not in (None, "normal"):
        raise ValueError(f"{path} is not a normal-condition result (mode={payload.get('mode')!r})")
    rows = payload.get("predictions", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no prediction rows")
    declared_n = payload.get("n")
    if declared_n is not None and int(declared_n) != len(rows):
        raise ValueError(f"{path} declares n={declared_n}, but contains {len(rows)} rows")
    benchmark_by_id: dict[str, str] = {}
    correct: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} prediction {row_index} is not an object")
        raw_id = row.get("id")
        sample_id = str(raw_id).strip() if raw_id is not None else ""
        if not sample_id:
            raise ValueError(f"{path} prediction {row_index} has an empty id")
        if sample_id in benchmark_by_id:
            raise ValueError(f"{path} contains duplicate prediction id {sample_id!r}")
        benchmark = str(row.get("benchmark", "")).strip()
        if not benchmark:
            raise ValueError(f"{path} prediction {sample_id!r} has no benchmark")
        if not isinstance(row.get("correct"), bool):
            raise ValueError(f"{path} prediction {sample_id!r} has non-boolean correct={row.get('correct')!r}")
        benchmark_by_id[sample_id] = benchmark
        if row["correct"]:
            correct.add(sample_id)
    return benchmark_by_id, correct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normal_results",
        nargs="+",
        required=True,
        help="Normal-condition JSON outputs for all compared models.",
    )
    parser.add_argument("--output", required=True, help="Newline-delimited shared-correct IDs")
    parser.add_argument(
        "--expected_count",
        type=int,
        default=362,
        help="The paper reports 362 shared-correct examples; set 0 to skip the count check.",
    )
    parser.add_argument(
        "--allow_different_count",
        action="store_true",
        help="Allow a different intersection size for a non-paper smoke test.",
    )
    parser.add_argument(
        "--min_models",
        type=int,
        default=2,
        help="Require at least this many independently evaluated normal results (default: 2).",
    )
    parser.add_argument(
        "--expected_models",
        type=int,
        default=0,
        help="Optionally require an exact number of compared model/seed result files.",
    )
    parser.add_argument(
        "--allow_single_model",
        action="store_true",
        help="Explicitly allow one result for a smoke test; this is not the paper shared-correct protocol.",
    )
    parser.add_argument(
        "--allow_partial_ids",
        action="store_true",
        help="Allow result files with different ID sets (smoke tests only).",
    )
    args = parser.parse_args()

    paths = [Path(path) for path in args.normal_results]
    if args.min_models < 1:
        raise ValueError("--min_models must be positive")
    if args.expected_models < 0:
        raise ValueError("--expected_models cannot be negative")
    if len(paths) < args.min_models and not (args.allow_single_model and len(paths) == 1):
        raise ValueError(
            f"Expected at least {args.min_models} independently evaluated normal results, got {len(paths)}. "
            "Pass every compared model/seed result; use --allow_single_model only for a smoke test."
        )
    if args.expected_models and len(paths) != args.expected_models:
        raise ValueError(f"Expected {args.expected_models} normal result files, got {len(paths)}")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("--normal_results contains the same result file more than once")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing normal-condition result(s): " + ", ".join(missing))
    loaded = [_load_normal_predictions(path) for path in paths]
    reference_ids = set(loaded[0][0])
    reference_benchmarks = loaded[0][0]
    for path, (benchmarks, _) in zip(paths[1:], loaded[1:]):
        ids = set(benchmarks)
        if not args.allow_partial_ids:
            if ids != reference_ids:
                missing_ids = sorted(reference_ids - ids)[:5]
                extra_ids = sorted(ids - reference_ids)[:5]
                raise ValueError(
                    f"{path} does not contain the same prediction IDs as {paths[0]}; "
                    f"missing={missing_ids}, extra={extra_ids}. Use --allow_partial_ids only for a smoke test."
                )
        # Even partial-ID smoke tests must agree on benchmark assignment for
        # IDs that are present in both files; otherwise the balanced metric is
        # not well-defined.
        common_ids = reference_ids & ids
        mismatched = sorted(
            sample_id for sample_id in common_ids if benchmarks[sample_id] != reference_benchmarks[sample_id]
        )
        if mismatched:
            sample_id = mismatched[0]
            raise ValueError(
                f"Benchmark assignment for ID {sample_id!r} differs between {paths[0]} "
                f"and {path}: {reference_benchmarks[sample_id]!r} vs {benchmarks[sample_id]!r}"
            )
    intersections = [correct for _, correct in loaded]
    shared = set.intersection(*intersections) if intersections else set()
    if args.expected_count and len(shared) != args.expected_count and not args.allow_different_count:
        raise ValueError(
            f"Expected {args.expected_count} shared-correct IDs for the paper audit, "
            f"but found {len(shared)}. Use --allow_different_count only for a smoke test."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{sample_id}\n" for sample_id in sorted(shared)), encoding="utf-8")
    print(json.dumps({"models": len(paths), "shared_correct": len(shared), "output": str(output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
