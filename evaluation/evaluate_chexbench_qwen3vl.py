#!/usr/bin/env python3
"""Evaluate CheXbench Multiple Disease Identification with a Qwen3-VL adapter."""

from __future__ import annotations

import sys

try:
    from .evaluate_single_disease_qwen3vl import main
except ImportError:  # direct ``python evaluation/...`` invocation
    from evaluate_single_disease_qwen3vl import main


def _inject_defaults() -> None:
    if "--benchmarks" not in sys.argv:
        sys.argv.extend(["--benchmarks", "multiple_disease_identification"])
    if "--no-medblink_use_task_label_space" not in sys.argv and "--medblink_use_task_label_space" not in sys.argv:
        sys.argv.append("--no-medblink_use_task_label_space")


if __name__ == "__main__":
    _inject_defaults()
    raise SystemExit(main())
