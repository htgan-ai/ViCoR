#!/usr/bin/env python3
"""Deterministic Stage-I response filter used before ViCoR scoring.

The filter removes empty/template answers and exact or very-near duplicate
question-answer pairs.  It writes a candidate JSONL, a row-level audit
manifest, and a runnable config whose dataset points at the candidate file.
No model or evaluation labels are read.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vicor.config import load_config


_SPACE_RE = re.compile(r"\s+")
_BAD_ANSWERS = {"", "n/a", "na", "none", "null", "unknown", "cannot answer", "no answer"}
_TEMPLATE_RE = re.compile(r"^(?:as an ai|i cannot|i don't have access|not enough information)[.! ]*$", re.I)


def normalize(text: Any) -> str:
    return _SPACE_RE.sub(" ", str(text or "").strip()).casefold()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output_config", required=True)
    p.add_argument("--candidate_ratio", type=float, default=0.0, help="Optional gamma budget cap; 0 keeps all survivors")
    p.add_argument("--near_threshold", type=float, default=0.98)
    p.add_argument("--min_answer_chars", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gamma = float(args.candidate_ratio)
    if gamma < 0.0 or (0.0 < gamma < 1.0):
        raise ValueError(
            "candidate_ratio (gamma) must be 0 (disabled) or >= 1.0; "
            f"got {gamma}"
        )
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    root = Path(str(cfg["data_root"]))
    source = root / str(data_cfg.get("jsonl_name", "data.jsonl"))
    if not source.exists():
        raise FileNotFoundError(source)

    # Candidate paths are stored inside a dataset config and are later joined
    # with ``data_root`` by GEMeXDataset.  Resolve them here so an output such
    # as ``outputs/stage1_candidates.jsonl`` is not interpreted as
    # ``<data_root>/outputs/stage1_candidates.jsonl``.
    out_cfg = Path(args.output_config).expanduser().resolve()
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = out_cfg.with_name("stage1_candidates.jsonl").resolve()
    manifest_path = out_cfg.with_name("stage1_manifest.jsonl").resolve()

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    exact_seen: dict[str, int] = {}
    buckets: dict[tuple[str, str, int], list[tuple[int, str]]] = {}
    with source.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            question = normalize(item.get("question"))
            answer = normalize(item.get("answer"))
            reasoning = normalize(item.get("reasoning"))
            # Deduplicate the same image-question-answer example.  Using only
            # question+answer would incorrectly collapse identical prompts
            # attached to different radiographs.
            image_identity = normalize(
                item.get("image_path") or item.get("image") or item.get("img") or item.get("id")
            )
            if not image_identity:
                image_identity = f"source_line:{source_line}"
            text_key = f"{image_identity}\n{question}\n{answer}"
            reason = "kept"
            if not question or answer in _BAD_ANSWERS or len(answer) < int(args.min_answer_chars):
                reason = "empty_or_too_short"
            elif _TEMPLATE_RE.match(answer):
                reason = "template_response"
            elif text_key in exact_seen:
                reason = f"exact_duplicate:{exact_seen[text_key]}"
            else:
                bucket = (image_identity, answer, int(len(question.split()) / 8))
                for previous_line, previous_text in buckets.get(bucket, []):
                    ratio = difflib.SequenceMatcher(None, text_key, previous_text).ratio()
                    if ratio >= float(args.near_threshold):
                        reason = f"near_duplicate:{previous_line}"
                        break
                if reason == "kept":
                    exact_seen[text_key] = source_line
                    buckets.setdefault(bucket, []).append((source_line, text_key))
                    rows.append(item)
            audit.append({"source_line": source_line, "status": reason})

    # ``candidate_ratio`` is the paper's gamma cap.  It is deliberately
    # deferred until after ViCoR scores have been computed: Stage-I must pass
    # every quality-filtered row to the scorer, then the scorer keeps the
    # highest-scoring gamma*K rows.  Hash/random truncation here would change
    # the candidate pool before scores exist and violate Appendix protocol.
    candidate_index = 0
    for item in audit:
        if item.get("status") == "kept":
            item["candidate_index"] = candidate_index
            candidate_index += 1

    with candidate_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in audit:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg.setdefault("data", {})["jsonl_name"] = str(candidate_path)
    cfg["stage1"] = {
        "source_jsonl": str(source),
        "candidate_jsonl": str(candidate_path),
        "audit_manifest": str(manifest_path),
        "near_threshold": float(args.near_threshold),
        "candidate_ratio": float(args.candidate_ratio),
        "candidate_ratio_applied_after_scoring": bool(args.candidate_ratio and args.candidate_ratio > 0),
    }
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"config": str(out_cfg), "candidates": len(rows), "audit": len(audit)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
