#!/usr/bin/env python3
"""Build DrVD Multi-turn QA conversations from ``joint_qa.jsonl``.

The official DrVD-Bench release contains ``visual_evidence_qa.jsonl``,
``independent_qa.jsonl``, ``joint_qa.jsonl`` and ``report_generation.jsonl``.
Multi-turn QA asks the hierarchical levels of each joint record one turn at a
time with the previous turns in context.  This script writes those
conversations as ``multi_turn_qa.jsonl`` in the ``turns`` schema accepted by
``evaluation/evaluate_drvd_qwen3vl.py`` (which can also derive them on the fly
with ``--drvd_multiturn_source joint``).

Usage:
    python scripts/build_drvd_multiturn_from_joint.py --joint /path/DrVD-Bench/joint_qa.jsonl --output /path/DrVD-Bench/multi_turn_qa.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LEVEL_QUESTIONS = [
    ("modality", "What imaging modality is used in this image?"),
    ("body_region", "Which body region is shown in this image?"),
    ("organ", "Which organ appears to be abnormal in this image?"),
    ("lesion", "Based on the abnormal organ, what lesion or finding is most clearly visible?"),
    ("diagnosis", "Considering all the above findings, what is the most likely diagnosis?"),
]


def convert(row: dict) -> dict:
    turns = []
    for name, question in LEVEL_QUESTIONS:
        options = row.get(f"{name}_options")
        answer = row.get(f"{name}_answer")
        if options is None and answer is None:
            continue  # older four-level mirrors have no body_region fields
        if options is None or answer in (None, ""):
            raise ValueError(f"record {row.get('unique_id', row.get('id'))!r}: level {name} lacks options or answer")
        turns.append({"level": name, "question": question, "options": options, "answer": answer})
    if len(turns) < 4:
        raise ValueError(f"record {row.get('unique_id', row.get('id'))!r} has only {len(turns)} levels")
    level_fields = {f"{name}_{suffix}" for name, _ in LEVEL_QUESTIONS for suffix in ("options", "answer", "question", "prompt")}
    out = {key: value for key, value in row.items() if key not in level_fields and key != "joint_prompt"}
    out["task"] = "multi_turn"
    out["turns"] = turns
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint", required=True, help="path to joint_qa.jsonl")
    parser.add_argument("--output", required=True, help="path of the multi_turn_qa.jsonl to write")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.joint).read_text(encoding="utf-8").splitlines() if line.strip()]
    converted = [convert(row) for row in rows]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in converted:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(converted)} multi-turn conversations to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
