#!/usr/bin/env python3
"""Build the two 40k CXR-CounterFact sources from a prepared JSONL.

For a complete raw-data preparation run use
``scripts/00_preprocess_training_pools.py --dataset cxr``.  This command is a
small post-classification utility for users who already have the Qwen3-4B
task labels and want to rebuild the deterministic random-normal sampling
step only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


_FILL_RE = re.compile(
    r"(?:\bfill[\s_-]*in[\s_-]*(?:the[\s_-]*)?blank(?:s)?\b|"
    r"\bfill[\s_-]*the[\s_-]*blank(?:s)?\b|complete the sentence|_{2,}|"
    r"\[\s*blank\s*\])",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Denoised/classified CXR JSONL")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--total_size", type=int, default=40000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--type_field", default="task_type")
    return p.parse_args()


def norm_type(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field, row.get("task", row.get("category", "")))).strip().lower()
    if value in {"single", "single-disease", "single_disease"}:
        return "single"
    if value in {"multi", "multi-disease", "multiple-disease", "multiple_diseases"}:
        return "multi"
    if value in {"normal", "negative", "no-finding", "no_finding"}:
        return "normal"
    return "unknown"


def is_fill_blank(row: dict[str, Any]) -> bool:
    explicit = row.get("is_fill_blank", False)
    if isinstance(explicit, str):
        explicit = explicit.strip().casefold() in {"1", "true", "yes", "y", "on"}
    elif isinstance(explicit, (int, float)):
        explicit = bool(explicit)
    else:
        explicit = bool(explicit)
    if explicit:
        return True
    text = " ".join(str(row.get(k, "")) for k in ("question", "answer", "reasoning", "task_type", "task", "category"))
    return bool(_FILL_RE.search(text))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(disease_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]], total: int, rng: random.Random) -> list[dict[str, Any]]:
    if total <= 0:
        raise ValueError("total_size must be positive")
    disease = list(disease_rows)
    if not disease:
        raise ValueError("no retained disease rows remain")
    if len(disease) >= total:
        if not normal_rows:
            raise ValueError("disease rows fill the target but no normal rows are available")
        normal_fraction = len(normal_rows) / float(len(normal_rows) + len(disease))
        normal_needed = max(1, min(len(normal_rows), int(round(total * normal_fraction))))
        disease_needed = total - normal_needed
        chosen = rng.sample(disease, disease_needed) + rng.sample(normal_rows, normal_needed)
    else:
        normal_needed = total - len(disease)
        if len(normal_rows) < normal_needed:
            raise ValueError(f"need {normal_needed} normal rows, found {len(normal_rows)}")
        chosen = disease + rng.sample(normal_rows, normal_needed)
    rng.shuffle(chosen)
    return chosen


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in rows if not is_fill_blank(row)]
    typed = {name: [row for row in kept if norm_type(row, args.type_field) == name] for name in ("single", "multi", "normal")}
    unknown = sum(norm_type(row, args.type_field) == "unknown" for row in kept)
    if unknown:
        raise ValueError(f"{unknown} rows have an unrecognized task type; complete classification first")
    rng = random.Random(args.seed)
    single = build(typed["single"], typed["normal"], args.total_size, rng)
    multi = build(typed["multi"], typed["normal"], args.total_size, rng)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "single_disease.jsonl", single)
    write_jsonl(out / "multiple_diseases.jsonl", multi)
    manifest = {
        "preprocessor": "scripts/00_preprocess_training_pools.py",
        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(Path(args.input).resolve()),
        "seed": args.seed,
        "total_size": args.total_size,
        "input_rows": len(rows),
        "after_fill_blank_filter": len(kept),
        "classified_counts": {k: len(v) for k, v in typed.items()},
        "single_output": len(single),
        "multi_output": len(multi),
        "normal_sampling": "random_without_replacement",
        "output_sha256": {
            "single_disease.jsonl": sha256_file(out / "single_disease.jsonl"),
            "multiple_diseases.jsonl": sha256_file(out / "multiple_diseases.jsonl"),
        },
    }
    (out / "cxr_source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
