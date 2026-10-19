#!/usr/bin/env python3
"""Create GEMeX's 60k pool and 1k geometry reserve from prepared JSONL.

For the complete denoise/deduplicate preparation, use
``scripts/00_preprocess_training_pools.py --dataset gemex``.  This utility
is a deterministic post-processing entry point for an already prepared file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from pathlib import Path


def boxes(row: dict) -> list[list[float]]:
    out = []
    raw_boxes = row.get("grounding_boxes", []) or []
    # Some GEMeX JSONL writers store the complete box list as a JSON/Python
    # string (for example ``"[[1,2,30,40]]"``).  Decode the container before
    # handling bare boxes and nested per-box strings.
    if isinstance(raw_boxes, str):
        try:
            raw_boxes = json.loads(raw_boxes)
        except json.JSONDecodeError:
            try:
                raw_boxes = ast.literal_eval(raw_boxes)
            except (SyntaxError, ValueError):
                return out
    if isinstance(raw_boxes, dict):
        raw_boxes = raw_boxes.get("boxes", raw_boxes.get("bbox", raw_boxes.get("box", [])))
        if isinstance(raw_boxes, str):
            try:
                raw_boxes = json.loads(raw_boxes)
            except json.JSONDecodeError:
                try:
                    raw_boxes = ast.literal_eval(raw_boxes)
                except (SyntaxError, ValueError):
                    return out
    if isinstance(raw_boxes, (list, tuple)) and len(raw_boxes) == 4 and all(
        isinstance(value, (int, float)) for value in raw_boxes
    ):
        raw_boxes = [raw_boxes]
    if not isinstance(raw_boxes, (list, tuple)):
        return out
    for raw in raw_boxes:
        try:
            value = raw if isinstance(raw, (list, tuple)) else ast.literal_eval(str(raw))
            if isinstance(value, (list, tuple)) and len(value) == 4:
                vals = [float(v) for v in value]
                if vals[2] > vals[0] and vals[3] > vals[1] and all(0 <= v <= 336 for v in vals):
                    out.append(vals)
        except (SyntaxError, ValueError, TypeError):
            continue
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Denoised GEMeX JSONL")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pool_size", type=int, default=60000)
    p.add_argument("--reserve_size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [row for row in rows if boxes(row)]
    if len(valid) < args.reserve_size:
        raise ValueError(f"need {args.reserve_size} geometry-valid rows, found {len(valid)}")
    rng = random.Random(args.seed)
    reserve = rng.sample(valid, args.reserve_size)
    reserve_ids = {row.get("preprocess_id", id(row)) for row in reserve}
    remaining = [row for row in rows if row.get("preprocess_id", id(row)) not in reserve_ids]
    if len(remaining) < args.pool_size:
        raise ValueError(f"need {args.pool_size} remaining rows, found {len(remaining)}")
    pool = rng.sample(remaining, args.pool_size)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "gemex_train_60k.jsonl", pool)
    write_jsonl(out / "gemex_grounding_reserve_1k.jsonl", reserve)
    pool_ids = {row.get("preprocess_id", id(row)) for row in pool}
    reserve_ids = {row.get("preprocess_id", id(row)) for row in reserve}
    if pool_ids & reserve_ids:
        raise RuntimeError("GEMeX training pool and grounding reserve overlap")
    manifest = {
        "preprocessor": "scripts/00_preprocess_training_pools.py",
        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(Path(args.input).resolve()),
        "seed": args.seed,
        "input_rows": len(rows),
        "geometry_valid_rows": len(valid),
        "training_pool_size": len(pool),
        "reserve_size": len(reserve),
        "reserve_role": "evidence-region perturbation audit",
        "disjoint": True,
        "output_sha256": {
            "gemex_train_60k.jsonl": sha256_file(out / "gemex_train_60k.jsonl"),
            "gemex_grounding_reserve_1k.jsonl": sha256_file(out / "gemex_grounding_reserve_1k.jsonl"),
        },
    }
    (out / "gemex_split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
