#!/usr/bin/env python3
"""Create deterministic same-group wrong-image pairings for faithfulness audits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _resolve_image_path(raw_path: object, *, input_path: Path, image_root: Path | None) -> Path:
    """Resolve source images once so the generated manifest is cwd-independent."""
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError("image path is empty")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        candidates = []
        if image_root is not None:
            candidates.append(image_root / candidate)
        candidates.append(input_path.parent / candidate)
        # Prefer a path that exists; if neither does, retain the documented
        # image-root interpretation so the eventual error names the resolved
        # path rather than depending on the launch directory.
        resolved = next((item.resolve() for item in candidates if item.exists()), candidates[0].resolve())
    if not resolved.exists():
        raise FileNotFoundError(f"image path does not exist: {resolved}")
    return resolved


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSONL with image and question fields")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--image_root",
        default="",
        help="Optional root for relative image paths; input JSONL directory is the fallback.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--max_per_group",
        type=int,
        default=500,
        help="Number of examples sampled from each benchmark group; 0 keeps every row.",
    )
    p.add_argument(
        "--group_field",
        default="benchmark",
        help="Field defining the same-benchmark pairing group; pass the actual field used by the manifest.",
    )
    p.add_argument(
        "--allow_global_group",
        action="store_true",
        help="Allow rows without group_field to share one global pool (not paper-faithful).",
    )
    args = p.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, row in enumerate(rows):
        source_image = row.get("image", row.get("image_path", ""))
        if not str(source_image).strip():
            raise ValueError(f"row {index} has no image or image_path")
        row["image"] = str(_resolve_image_path(source_image, input_path=input_path, image_root=image_root))
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        raw_group = row.get(args.group_field)
        if raw_group is None or not str(raw_group).strip():
            if not args.allow_global_group:
                raise ValueError(
                    f"row {i} has no {args.group_field!r}; provide the benchmark field explicitly "
                    "or pass --allow_global_group only for a non-paper smoke test"
                )
            raw_group = "__all__"
        groups.setdefault(str(raw_group), []).append(i)
    rng = random.Random(args.seed)
    output = []
    for group, indices in sorted(groups.items()):
        if args.max_per_group > 0 and len(indices) > args.max_per_group:
            indices = sorted(rng.sample(indices, args.max_per_group))
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for pos, idx in enumerate(indices):
            if len(shuffled) < 2:
                raise ValueError(f"group {group!r} needs at least two images")
            own_image = str(rows[idx].get("image", ""))
            wrong_idx = None
            for offset in range(len(shuffled)):
                candidate = shuffled[(pos + offset) % len(shuffled)]
                if candidate != idx and str(rows[candidate].get("image", "")) != own_image:
                    wrong_idx = candidate
                    break
            if wrong_idx is None:
                raise ValueError(f"group {group!r} has no second distinct image for row {rows[idx].get('id', idx)!r}")
            row = dict(rows[idx])
            row["wrong_image"] = rows[wrong_idx]["image"]
            row["wrong_image_id"] = rows[wrong_idx].get("id", str(wrong_idx))
            row["pair_seed"] = args.seed
            row["pair_group"] = group
            output.append(row)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(output), "groups": len(groups), "seed": args.seed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
