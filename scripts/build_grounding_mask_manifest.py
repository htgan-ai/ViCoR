#!/usr/bin/env python3
"""Materialize GEMeX grounding-box masks and an auditable inference manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw


def _distance_bin(distance: float, max_radius: float, bin_edges: list[float]) -> int:
    """Return the interval containing normalized center distance in [0, 1]."""
    if max_radius <= 0:
        return 0
    normalized = min(1.0, max(0.0, distance / max_radius))
    for index, right_edge in enumerate(bin_edges[1:]):
        if normalized < right_edge or index == len(bin_edges) - 2:
            return index
    return len(bin_edges) - 2


def _parse_distance_bin_edges(raw: str, bin_count: int) -> tuple[list[float], str]:
    """Parse user-supplied normalized bin edges or make an explicit approximation.

    The paper specifies matching within a center-distance bin but does not
    publish the historical edges.  Equal-width edges remain the deterministic
    default; users can provide the exact edges used in their experiment so the
    manifest records an auditable protocol.
    """
    if raw.strip():
        try:
            edges = [float(value.strip()) for value in raw.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("--distance_bin_edges must be comma-separated finite numbers") from exc
        source = "user_supplied"
    else:
        if bin_count < 1:
            raise ValueError("--distance_bin_count must be positive")
        edges = [index / bin_count for index in range(bin_count + 1)]
        source = "equal_width_approximation"
    if len(edges) < 2 or any(not math.isfinite(value) for value in edges):
        raise ValueError("distance bin edges must contain at least two finite numbers")
    if abs(edges[0]) > 1e-9 or abs(edges[-1] - 1.0) > 1e-9:
        raise ValueError("distance bin edges must start at 0 and end at 1")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("distance bin edges must be strictly increasing")
    return edges, source


def _stable_source_id(row: dict) -> tuple[object, str]:
    """Return a source ID, deriving one from content when the dataset has none.

    GEMeX's released JSONL does not contain an explicit ID.  A line number is
    not stable under filtering or shuffling, so use the identifying fields to
    produce a deterministic content ID instead.
    """
    for key in ("id", "sample_id", "question_id", "uid", "uuid"):
        if row.get(key) is not None and str(row.get(key)).strip():
            return row[key], "source"
    answer = row.get("answer", row.get("ground_truth", row.get("gt", row.get("label", ""))))
    identity = {
        "image_path": row.get("image_path") or row.get("image", ""),
        "question": row.get("question", ""),
        "answer": answer,
        "grounding_boxes": row.get("grounding_boxes", []),
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return "gemex-" + hashlib.sha256(payload).hexdigest()[:24], "derived_content_hash"


def _parse_boxes(raw_boxes: object) -> list[list[int]]:
    """Normalize nested, bare, string, and bbox-dict exports to integer xyxy boxes."""
    if isinstance(raw_boxes, dict):
        raw_boxes = raw_boxes.get("boxes", raw_boxes.get("bbox", raw_boxes.get("box", [])))
    if isinstance(raw_boxes, str):
        try:
            raw_boxes = ast.literal_eval(raw_boxes)
        except (SyntaxError, ValueError):
            return []
    if isinstance(raw_boxes, (list, tuple)) and len(raw_boxes) == 4 and all(
        isinstance(value, (int, float)) for value in raw_boxes
    ):
        raw_boxes = [raw_boxes]
    if not isinstance(raw_boxes, (list, tuple)):
        return []
    parsed: list[list[int]] = []
    for raw in raw_boxes:
        try:
            value = raw if isinstance(raw, (list, tuple)) else ast.literal_eval(str(raw))
            if isinstance(value, (list, tuple)) and len(value) == 4:
                parsed.append([int(float(v)) for v in value])
        except (SyntaxError, ValueError, TypeError):
            continue
    return parsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="GEMeX JSONL with grounding_boxes")
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--distance_bin_count",
        type=int,
        default=4,
        help="Number of equal-width normalized center-distance bins for matched backgrounds.",
    )
    p.add_argument(
        "--distance_bin_edges",
        default="",
        help=(
            "Optional comma-separated normalized edges, including 0 and 1 "
            "(for example 0,0.2,0.5,1). Overrides --distance_bin_count."
        ),
    )
    args = p.parse_args()
    distance_bin_edges, distance_bin_source = _parse_distance_bin_edges(
        args.distance_bin_edges, args.distance_bin_count
    )
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root).expanduser().resolve()
    rows = []
    rng = random.Random(args.seed)
    for line_no, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        boxes = _parse_boxes(row.get("grounding_boxes", []))
        raw_image_path = str(row.get("image_path") or row.get("image", "")).strip()
        if not raw_image_path:
            continue
        image_candidate = Path(raw_image_path).expanduser()
        image_path = image_candidate if image_candidate.is_absolute() else image_root / image_candidate
        image_path = image_path.resolve()
        if not boxes or not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        # Appendix H defines boxes in the 336x336 GEMeX coordinate frame.
        sx, sy = width / 336.0, height / 336.0
        scaled = []
        for x1, y1, x2, y2 in boxes:
            if not (0 <= x1 < x2 <= 336 and 0 <= y1 < y2 <= 336):
                raise ValueError(f"Invalid xyxy box at source line {line_no}: {(x1, y1, x2, y2)}")
            scaled.append((round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)))
        if any(c <= a or d <= b for a, b, c, d in scaled):
            raise ValueError(f"Box collapsed after scaling at source line {line_no}")
        masked = image.copy()
        draw = ImageDraw.Draw(masked)
        for x1, y1, x2, y2 in scaled:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=(0, 0, 0))
        masked_name = f"{line_no:08d}_ground_truth_mask.png"
        masked.save(out / masked_name)

        # Matched background mask: choose a same-area rectangle outside the
        # annotated union when possible, with deterministic seeded fallback.
        x1 = min(v[0] for v in scaled); y1 = min(v[1] for v in scaled)
        x2 = max(v[2] for v in scaled); y2 = max(v[3] for v in scaled)
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        candidates = []
        for yy in range(0, max(1, height - box_h + 1), max(1, box_h // 4)):
            for xx in range(0, max(1, width - box_w + 1), max(1, box_w // 4)):
                if xx >= x2 or xx + box_w <= x1 or yy >= y2 or yy + box_h <= y1:
                    candidates.append((xx, yy, min(width, xx + box_w), min(height, yy + box_h)))
        if candidates:
            # The paper requires the background region to come from the same
            # center-distance bin whenever possible.  We make that rule
            # explicit and reproducible: normalized distance from image center
            # to the candidate center is partitioned using the recorded bins.
            gt_cx = (x1 + x2) / 2.0
            gt_cy = (y1 + y2) / 2.0
            image_cx, image_cy = width / 2.0, height / 2.0
            max_radius = math.hypot(width, height) / 2.0
            gt_dist = math.hypot(gt_cx - image_cx, gt_cy - image_cy)
            gt_bin = _distance_bin(gt_dist, max_radius, distance_bin_edges)
            scored = []
            for candidate in candidates:
                cx = (candidate[0] + candidate[2]) / 2.0
                cy = (candidate[1] + candidate[3]) / 2.0
                dist = math.hypot(cx - image_cx, cy - image_cy)
                candidate_bin = _distance_bin(dist, max_radius, distance_bin_edges)
                scored.append((candidate_bin, abs(dist - gt_dist), candidate, dist))
            same_bin = [item for item in scored if item[0] == gt_bin]
            if same_bin:
                matched_bin = gt_bin
                pool = same_bin
                fallback = False
            else:
                # The paper allows a fallback when the same bin has no valid
                # non-overlapping placement. Choose the nearest available bin,
                # then the nearest center within that bin.
                nearest_bin_gap = min(abs(item[0] - gt_bin) for item in scored)
                matched_bin = min(item[0] for item in scored if abs(item[0] - gt_bin) == nearest_bin_gap)
                pool = [item for item in scored if item[0] == matched_bin]
                fallback = True
            best_gap = min(item[1] for item in pool)
            nearest = [item[2] for item in pool if abs(item[1] - best_gap) < 1e-9]
            bg_box = rng.choice(nearest)
            bg_dist = next(item[3] for item in pool if item[2] == bg_box and abs(item[1] - best_gap) < 1e-9)
        else:
            raise ValueError(f"No non-overlapping translated evidence mask for source line {line_no}; supply a validated background mask")
        background = image.copy()
        background_draw = ImageDraw.Draw(background)
        dx, dy = bg_box[0] - x1, bg_box[1] - y1
        background_boxes = [(a + dx, b + dy, c + dx, d + dy) for a, b, c, d in scaled]
        # Translate the entire evidence union, preserving its exact pixel area
        # even when the annotation contains multiple overlapping boxes.
        for a, b, c, d in background_boxes:
            background_draw.rectangle((a, b, c - 1, d - 1), fill=(0, 0, 0))
        background_name = f"{line_no:08d}_background_mask.png"
        background.save(out / background_name)
        source_id, source_id_kind = _stable_source_id(row)
        answer = row.get("answer", row.get("ground_truth", row.get("gt", row.get("label", ""))))
        label_space = row.get("label_space", row.get("labels", row.get("classes", [])))
        options = row.get("options", row.get("choices"))
        manifest_row = {
            # Preserve task identity and answer metadata from the source row.
            # The evaluator/shared-ID builder must never fall back to a
            # position-dependent manifest index.
            "id": source_id,
            "source_id_kind": source_id_kind,
            "source_line": line_no,
            "image": str(image_path),
            "masked_image": str(out / masked_name),
            "background_masked_image": str(out / background_name),
            "benchmark": row.get("benchmark", row.get("source_benchmark", "gemex_grounded")),
            "question": row.get("question", ""),
            "answer": answer,
            "ground_truth": row.get("ground_truth", answer),
            "gt": row.get("gt", answer),
            "options": options,
            "label_space": label_space,
            "labels": row.get("labels", label_space),
            "grounding_refs": row.get("grounding_refs", []),
            "grounding_boxes": boxes,
            "scaled_grounding_boxes": [list(v) for v in scaled],
            "background_box": list(bg_box),
            "background_boxes": [list(v) for v in background_boxes],
            "matching_rule": "nonoverlapping_union_translation_same_normalized_center_distance_bin",
            "distance_bin_count": len(distance_bin_edges) - 1,
            "distance_bin_edges": [round(value, 6) for value in distance_bin_edges],
            "distance_bin_edges_source": distance_bin_source,
            "ground_truth_center_distance": gt_dist,
            "background_center_distance": bg_dist,
            "ground_truth_distance_bin": gt_bin,
            "background_distance_bin": matched_bin,
            "distance_bin_fallback": fallback,
            "seed": args.seed,
        }
        # Preserve flat option fields (option_0, option_A, choice_B, ...) used
        # by some benchmark exports in addition to a nested ``options`` map.
        for key, value in row.items():
            if re.fullmatch(r"(?:option|choice|class)[_-]?[A-Ea-e0-4]", str(key)):
                manifest_row[str(key)] = value
        rows.append(manifest_row)
        if args.max_samples and len(rows) >= args.max_samples:
            break
    manifest = out / "grounding_mask_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "manifest": str(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
