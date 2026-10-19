#!/usr/bin/env python3
"""Reproduce the pre-finetuning visual-dependence audit on a Qwen3-VL model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .qwen3vl_common import load_peft_vlm
except ImportError:  # direct ``python evaluation/...`` invocation
    from qwen3vl_common import load_peft_vlm
from vicor.data.collators import build_collator


def _resolve_image_path(raw_path: object, manifest_path: Path) -> Path | None:
    """Resolve a manifest image path independently of the launch directory."""
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    relative_to_manifest = (manifest_path.parent / candidate).resolve()
    if relative_to_manifest.exists():
        return relative_to_manifest
    # Keep a useful error path for manifests whose paths were relative
    # to the directory from which the manifest was created.
    return candidate.resolve()


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _loss(model: torch.nn.Module, processor: Any, image: Image.Image, question: str, answer: str, device: str) -> float:
    collator = build_collator(
        "qwen3_vl",
        "custom_jsonl",
        processor=processor,
        data_cfg={"supervision_mode": "final_answer"},
    )
    batch = collator([{"id": "audit", "image": image, "question": question, "answer": answer}])
    model_inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
        and key in {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
    }
    if "pixel_values" in model_inputs:
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        output = model(**model_inputs, use_cache=False)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("model returned a non-finite final-answer loss")
    return float(output.loss.detach().cpu())


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positives for neg in negatives)
    return wins / (len(positives) * len(negatives))


def _rank_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, float | None]:
    values = []
    for index, row in enumerate(rows):
        raw_score = row.get(score_key)
        if raw_score is None or not isinstance(raw_score, (int, float)) or not math.isfinite(float(raw_score)):
            raise ValueError(
                f"visual-dependence manifest is missing finite {score_key!r} at row {index}; "
                "run the ViCoR scorer first or choose a score column present in every row"
            )
        raw_target = row.get("visual_dependence_target")
        if raw_target is None or not isinstance(raw_target, (int, float)) or not math.isfinite(float(raw_target)):
            raise ValueError(f"row {index} has no finite visual_dependence_target")
        values.append((float(raw_score), float(raw_target)))
    if not values:
        raise ValueError("visual-dependence manifest has no rows")
    # Standardize the positive/negative score gap using the *full scored
    # pool*.  Restricting the denominator to the pseudo-positive/negative
    # subset changes the reported effect size whenever the middle target band
    # has a different score variance.
    all_scores = [score for score, _ in values]
    score_mean = sum(all_scores) / len(all_scores)
    score_std = math.sqrt(sum((score - score_mean) ** 2 for score in all_scores) / len(all_scores))
    values.sort(key=lambda item: item[1])
    n = len(values)
    low = max(1, math.ceil(n * 0.30))
    high = max(low + 1, math.floor(n * 0.70))
    target_order = [target for _, target in values]
    threshold_low = target_order[low - 1]
    threshold_high = target_order[min(high - 1, n - 1)]
    labels = [1 if target >= threshold_high else 0 if target <= threshold_low else -1 for _, target in values]
    valid = [(score, label) for (score, _), label in zip(values, labels) if label >= 0]
    top20 = max(1, math.ceil(n * 0.20))
    score_ranked = sorted(values, key=lambda item: item[0], reverse=True)
    return {
        "top20_mean": sum(target for _, target in score_ranked[:top20]) / min(top20, n),
        "auc": _auc([label for _, label in valid], [score for score, _ in valid]),
        # Precision@20% is defined over the full scored pool.  The middle
        # target band remains in the ranked pool; removing it first changes
        # which examples belong to the top 20%.
        "precision_at20": sum(
            1 for _, target in score_ranked[:top20] if target >= threshold_high
        ) / top20,
        "delta_z": (
            (sum(score for score, label in valid if label == 1) / max(1, sum(label == 1 for _, label in valid))
             - sum(score for score, label in valid if label == 0) / max(1, sum(label == 0 for _, label in valid)))
            / (score_std or 1.0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL containing id, question, answer, image, and wrong_image")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--image_max_pixels", type=int, default=262144)
    parser.add_argument("--image_min_pixels", type=int, default=50176)
    parser.add_argument(
        "--score_columns",
        nargs="+",
        default=["score_delta_log1p_rho_nov", "score_delta", "score_delta_nov"],
    )
    args = parser.parse_args()
    processor, model, backbone = load_peft_vlm(
        adapter_path="",
        base_model_path=args.base_model_path,
        device=args.device,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
    )
    if backbone != "qwen3_vl":
        raise ValueError("The pre-finetuning visual-dependence audit is defined for Qwen3-VL")

    manifest_path = Path(args.manifest).expanduser().resolve()
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_path = _resolve_image_path(row.get("image", ""), manifest_path)
        wrong_path = _resolve_image_path(row.get("wrong_image", ""), manifest_path)
        if image_path is None or wrong_path is None:
            raise ValueError(f"row {index} requires image and wrong_image")
        original = _load_image(image_path)
        wrong = _load_image(wrong_path)
        blank = Image.new("RGB", original.size, (0, 0, 0))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        loss_original = _loss(model, processor, original, question, answer, args.device)
        loss_no_image = _loss(model, processor, blank, question, answer, args.device)
        loss_wrong_image = _loss(model, processor, wrong, question, answer, args.device)
        output = dict(row)
        output.update(
            {
                "loss_original": loss_original,
                "loss_no_image": loss_no_image,
                "loss_wrong_image": loss_wrong_image,
                "visual_dependence_target": max(loss_no_image - loss_original, loss_wrong_image - loss_original),
            }
        )
        outputs.append(output)

    summary = {
        "model": "qwen3_vl",
        "samples": len(outputs),
        "metrics": {key: _rank_metrics(outputs, key) for key in args.score_columns},
        "output_jsonl": str(Path(args.output).with_suffix(".jsonl")),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
