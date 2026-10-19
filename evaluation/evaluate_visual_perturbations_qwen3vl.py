#!/usr/bin/env python3
"""Evaluate the paper's image-level and evidence-region perturbations."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

try:
    from .qwen3vl_common import load_peft_vlm, run_chat_generation_generic
except ImportError:  # direct ``python evaluation/...`` invocation
    from qwen3vl_common import load_peft_vlm, run_chat_generation_generic


def norm(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", str(text or ""), flags=re.I | re.S)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _normalize_label(text: str) -> str:
    """Normalize a benchmark label without treating a whole rationale as a label."""
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def _extract_choice(raw_text: str, valid: Iterable[str]) -> str:
    valid_set = {str(v).upper() for v in valid}
    text = re.sub(r"<[^>]+>", " ", str(raw_text or "")).strip()
    for pattern in (
        r"(?i)(?:final\s+answer|answer|答案)\s*[:：]\s*([A-Z])\b",
        r"(?i)\\boxed\{\s*([A-Z])\s*\}",
        r"(?i)option\s*([A-Z])\b",
        r"(?i)\(([A-Z])\)",
    ):
        match = re.search(pattern, text)
        if match and match.group(1).upper() in valid_set:
            return match.group(1).upper()
    letters = [token.upper() for token in re.findall(r"\b([A-Z])\b", text) if token.upper() in valid_set]
    return letters[-1] if letters else ""


def _prompt_text(row: dict[str, Any], question: str) -> str:
    """Question plus the answer space, matching the benchmark evaluators.

    Multiple-choice rows list their options and ask for the option letter;
    label-space rows list the admissible labels.  Scoring in
    ``_score_prediction`` uses the same option map / label space, so the
    model sees exactly the candidates it is scored against.
    """
    options = _option_map(row)
    if options:
        option_text = "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
        return (
            f"Question: {question}\nOptions:\n{option_text}\n"
            "Output exactly one line: Answer: <LETTER>. No explanation."
        )
    label_space = row.get("label_space", row.get("labels", []))
    if isinstance(label_space, (list, tuple)) and label_space:
        labels = ", ".join(str(label) for label in label_space)
        return f"Question: {question}\nAnswer with exactly one of the following labels: {labels}."
    return question


def _option_map(row: dict[str, Any]) -> dict[str, str]:
    """Return a canonical option map from common benchmark schemas."""
    raw = row.get("options", row.get("choices"))
    if isinstance(raw, dict):
        options: dict[str, str] = {}
        for key, value in raw.items():
            key_text = str(key).strip()
            letter = re.fullmatch(r"(?:option|choice|class)?[_-]?([A-Ea-e])", key_text, flags=re.I)
            numeric = re.fullmatch(r"(?:option|choice|class)?[_-]?([0-4])", key_text, flags=re.I)
            if letter:
                canonical = letter.group(1).upper()
            elif numeric:
                canonical = chr(65 + int(numeric.group(1)))
            else:
                continue
            options[canonical] = str(value)
        return {key: value for key, value in options.items() if value.strip()}
    if isinstance(raw, (list, tuple)):
        return {
            chr(65 + index): str(value)
            for index, value in enumerate(raw)
            if index < 26 and str(value).strip()
        }
    options: dict[str, str] = {}
    for key, value in row.items():
        key_text = str(key).strip()
        letter = re.fullmatch(r"(?:option|choice|class)?[_-]?([A-Ea-e])", key_text, flags=re.I)
        numeric = re.fullmatch(r"(?:option|choice|class)[_-]?([0-4])", key_text, flags=re.I)
        if letter:
            options[letter.group(1).upper()] = str(value)
        elif numeric:
            options[chr(65 + int(numeric.group(1)))] = str(value)
    return {key: value for key, value in options.items() if value.strip()}


def _resolve_image_path(raw_path: object, manifest_path: Path) -> Path | None:
    """Resolve image paths relative to the manifest, independent of cwd."""
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate
    manifest_relative = (manifest_path.parent / candidate).resolve()
    if manifest_relative.exists():
        return manifest_relative
    # Compatibility fallback for manifests whose paths were documented
    # relative to the launch directory.
    return candidate.resolve()


def _map_answer_to_option(answer: str, options: dict[str, str]) -> str:
    answer_norm = _normalize_label(answer)
    if not answer_norm:
        return ""
    exact = [key for key, value in options.items() if _normalize_label(value) == answer_norm]
    if exact:
        return sorted(exact)[0]
    contained = [key for key, value in options.items() if _normalize_label(value) in answer_norm]
    return sorted(contained, key=lambda key: len(_normalize_label(options[key])), reverse=True)[0] if contained else ""


def _extract_yes_no(raw_text: str) -> str:
    text = _normalize_label(raw_text)
    match = re.search(r"(?:final answer|answer|答案)\s+(yes|no)\b", text)
    if match:
        return match.group(1)
    tokens = re.findall(r"\b(yes|no)\b", text)
    return tokens[-1] if tokens else ""


def _score_prediction(
    row: dict[str, Any], prediction: str, *, allow_free_form_exact: bool = False
) -> tuple[bool, str, str, str]:
    """Apply the task-aware answer protocol used by the benchmark evaluators.

    The manifest may contain either option fields or a free-form label.  A raw
    rationale is never compared to an answer when an option/label vocabulary is
    available; this avoids the old normalized-rationale proxy.
    """
    answer = row.get("answer", row.get("ground_truth", row.get("gt", row.get("label", ""))))
    options = _option_map(row)
    if options:
        pred = _extract_choice(prediction, options)
        if not pred:
            pred = _map_answer_to_option(prediction, options)
        if not pred:
            numeric = re.search(r"\b(?:option[_ -]?)?([0-4])\b", str(prediction), flags=re.I)
            if numeric:
                index = int(numeric.group(1))
                ordered_keys = sorted(options)
                pred = ordered_keys[index] if index < len(ordered_keys) else ""
        gt = str(answer).strip().upper()
        if gt not in options:
            numeric = re.fullmatch(r"(?:option[_ -]?)?([0-4])", gt.casefold())
            if numeric:
                index = int(numeric.group(1))
                ordered_keys = sorted(options)
                gt = ordered_keys[index] if index < len(ordered_keys) else ""
            else:
                gt = _extract_choice(str(answer), options) or _map_answer_to_option(str(answer), options)
        return bool(pred) and pred == gt, "multiple_choice", pred, gt

    answer_text = str(answer or "").strip()
    if re.fullmatch(r"[A-Ea-e](?:\s*[,/]\s*[A-Ea-e]){1,3}", answer_text):
        gt = ",".join(part.strip().upper() for part in re.split(r"[,/]", answer_text))
        pred_letters = [token.upper() for token in re.findall(r"\b([A-E])\b", prediction.upper())]
        pred = ",".join(pred_letters[-len(gt.split(",")):]) if pred_letters else ""
        return pred == gt, "ordered_choice_sequence", pred, gt

    label_space = row.get("label_space", row.get("labels", []))
    if isinstance(label_space, (list, tuple)) and label_space:
        labels = [_normalize_label(label) for label in label_space if _normalize_label(label)]
        prediction_norm = _normalize_label(prediction)
        candidates = [label for label in labels if prediction_norm == label or re.search(r"(?:^|\s)" + re.escape(label) + r"(?:$|\s)", prediction_norm)]
        pred = sorted(candidates, key=len, reverse=True)[0] if candidates else ""
        gt = _normalize_label(answer_text)
        return pred == gt, "label_vocabulary", pred, gt

    if _normalize_label(answer_text) in {"yes", "no"}:
        pred, gt = _extract_yes_no(prediction), _normalize_label(answer_text)
        return pred == gt, "yes_no", pred, gt

    if not allow_free_form_exact:
        benchmark = row.get("benchmark", row.get("source_benchmark", "unknown"))
        raise ValueError(
            f"benchmark {benchmark!r} has no task-aware answer vocabulary; "
            "add options/label_space to the manifest or pass --allow_free_form_exact "
            "for an explicitly requested open-answer smoke test"
        )
    pred, gt = _normalize_label(prediction), _normalize_label(answer_text)
    return bool(pred) and pred == gt, "explicit_free_form_exact", pred, gt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="JSONL with question, answer, image, wrong_image")
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--base_model_path", default="")
    p.add_argument(
        "--mode",
        choices=["normal", "no_image", "wrong_image", "ground_truth_mask", "background_mask"],
        required=True,
    )
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--image_max_pixels", type=int, default=262144)
    p.add_argument("--image_min_pixels", type=int, default=50176)
    p.add_argument("--benchmark_field", default="benchmark", help="Manifest field used for benchmark-balanced aggregation")
    p.add_argument(
        "--expected_per_benchmark",
        type=int,
        default=500,
        help="Paper protocol samples 500 examples per benchmark; set 0 to skip this count check.",
    )
    p.add_argument(
        "--allow_nonpaper_sample_size",
        action="store_true",
        help="Allow a smoke-test manifest whose group size is not the paper's 500 examples.",
    )
    p.add_argument(
        "--allow_free_form_exact",
        action="store_true",
        help="Explicitly allow normalized free-form answer matching for a smoke test.",
    )
    args = p.parse_args()
    processor, model, _ = load_peft_vlm(
        args.adapter_path, args.base_model_path, args.device,
        args.image_max_pixels, args.image_min_pixels,
    )

    manifest_path = Path(args.manifest).expanduser().resolve()
    rows = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    predictions = []
    for idx, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", row.get("ground_truth", row.get("gt", row.get("label", ""))))).strip()
        image_key = {
            "normal": "image",
            "wrong_image": "wrong_image",
            "ground_truth_mask": "masked_image",
            "background_mask": "background_masked_image",
        }.get(args.mode)
        image_path = _resolve_image_path(row.get(image_key, ""), manifest_path)
        if args.mode in {"wrong_image", "ground_truth_mask", "background_mask"} and image_path is None:
            raise ValueError(f"row {idx} has no paired perturbation image")
        content = []
        images = []
        if args.mode == "no_image":
            # The paper's no-image condition keeps the multimodal template
            # and replaces diagnostic content with a blank visual input.
            original_path = _resolve_image_path(row.get("image", ""), manifest_path)
            if original_path is None:
                raise ValueError(f"row {idx} has no source image for blank control")
            with Image.open(original_path) as original:
                blank = Image.new("RGB", original.size, (0, 0, 0))
            content.append({"type": "image", "image": blank})
            images = [blank]
        else:
            if image_path is None:
                raise ValueError(f"row {idx} has no image path for {args.mode}")
            content.append({"type": "image", "image": str(image_path)})
            images = [Image.open(image_path).convert("RGB")]
        content.append({"type": "text", "text": _prompt_text(row, question)})
        conversation = [{"role": "user", "content": content}]
        prediction = run_chat_generation_generic(model, processor, conversation, images, args.device, args.max_new_tokens)
        benchmark = str(row.get(args.benchmark_field, row.get("source_benchmark", row.get("pair_group", "")))).strip()
        if not benchmark:
            raise ValueError(f"row {idx} has no benchmark field {args.benchmark_field!r}; benchmark-balanced accuracy is undefined")
        correct, matching_rule, parsed_prediction, parsed_answer = _score_prediction(
            row, prediction, allow_free_form_exact=args.allow_free_form_exact
        )
        predictions.append({
            "id": row.get("id", str(idx)),
            "benchmark": benchmark,
            "prediction": prediction,
            "parsed_prediction": parsed_prediction,
            "answer": answer,
            "parsed_answer": parsed_answer,
            "matching_rule": matching_rule,
            "correct": correct,
        })
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["benchmark"]].append(row)
    # The 500-per-benchmark requirement applies to Figure 3(a)'s image-level
    # audit.  The evidence-region audit is a single held-out GEMeX group and
    # is filtered separately by the shared-correct ID list.
    if args.mode in {"normal", "no_image", "wrong_image"} and args.expected_per_benchmark and not args.allow_nonpaper_sample_size:
        invalid = {name: len(items) for name, items in grouped.items() if len(items) != args.expected_per_benchmark}
        if invalid:
            raise ValueError(
                "Paper image-level audit requires 500 examples per benchmark; "
                f"got {invalid}. Use --allow_nonpaper_sample_size only for a smoke test."
            )
    benchmark_accuracy = {
        name: sum(bool(item["correct"]) for item in items) / len(items)
        for name, items in sorted(grouped.items())
    }
    accuracy = sum(benchmark_accuracy.values()) / len(benchmark_accuracy) if benchmark_accuracy else 0.0
    micro_accuracy = sum(bool(row["correct"]) for row in predictions) / len(predictions) if predictions else 0.0
    payload = {
        "metric": "benchmark_balanced_accuracy",
        "mode": args.mode,
        "manifest": str(manifest_path),
        "benchmark_field": args.benchmark_field,
        "n": len(predictions),
        "num_benchmarks": len(benchmark_accuracy),
        "samples_per_benchmark": {name: len(items) for name, items in sorted(grouped.items())},
        "benchmark_accuracy": benchmark_accuracy,
        "accuracy": accuracy,
        "micro_accuracy": micro_accuracy,
        "predictions": predictions,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mode": args.mode, "n": len(predictions), "accuracy": accuracy, "benchmark_accuracy": benchmark_accuracy}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
