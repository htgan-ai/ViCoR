#!/usr/bin/env python3
"""Score every CheXbench option and report the paper's fine-grained metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

try:
    from .qwen3vl_common import apply_chat_template_compat, load_peft_vlm
except ImportError:  # direct ``python evaluation/...`` invocation
    from qwen3vl_common import apply_chat_template_compat, load_peft_vlm


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().replace("-", " ").replace("/", " ").replace("&", " and ").split())


def build_alias_map(config_path: str) -> dict[str, str]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    selection = config["vicor"]["selection"]
    vocab = [normalize_text(x) for x in selection.get("target_label_vocab", [])]
    aliases = selection.get("target_label_aliases", {})
    result = {x: x for x in vocab}
    for canonical, raw_aliases in aliases.items():
        canon = normalize_text(canonical)
        result[canon] = canon
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        for alias in raw_aliases:
            result[normalize_text(alias)] = canon
    return result


def parse_disease_set(text: str, alias_map: dict[str, str]) -> set[str]:
    normalized = normalize_text(text)
    return {canonical for phrase, canonical in alias_map.items() if phrase and phrase in normalized}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_options(processor: Any, model: torch.nn.Module, image: Image.Image, question: str, options: dict[str, str], device: str) -> dict[str, float]:
    """Score one completion per option *label* (``A``, ``B``, ...).

    The paper's option-level likelihood is deliberately computed over the
    answer labels rather than the option descriptions.  Scoring the complete
    descriptions would make the result depend on description length,
    wording, and the tokenization of the finding names.
    """
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "Carefully inspect the image and select the best option."}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Question: {question}\nOptions:\n{option_text}\nAnswer:"}]},
    ]
    prompt = apply_chat_template_compat(processor, conversation, add_generation_prompt=True)
    letters = sorted(options)
    targets = letters
    prompt_inputs = processor(text=[prompt] * len(targets), images=[image] * len(targets), padding=True, return_tensors="pt")
    full_inputs = processor(text=[prompt + target for target in targets], images=[image] * len(targets), padding=True, return_tensors="pt")
    dtype = next(model.parameters()).dtype
    for inputs in (prompt_inputs, full_inputs):
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    prompt_lens = prompt_inputs.attention_mask.sum(dim=1).tolist()
    full_inputs = full_inputs.to(device)
    labels = full_inputs.input_ids.clone()
    full_attention = full_inputs.attention_mask.bool()
    prompt_attention = prompt_inputs.attention_mask.bool()
    for index, prompt_len in enumerate(prompt_lens):
        # Qwen3-VL uses left padding.  Prompt lengths are counts of valid
        # tokens, so masking from column zero would include padding and would
        # leave part of the prompt in the option likelihood.  Locate the
        # valid positions first, then expose only the appended option tokens.
        valid_positions = full_attention[index].nonzero(as_tuple=True)[0]
        prompt_positions = prompt_attention[index].nonzero(as_tuple=True)[0]
        if int(prompt_len) >= int(valid_positions.numel()):
            raise ValueError("Option text is empty or the prompt is not a prefix of the full input")
        if not torch.equal(
            full_inputs.input_ids[index, valid_positions[: int(prompt_len)]].cpu(),
            prompt_inputs.input_ids[index, prompt_positions[: int(prompt_len)]].cpu(),
        ):
            raise ValueError("Option likelihood prompt tokenization is not a verified prefix")
        labels[index, ~full_attention[index]] = -100
        labels[index, valid_positions[: int(prompt_len)]] = -100
    with torch.inference_mode():
        logits = model(**full_inputs, use_cache=False).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(~valid, 0.0)
    mean_logprobs = token_logprobs.sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    return {key: float(value) for key, value in zip(letters, mean_logprobs.detach().cpu().tolist())}


def resolve_ground_truth(raw_answer: Any, options: dict[str, str]) -> str:
    """Map benchmark answers to an option label without fabricating labels.

    CheXbench stores answers as zero-based indices, labels, or option text.
    Invalid numeric values (for example ``99``) are represented by an empty
    label so that an imperfect input row is auditable instead of raising a
    ``KeyError`` while computing the metrics.
    """
    if isinstance(raw_answer, bool):
        return ""
    answer_text = str(raw_answer).strip()
    if isinstance(raw_answer, int) or (answer_text and answer_text.lstrip("+-").isdigit()):
        try:
            answer_index = int(raw_answer)
        except (TypeError, ValueError):
            return ""
        ordered_keys = sorted(options)
        return ordered_keys[answer_index] if 0 <= answer_index < len(ordered_keys) else ""
    if len(answer_text) == 1 and answer_text.upper() in options:
        return answer_text.upper()
    normalized_answer = normalize_text(answer_text)
    if normalized_answer:
        for key, value in options.items():
            if normalize_text(value) == normalized_answer:
                return key
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--benchmark_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/cxr_multi_qwen3vl_vicor.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--image_max_pixels", type=int, default=262144)
    parser.add_argument("--image_min_pixels", type=int, default=50176)
    args = parser.parse_args()
    processor, model, _ = load_peft_vlm(
        args.adapter_path, args.base_model_path, args.device,
        args.image_max_pixels, args.image_min_pixels,
    )
    alias_map = build_alias_map(args.config)
    items = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
    if args.max_samples > 0:
        items = items[: args.max_samples]

    rows = []
    for index, item in enumerate(items):
        image = Image.open(Path(args.image_root) / str(item["image_path"])).convert("RGB")
        options = {chr(65 + i): str(item[f"option_{i}"]) for i in range(26) if str(item.get(f"option_{i}", "")).strip()}
        scores = score_options(processor, model, image, str(item.get("question", "")), options, args.device)
        raw_answer = item.get("answer", "")
        ground_truth = resolve_ground_truth(raw_answer, options)
        probabilities = torch.softmax(torch.tensor([scores[key] for key in sorted(scores)]), dim=0).tolist()
        probability_map = dict(zip(sorted(scores), probabilities))
        gt_set = parse_disease_set(options.get(ground_truth, ""), alias_map) if ground_truth else set()
        incorrect = [key for key in options if key != ground_truth]
        near = (
            max(incorrect, key=lambda key: (jaccard(parse_disease_set(options[key], alias_map), gt_set), scores[key]))
            if ground_truth and incorrect
            else ""
        )
        predicted = max(scores, key=scores.get)
        rows.append({
            "id": str(item.get("id", index)),
            "answer": ground_truth,
            "pred": predicted,
            "scores": scores,
            "probabilities": probability_map,
            "near_miss": near,
            "correct": bool(ground_truth) and predicted == ground_truth,
            "near_miss_win": bool(ground_truth and near) and scores[ground_truth] > scores[near],
            "invalid_answer": bool(raw_answer not in (None, "") and not ground_truth),
        })

    n = len(rows)
    summary = {
        "samples": n,
        "choice_accuracy": sum(row["correct"] for row in rows) / n if n else 0.0,
        "correct_option_probability": sum(row["probabilities"].get(row["answer"], 0.0) for row in rows) / n if n else 0.0,
        "near_miss_win_rate": sum(row["near_miss_win"] for row in rows) / n if n else 0.0,
        "invalid_answers": sum(row["invalid_answer"] for row in rows),
        "output_jsonl": str(Path(args.output).with_suffix(".jsonl")),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
