#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import argparse
import os
import base64
import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Any

import torch
from peft import PeftConfig, PeftModel
from PIL import Image
from tqdm import tqdm
try:
    from .qwen3vl_common import load_peft_vlm, run_chat_generation_generic
except ImportError:  # direct ``python evaluation/...`` invocation
    from qwen3vl_common import load_peft_vlm, run_chat_generation_generic

Image.MAX_IMAGE_PIXELS = None


SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant. "
    "Answer strictly and concisely based on image evidence."
)

MCQ_OUTPUT_RULE = "Output exactly one line: Answer: <LETTER>. No explanation."


@dataclass
class Sample:
    sample_id: str
    question: str
    images: list[str]
    options: dict[str, str]
    ground_truth: str
    metadata: dict


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def decode_base64_image(payload: str) -> Image.Image:
    raw = base64.b64decode(payload)
    with Image.open(io.BytesIO(raw)) as img:
        return img.convert("RGB")


def extract_choice(raw_text: str, valid: Iterable[str]) -> str:
    valid_set = {v.upper() for v in valid}
    text = (raw_text or "").strip().replace("\u00a0", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text_upper = text.upper()

    m = re.search(r"(?i)(?:final\s+answer|answer|答案)\s*[:：]\s*([A-Z])\b", text)
    if m:
        cand = m.group(1).upper()
        if cand in valid_set:
            return cand

    m = re.search(r"\\boxed\{\s*([A-Z])\s*\}", text_upper)
    if m:
        cand = m.group(1).upper()
        if cand in valid_set:
            return cand

    m = re.search(r"(?i)option\s*([A-Z])\b", text)
    if m:
        cand = m.group(1).upper()
        if cand in valid_set:
            return cand

    m = re.search(r"(?i)\(([A-Z])\)", text)
    if m:
        cand = m.group(1).upper()
        if cand in valid_set:
            return cand

    all_letters = [c.upper() for c in re.findall(r"\b([A-Z])\b", text) if c.upper() in valid_set]
    if len(all_letters) == 1:
        return all_letters[0]

    if all_letters:
        tail = text_upper[-120:]
        tail_letters = [c.upper() for c in re.findall(r"\b([A-Z])\b", tail) if c.upper() in valid_set]
        if tail_letters:
            return tail_letters[-1]
        return all_letters[-1]

    lines = [ln.strip().upper() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines[-3:]):
        m = re.fullmatch(r"(?:ANSWER\s*[:：]\s*)?([A-Z])(?:[\).])?", line)
        if m:
            cand = m.group(1).upper()
            if cand in valid_set:
                return cand

    return ""


def match_option_by_text(raw_text: str, options: dict[str, str]) -> str:
    if not options:
        return ""
    raw_norm = normalize_text(raw_text)
    if not raw_norm:
        return ""

    exact_hits: list[tuple[int, str]] = []
    contain_hits: list[tuple[int, str]] = []
    for key, value in options.items():
        opt_norm = normalize_text(value)
        if not opt_norm:
            continue
        if raw_norm == opt_norm:
            exact_hits.append((len(opt_norm), key.upper()))
        elif opt_norm in raw_norm:
            contain_hits.append((len(opt_norm), key.upper()))

    if exact_hits:
        exact_hits.sort(reverse=True)
        return exact_hits[0][1]
    if contain_hits:
        contain_hits.sort(reverse=True)
        return contain_hits[0][1]

    return ""


def extract_label_from_vocab(raw_text: str, labels: Iterable[str]) -> str:
    raw_norm = normalize_text(raw_text)
    if not raw_norm:
        return ""

    exact_hits: list[tuple[int, str]] = []
    contain_hits: list[tuple[int, str]] = []
    for label in labels:
        label_norm = normalize_text(label)
        if not label_norm:
            continue
        if raw_norm == label_norm:
            exact_hits.append((len(label_norm), label_norm))
        elif re.search(r"(^|\s)" + re.escape(label_norm) + r"(\s|$)", raw_norm):
            contain_hits.append((len(label_norm), label_norm))

    if exact_hits:
        exact_hits.sort(reverse=True)
        return exact_hits[0][1]
    if contain_hits:
        contain_hits.sort(reverse=True)
        return contain_hits[0][1]

    return ""


def legacy_substring_choice(raw_text: str, valid: Iterable[str]) -> str:
    valid_set = {v.upper() for v in valid}
    upper = (raw_text or "").upper()
    for key in sorted(valid_set):
        if key in upper:
            return key
    return ""

def extract_yes_no(raw_text: str) -> str:
    text = (raw_text or "").strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    m = re.search(r"(?:final\s+answer|answer|答案)\s*[:：]\s*(yes|no)\b", text)
    if m:
        return m.group(1)
    tokens = re.findall(r"\b(yes|no)\b", text)
    if tokens:
        return tokens[-1]
    return ""


def map_gt_to_option(gt_text: str, options: dict[str, str]) -> str:
    gt_norm = normalize_text(gt_text)
    if not gt_norm:
        return ""

    exact_hits: list[tuple[int, str]] = []
    contain_hits: list[tuple[int, str]] = []
    for k, v in options.items():
        opt_norm = normalize_text(v)
        if not opt_norm:
            continue
        if opt_norm == gt_norm:
            exact_hits.append((len(opt_norm), k.upper()))
        elif opt_norm in gt_norm or gt_norm in opt_norm:
            contain_hits.append((len(opt_norm), k.upper()))

    if exact_hits:
        exact_hits.sort(reverse=True)
        return exact_hits[0][1]
    if contain_hits:
        contain_hits.sort(reverse=True)
        return contain_hits[0][1]
    return ""


def iter_gmai_val(path: str, max_samples: int) -> Iterator[Sample]:
    csv.field_size_limit(10**9)
    yielded = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            options = {
                key: (row.get(key) or "").strip()
                for key in ["A", "B", "C", "D", "E"]
                if (row.get(key) or "").strip()
            }
            if not row.get("image"):
                continue
            yield Sample(
                sample_id=str(row.get("index", yielded)),
                question=(row.get("question") or "").strip(),
                images=[f"base64://{row['image']}"],
                options=options,
                ground_truth=(row.get("answer") or "").strip().upper(),
                metadata={
                    "modality": row.get("modality", ""),
                    "category": row.get("category", ""),
                },
            )
            yielded += 1
            if max_samples > 0 and yielded >= max_samples:
                break



def iter_medxpert_test(path: str, image_root: str, max_samples: int) -> Iterator[Sample]:
    yielded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            options = {str(k).upper(): str(v) for k, v in (row.get("options") or {}).items()}
            image_paths = [str(Path(image_root) / p) for p in (row.get("images") or [])]
            if not image_paths:
                continue
            yield Sample(
                sample_id=str(row.get("id", yielded)),
                question=(row.get("question") or "").strip(),
                images=image_paths,
                options=options,
                ground_truth=str(row.get("label", "")).strip().upper(),
                metadata={
                    "medical_task": row.get("medical_task", ""),
                    "body_system": row.get("body_system", ""),
                    "question_type": row.get("question_type", ""),
                },
            )
            yielded += 1
            if max_samples > 0 and yielded >= max_samples:
                break



def _sort_labels(labels: set[str]) -> list[str]:
    if not labels:
        return []
    if all(label.isdigit() for label in labels):
        return [str(x) for x in sorted(int(v) for v in labels)]
    return sorted(labels)


def build_medblink_task_label_space(path: str) -> dict[str, list[str]]:
    label_map: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            task = str(row.get("task", "")).strip()
            label = str(row.get("answer", "")).strip().lower()
            if not task or not label:
                continue
            label_map.setdefault(task, set()).add(label)
    return {task: _sort_labels(values) for task, values in label_map.items()}


def iter_medblink(
    path: str,
    root: str,
    max_samples: int,
    task_label_space: dict[str, list[str]] | None = None,
) -> Iterator[Sample]:
    yielded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rel = row.get("image_relpath", "")
            image_path = str(Path(root) / rel)
            task = str(row.get("task", "")).strip()
            gt_label = str(row.get("answer", "")).strip().lower()

            options: dict[str, str] = {}
            ground_truth = gt_label
            label_space = []
            if task_label_space and task in task_label_space:
                label_space = task_label_space[task]
                options = {chr(65 + i): label for i, label in enumerate(label_space) if i < 26}
                for letter, label in options.items():
                    if label == gt_label:
                        ground_truth = letter
                        break

            yield Sample(
                sample_id=str(row.get("question_id", yielded)),
                question=f"Task: {task}\nQuestion: {(row.get('question') or '').strip()}",
                images=[image_path],
                options=options,
                ground_truth=ground_truth,
                metadata={"task": task, "label_space": label_space},
            )
            yielded += 1
            if max_samples > 0 and yielded >= max_samples:
                break



MISSING_IMAGE_COUNTS: Counter = Counter()


def _handle_missing_image(benchmark: str, image_path: str, sample_id: object, skip_missing_images: bool) -> None:
    """Either count a missing image explicitly or stop the evaluation.

    Silently dropping rows would shrink the benchmark denominator and inflate
    accuracy; the DrVD evaluator already refuses to do that.
    """
    if skip_missing_images:
        MISSING_IMAGE_COUNTS[benchmark] += 1
        return
    raise FileNotFoundError(
        f"{benchmark} sample {sample_id!r} image does not exist: {image_path}. "
        "Fix the benchmark image root, or pass --skip_missing_images to count and skip such rows explicitly."
    )


def iter_omni_open(open_dir: str, root: str, max_samples: int, skip_missing_images: bool = False) -> Iterator[Sample]:
    yielded = 0
    for json_file in sorted(Path(open_dir).glob("*.json")):
        records = json.loads(Path(json_file).read_text(encoding="utf-8"))
        for row in records:
            image_rel = str(row.get("image_path", ""))
            image_path = str(Path(root) / image_rel)
            if not Path(image_path).exists():
                _handle_missing_image("omnimedvqa_open", image_path, row.get("question_id", yielded), skip_missing_images)
                continue
            options = {
                "A": str(row.get("option_A", "")).strip(),
                "B": str(row.get("option_B", "")).strip(),
                "C": str(row.get("option_C", "")).strip(),
                "D": str(row.get("option_D", "")).strip(),
            }
            options = {k: v for k, v in options.items() if v}
            yield Sample(
                sample_id=str(row.get("question_id", yielded)),
                question=(row.get("question") or "").strip(),
                images=[image_path],
                options=options,
                ground_truth=str(row.get("gt_answer", "")).strip(),
                metadata={
                    "dataset": row.get("dataset", ""),
                    "question_type": row.get("question_type", ""),
                    "modality_type": row.get("modality_type", ""),
                },
            )
            yielded += 1
            if max_samples > 0 and yielded >= max_samples:
                return


def iter_multiple_disease_identification(path: str, image_root: str, max_samples: int, skip_missing_images: bool = False) -> Iterator[Sample]:
    """Load the CheXbench Multiple Disease Identification task.

    The benchmark stores option fields as ``option_0``, ``option_1``, ... and
    answers either as a zero-based index, an option letter, or option text.
    Converting all three forms to one letter here keeps accuracy and Macro-F1
    evaluation follows the Qwen3-VL benchmark protocol.
    """
    yielded = 0
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    for row_idx, row in enumerate(rows):
        image_rel = str(row.get("image_path", "")).strip()
        if not image_rel:
            continue
        image_path = Path(image_root) / image_rel
        if not image_path.exists():
            _handle_missing_image("multiple_disease_identification", str(image_path), row.get("id", row_idx), skip_missing_images)
            continue

        options = {
            chr(65 + idx): str(row.get(f"option_{idx}", "")).strip()
            for idx in range(26)
            if str(row.get(f"option_{idx}", "")).strip()
        }
        if not options:
            continue

        raw_answer = row.get("answer", "")
        answer_text = str(raw_answer).strip()
        if isinstance(raw_answer, int) or answer_text.isdigit():
            answer_idx = int(raw_answer)
            ground_truth = chr(65 + answer_idx) if chr(65 + answer_idx) in options else ""
        elif len(answer_text) == 1 and answer_text.upper() in options:
            ground_truth = answer_text.upper()
        else:
            ground_truth = map_gt_to_option(answer_text, options) or answer_text.upper()

        yield Sample(
            sample_id=str(row.get("id", row_idx)),
            question=(row.get("question") or "").strip(),
            images=[str(image_path)],
            options=options,
            ground_truth=ground_truth,
            metadata={
                "task_name": row.get("task_name", ""),
                "data_source": row.get("data_source", ""),
                "slice": row.get("slice", ""),
                "image_path": image_rel,
            },
        )
        yielded += 1
        if max_samples > 0 and yielded >= max_samples:
            break


def _macro_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, float, float, float]:
    true_counter = Counter(y_true)
    pred_counter = Counter(y_pred)
    pair_counter = Counter(zip(y_true, y_pred))
    macro_p = macro_r = macro_f1 = weighted_f1 = 0.0
    for label in labels:
        tp = pair_counter[(label, label)]
        fp = pred_counter[label] - tp
        fn = true_counter[label] - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_f1 += f1 * true_counter[label]
    k = float(len(labels))
    return macro_p / k, macro_r / k, macro_f1 / k, (weighted_f1 / len(y_true) if y_true else 0.0)


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Macro metrics over the benchmark's true label set.

    An unparseable or invalid prediction (``__EMPTY__`` or a letter outside the
    option set) counts as a wrong answer for its true class.  It is not treated
    as an additional class, because that would change the macro denominator
    with the model's failure modes.  The union-label variant is reported for
    comparison under ``*_including_invalid_class``.
    """
    true_labels = sorted(set(y_true))
    if not true_labels:
        return {}
    union_labels = sorted(set(y_true) | set(y_pred))
    macro_p, macro_r, macro_f1, weighted_f1 = _macro_metrics(y_true, y_pred, true_labels)
    _, _, macro_f1_union, _ = _macro_metrics(y_true, y_pred, union_labels)
    invalid = sum(1 for pred in y_pred if pred not in true_labels)
    return {
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_f1_including_invalid_class": round(macro_f1_union, 6),
        "invalid_predictions": int(invalid),
        "label_set_size": len(true_labels),
    }


def load_rgb_images(items: list[str]) -> list[Image.Image]:
    resolved: list[Image.Image] = []
    for item in items:
        if isinstance(item, str) and item.startswith("base64://"):
            payload = item[len("base64://") :]
            resolved.append(decode_base64_image(payload))
        else:
            with Image.open(item) as img:
                resolved.append(img.convert("RGB"))
    return resolved


def run_chat_generation(
    model: torch.nn.Module,
    processor: Any,
    conversation: list[dict],
    images: list[Image.Image],
    device: str,
    max_new_tokens: int,
) -> str:
    return run_chat_generation_generic(model, processor, conversation, images, device, max_new_tokens)


def predict_one(
    model: torch.nn.Module,
    processor: Any,
    sample: Sample,
    device: str,
    max_new_tokens: int,
    mcq_style: str,
) -> str:
    images = load_rgb_images(sample.images)

    if sample.options:
        task_hint = ""
        task_name = str(sample.metadata.get("task", "")).strip() if isinstance(sample.metadata, dict) else ""
        if task_name:
            task_hint = f"Task: {task_name}\n"

        option_text = "\n".join([f"{k}. {v}" for k, v in sample.options.items()])

        if mcq_style == "two_stage":
            stage1_text = (
                f"{task_hint}Question: {sample.question}\n"
                f"Options:\n{option_text}\n"
                "Briefly reason about the image and options in at most 2 short sentences. "
                "Do not output the final option letter."
            )
            stage1_conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": ([{"type": "image"} for _ in sample.images] + [{"type": "text", "text": stage1_text}]),
                },
            ]
            rationale = run_chat_generation(
                model=model,
                processor=processor,
                conversation=stage1_conversation,
                images=images,
                device=device,
                max_new_tokens=max(24, min(max_new_tokens * 2, 64)),
            )

            stage2_text = (
                f"Question: {sample.question}\n"
                f"Options:\n{option_text}\n"
                f"Rationale: {rationale}\n"
                f"Now return the final option letter only. {MCQ_OUTPUT_RULE}"
            )
            stage2_conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": ([{"type": "image"} for _ in sample.images] + [{"type": "text", "text": stage2_text}]),
                },
            ]
            return run_chat_generation(
                model=model,
                processor=processor,
                conversation=stage2_conversation,
                images=images,
                device=device,
                max_new_tokens=min(max_new_tokens, 12),
            )

        if mcq_style == "cot_brief":
            user_text = (
                f"{task_hint}Question: {sample.question}\n"
                f"Options:\n{option_text}\n"
                "Think briefly (<=2 sentences), then output final letter.\n"
                "Format exactly:\nReason: <short>\nAnswer: <LETTER>"
            )
        else:
            user_text = (
                f"{task_hint}Question: {sample.question}\n"
                f"Options:\n{option_text}\n"
                f"Select the single best option.\n"
                f"{MCQ_OUTPUT_RULE}"
            )
    else:
        user_text = f"Question: {sample.question}\nOutput only the final short answer."

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": ([{"type": "image"} for _ in sample.images] + [{"type": "text", "text": user_text}]),
        },
    ]
    gen_tokens = min(max_new_tokens, 12) if sample.options else max_new_tokens
    return run_chat_generation(
        model=model,
        processor=processor,
        conversation=conversation,
        images=images,
        device=device,
        max_new_tokens=gen_tokens,
    )


def evaluate_samples(
    benchmark: str,
    samples: Iterable[Sample],
    model: torch.nn.Module,
    processor: Any,
    device: str,
    max_new_tokens: int,
    out_jsonl: Path,
    resume: bool,
    mcq_style: str,
) -> dict:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed_ids: set[str] = set()
    correct = 0
    total = 0
    cls_true: list[str] = []
    cls_pred: list[str] = []

    mode = "w"
    if resume and out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(row.get("id", ""))
                if not sid:
                    continue
                if sid in processed_ids:
                    continue
                processed_ids.add(sid)
                total += 1
                if bool(row.get("correct", False)):
                    correct += 1
                gt_old = str(row.get("gt", "")).strip()
                pred_old = str(row.get("pred", "")).strip()
                if gt_old:
                    cls_true.append(gt_old)
                    cls_pred.append(pred_old if pred_old else "__EMPTY__")
        mode = "a"
        print(f"[{benchmark}] resume enabled: loaded {len(processed_ids)} existing predictions")

    with out_jsonl.open(mode, encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"{benchmark}"):
            sid = str(sample.sample_id)
            if sid in processed_ids:
                continue
            raw_pred = predict_one(
                model,
                processor,
                sample,
                device=device,
                max_new_tokens=max_new_tokens,
                mcq_style=mcq_style,
            )
            if sample.options:
                pred = extract_choice(raw_pred, sample.options.keys())
                if not pred:
                    pred = match_option_by_text(raw_pred, sample.options)
                if not pred and benchmark == "medxpertqa_mm_test":
                    pred = legacy_substring_choice(raw_pred, sample.options.keys())
                gt = sample.ground_truth.upper()
                if benchmark == "omnimedvqa_open" and len(gt) != 1:
                    mapped = map_gt_to_option(sample.ground_truth, sample.options)
                    gt = (mapped or sample.ground_truth).upper()
                is_correct = bool(pred) and pred == gt
                cls_true.append(gt)
                cls_pred.append(pred if pred else "__EMPTY__")
            else:
                if benchmark == "medblink_val":
                    label_space = []
                    if isinstance(sample.metadata, dict):
                        label_space = [str(x).lower() for x in sample.metadata.get("label_space", [])]
                    pred_label = extract_label_from_vocab(raw_pred, label_space) if label_space else ""
                    if not pred_label:
                        yn = extract_yes_no(raw_pred)
                        pred_label = yn if yn else normalize_text(raw_pred)
                    pred = pred_label
                else:
                    pred = normalize_text(raw_pred)
                gt = normalize_text(sample.ground_truth)
                is_correct = pred == gt
                cls_true.append(gt)
                cls_pred.append(pred if pred else "__EMPTY__")

            if is_correct:
                correct += 1
            total += 1

            row = {
                "benchmark": benchmark,
                "id": sid,
                "question": sample.question,
                "pred_raw": raw_pred,
                "pred": pred,
                "gt": sample.ground_truth,
                "correct": is_correct,
                "metadata": sample.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    accuracy = (correct / total) if total else 0.0
    result = {
        "benchmark": benchmark,
        "samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "output_jsonl": str(out_jsonl),
    }
    if cls_true and cls_pred and len(cls_true) == len(cls_pred):
        result.update(compute_classification_metrics(cls_true, cls_pred))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model_path", default="")
    parser.add_argument("--benchmarks", nargs="+", default=["gmai_val", "medblink_val", "omnimedvqa_open"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--medblink_use_task_label_space", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mcq_style",
        choices=["strict", "cot_brief", "two_stage"],
        default="strict",
        help="Prompting style for MCQ benchmarks",
    )
    parser.add_argument("--gmai_val_tsv", default="${EVAL_ROOT}/GMAI-MMBench/GMAI_mm_bench_VAL.tsv")
    parser.add_argument("--medblink_jsonl", default="${EVAL_ROOT}/MedBLINK/metadata.jsonl")
    parser.add_argument("--medblink_root", default="${EVAL_ROOT}/MedBLINK")
    parser.add_argument("--omni_open_dir", default="${EVAL_ROOT}/OmniMedVQA/QA_information/Open-access")
    parser.add_argument("--omni_root", default="${EVAL_ROOT}/OmniMedVQA")
    parser.add_argument("--medxpert_test_jsonl", default="${EVAL_ROOT}/MedXpertQA/test.jsonl")
    parser.add_argument("--medxpert_images", default="${EVAL_ROOT}/MedXpertQA/images")
    parser.add_argument(
        "--multiple_disease_json",
        default="${EVAL_ROOT}/Multiple_Disease_Identification/Multiple_Disease_Identification.json",
    )
    parser.add_argument(
        "--multiple_disease_root",
        default="${EVAL_ROOT}/Multiple_Disease_Identification/images",
    )
    parser.add_argument("--image_max_pixels", type=int, default=262144)
    parser.add_argument("--image_min_pixels", type=int, default=50176)
    parser.add_argument(
        "--skip_missing_images",
        action="store_true",
        help="Count and skip benchmark rows whose image file is missing instead of stopping (not the paper protocol).",
    )
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, str):
            setattr(args, key, os.path.expandvars(value))

    adapter_path = args.adapter_path
    base_model_path = args.base_model_path
    processor, model, backbone = load_peft_vlm(
        adapter_path=adapter_path,
        base_model_path=base_model_path,
        device=args.device,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    medblink_label_space = None
    if bool(args.medblink_use_task_label_space):
        medblink_label_space = build_medblink_task_label_space(args.medblink_jsonl)
        print(f"[medblink_val] loaded task label spaces for {len(medblink_label_space)} tasks")

    for benchmark in args.benchmarks:
        if benchmark == "gmai_val":
            samples = iter_gmai_val(args.gmai_val_tsv, args.max_samples)
        elif benchmark == "medblink_val":
            samples = iter_medblink(
                args.medblink_jsonl,
                args.medblink_root,
                args.max_samples,
                task_label_space=medblink_label_space,
            )
        elif benchmark == "omnimedvqa_open":
            samples = iter_omni_open(args.omni_open_dir, args.omni_root, args.max_samples, args.skip_missing_images)
        elif benchmark == "medxpertqa_mm_test":
            samples = iter_medxpert_test(args.medxpert_test_jsonl, args.medxpert_images, args.max_samples)
        elif benchmark == "multiple_disease_identification":
            samples = iter_multiple_disease_identification(
                args.multiple_disease_json,
                args.multiple_disease_root,
                args.max_samples,
                args.skip_missing_images,
            )
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        out_jsonl = output_dir / f"{benchmark}_predictions.jsonl"
        result = evaluate_samples(
            benchmark=benchmark,
            samples=samples,
            model=model,
            processor=processor,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            out_jsonl=out_jsonl,
            resume=bool(args.resume),
            mcq_style=args.mcq_style,
        )
        result["skipped_missing_images"] = int(MISSING_IMAGE_COUNTS.get(benchmark, 0))
        summary.append(result)

    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
