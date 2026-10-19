#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import argparse
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
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


# DrVD-Bench's official report-generation protocol first asks DeepSeek to
# extract visually identifiable findings, then computes sentence BLEU and
# BERTScore F1 on those extracted findings.  These imports are intentionally
# lazy in ``compute_report_generation_metrics``: QA-only evaluation should not
# require the NLP metric packages or a DeepSeek key.
REPORT_BERTSCORE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant. "
    "Answer strictly and concisely based on image evidence."
)

MCQ_OUTPUT_RULE = "Output exactly one line: Answer: <LETTER>. No explanation."
JOINT_OUTPUT_RULE = "Output exactly one line with one option letter per level, separated by commas. No explanation."


@dataclass
class Sample:
    sample_id: str
    question: str
    images: list[str]
    options: dict[str, str]
    ground_truth: str
    metadata: dict
    task_type: str  # single | joint | multi_turn | report


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _report_keyword_prompt(text: str) -> str:
    return (
        "Given the following description of a medical image, extract only clinically "
        "relevant information that can be visually determined from the image. "
        "This includes both normal findings (e.g., 'no lung opacity', 'normal heart size') "
        "and abnormal findings (e.g., 'fracture', 'tumor mass'). Exclude details that "
        "cannot be inferred from the image itself (e.g., patient history, lab values).\n\n"
        f"{text}\n\n"
        "Return a concise, comma-separated list of visually identifiable clinical features. "
        "Do not include irrelevant words or explanations."
    )


def _extract_report_keywords(texts: list[str], api_key: str) -> list[str]:
    """Extract report findings with the official DrVD DeepSeek protocol."""
    if not api_key.strip():
        raise ValueError(
            "DrVD official report metrics require a DeepSeek API key. Pass "
            "--deepseek_api_key or set DEEPSEEK_API_KEY. Use "
            "--report_metric_mode raw only for an explicitly non-official smoke test."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - only reached in incomplete envs
        raise RuntimeError("Official report metrics require the 'openai' package") from exc

    client = OpenAI(api_key=api_key.strip(), base_url="https://api.deepseek.com")
    cache: dict[str, str] = {}

    def one(text: str) -> str:
        if text in cache:
            return cache[text]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": _report_keyword_prompt(text)}],
            temperature=0,
        )
        content = str(response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("DeepSeek returned an empty report keyword list")
        cache[text] = content
        return content

    # Preserve input order while avoiding duplicate API calls.  The official
    # implementation parallelizes extraction; a bounded pool keeps the same
    # behavior without creating one thread per report.
    unique = list(dict.fromkeys(texts))
    with ThreadPoolExecutor(max_workers=min(16, max(1, len(unique)))) as pool:
        extracted = list(pool.map(one, unique))
    mapping = dict(zip(unique, extracted))
    return [mapping[text] for text in texts]


def compute_report_generation_metrics(
    prediction_jsonl: Path,
    *,
    deepseek_api_key: str = "",
    mode: str = "official",
    bert_score_model: str = REPORT_BERTSCORE_MODEL,
) -> dict[str, Any]:
    """Compute DrVD report BLEU and BERTScore F1 from official fields.

    The official evaluator consumes ``answer`` and ``model_response``.  The
    release evaluator writes those fields for every report row and refuses to
    silently substitute the internal ``gt``/``pred`` fields.  ``mode=official``
    performs the DeepSeek visual-finding extraction used by DrVD-Bench.  The
    explicit ``mode=raw`` option is available only for offline smoke tests and
    is labeled as such in the returned protocol metadata.
    """
    if mode not in {"official", "raw"}:
        raise ValueError(f"Unsupported report metric mode: {mode}")

    references: list[str] = []
    predictions: list[str] = []
    modalities: list[str] = []
    with prediction_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Report prediction line {line_no} is not an object")
            if "answer" not in row or "model_response" not in row:
                raise ValueError(
                    f"Report prediction line {line_no} must contain answer and model_response"
                )
            answer = str(row.get("answer") or "").strip()
            response = str(row.get("model_response") or "").strip()
            if not answer or not response:
                # This mirrors the official script's non-empty pair contract,
                # but reports the offending record instead of silently dropping it.
                raise ValueError(f"Report prediction line {line_no} has an empty answer/model_response")
            references.append(answer)
            predictions.append(response)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            modalities.append(str(row.get("modality") or metadata.get("modality") or "Unknown").strip() or "Unknown")

    if not references:
        return {
            "bleu": 0.0,
            "bert_f1": 0.0,
            "metric_protocol": "drvd_official_keyword_extraction" if mode == "official" else "raw_text_smoke",
            "metric_samples": 0,
        }

    if mode == "official":
        references = _extract_report_keywords(references, deepseek_api_key)
        predictions = _extract_report_keywords(predictions, deepseek_api_key)
        protocol = "drvd_official_keyword_extraction"
    else:
        protocol = "raw_text_smoke"

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError as exc:  # pragma: no cover - only reached in incomplete envs
        raise RuntimeError("Report metrics require the 'nltk' package") from exc
    try:
        from bert_score import score as bert_score
    except ImportError as exc:  # pragma: no cover - only reached in incomplete envs
        raise RuntimeError("Report metrics require the 'bert-score' package") from exc

    smoothing = SmoothingFunction().method4
    def score_group(group_references: list[str], group_predictions: list[str]) -> tuple[float, float]:
        bleu_values = [
            sentence_bleu([reference.split()], prediction.split(), smoothing_function=smoothing)
            for reference, prediction in zip(group_references, group_predictions)
        ]
        _, _, f1 = bert_score(
            group_predictions,
            group_references,
            model_type=bert_score_model,
            num_layers=12,
            lang="en",
            rescale_with_baseline=False,
            device="cpu",
        )
        return (
            sum(bleu_values) / len(bleu_values) if bleu_values else 0.0,
            float(f1.mean().item()) if len(f1) else 0.0,
        )

    # DrVD's reference implementation reports report-generation scores by
    # modality.  Use a macro average over modality scores for the scalar
    # summary, while retaining each modality so the aggregation is auditable.
    grouped: dict[str, tuple[list[str], list[str]]] = {}
    for modality, reference, prediction in zip(modalities, references, predictions):
        refs, preds = grouped.setdefault(modality, ([], []))
        refs.append(reference)
        preds.append(prediction)
    by_modality: dict[str, dict[str, Any]] = {}
    for modality in sorted(grouped):
        refs, preds = grouped[modality]
        bleu, bert_f1 = score_group(refs, preds)
        by_modality[modality] = {
            "bleu": round(bleu, 6),
            "bert_f1": round(bert_f1, 6),
            "samples": len(refs),
        }
    bleu = sum(item["bleu"] for item in by_modality.values()) / len(by_modality)
    bert_f1 = sum(item["bert_f1"] for item in by_modality.values()) / len(by_modality)
    return {
        "bleu": round(bleu, 6),
        "bert_f1": round(bert_f1, 6),
        "metric_protocol": protocol,
        "metric_samples": len(references),
        "bert_score_model": bert_score_model,
        "modality_aggregation": "macro",
        "by_modality": by_modality,
    }


def parse_option_list(items: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    next_idx = 0
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        m = re.match(r"^\s*([A-Z])\s*[\.:\)]\s*(.*)\s*$", s)
        if m:
            k = m.group(1).upper()
            v = m.group(2).strip()
        else:
            k = chr(ord("A") + next_idx)
            v = s
            next_idx += 1
        if k not in options:
            options[k] = v
    return options


def parse_options(raw: object, row: dict[str, Any] | None = None, prefix: str = "") -> dict[str, str]:
    """Normalize the option encodings used by the public DrVD exports.

    Official files generally store ``["A. ...", "B. ..."]``.  A few
    mirrors use a mapping or flat ``choice_A``/``class_A`` fields instead;
    accepting those representations here avoids silently treating option
    labels as option text.
    """
    if isinstance(raw, dict):
        return {str(key).strip().upper(): str(value).strip() for key, value in raw.items() if str(key).strip()}
    if isinstance(raw, (list, tuple)):
        return parse_option_list([str(value) for value in raw])
    if row is not None:
        flat: dict[str, str] = {}
        for key, value in row.items():
            key_text = str(key)
            match = re.fullmatch(rf"(?:{re.escape(prefix)})(?:option|choice|class)[_-]?([A-Z]|[0-9]+)", key_text, re.I)
            if match:
                label = match.group(1).upper()
                if label.isdigit():
                    label = chr(ord("A") + int(label))
                flat[label] = str(value)
        if flat:
            return dict(sorted(flat.items()))
    return {}


def _row_image(row: dict[str, Any]) -> object:
    """Return the first supported image field without changing its value."""
    for key in ("image_paths", "image_path", "image", "img"):
        value = row.get(key)
        if value not in (None, ""):
            if isinstance(value, (list, tuple)):
                return value[0] if value else ""
            return value
    return ""


def _row_level(row: dict[str, Any], benchmark: str) -> str:
    value = row.get("task", row.get("level", row.get("subtask", row.get("question_type", ""))))
    level = str(value or "").strip()
    if not level:
        raise ValueError(
            f"DrVD {benchmark} sample {row.get('unique_id', row.get('id', '?'))!r} has no task/level; "
            "the official protocol requires accuracy to be computed within each level"
        )
    return level


def _row_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("unique_id", "id", "sample_id", "question_id", "uid"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(fallback)


def parse_answer_letter(ans: str) -> str:
    text = str(ans or "").strip().upper()
    m = re.match(r"^\s*([A-Z])\s*(?:[\.:\)].*)?$", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z])\b", text)
    return m.group(1) if m else ""


def extract_choice(raw_text: str, valid: Iterable[str]) -> str:
    valid_set = {v.upper() for v in valid}
    text = (raw_text or "").strip().replace("\u00a0", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text_upper = text.upper()

    m = re.search(r"(?i)(?:final\s+answer|answer|答案)\s*[:：]\s*([A-Z])\b", text)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    m = re.search(r"\\boxed\{\s*([A-Z])\s*\}", text_upper)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    m = re.search(r"(?i)option\s*([A-Z])\b", text)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    m = re.search(r"(?i)\(([A-Z])\)", text)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

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
    for line in reversed(lines[-4:]):
        m = re.fullmatch(r"(?:ANSWER\s*[:：]\s*)?([A-Z])(?:[\).])?", line)
        if m and m.group(1).upper() in valid_set:
            return m.group(1).upper()

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


def extract_joint_letters(raw_text: str, valid_per_step: list[set[str]]) -> list[str]:
    """Extract one answer per DrVD reasoning level (four or five levels)."""
    n_steps = len(valid_per_step)
    if n_steps == 0:
        return []
    text = (raw_text or "").strip().upper()
    text = re.sub(r"<[^>]+>", " ", text)

    def validate(tokens: list[str]) -> list[str]:
        if len(tokens) != n_steps:
            return []
        return tokens if all(tokens[i] in valid_per_step[i] for i in range(n_steps)) else []

    # Accept comma-separated output with or without an ``Answer:`` prefix,
    # including the Python-list representation emitted by the official demo.
    answer_match = re.search(r"(?:FINAL\s+ANSWER|ANSWER|ANSWERS?)\s*[:：]\s*(.+)", text)
    candidates_text = answer_match.group(1) if answer_match else text
    tokens = [token.upper() for token in re.findall(r"\b([A-Z])\b", candidates_text)]
    all_valid = set().union(*valid_per_step)
    valid_tokens = [token for token in tokens if token in all_valid]
    candidate = validate(valid_tokens[:n_steps])
    if candidate:
        return candidate
    candidate = validate(valid_tokens[-n_steps:])
    if candidate:
        return candidate

    # A stricter fallback handles compact comma-separated output where an
    # option letter is adjacent to punctuation or quotes.
    compact = re.findall(r"[A-Z]", candidates_text)
    candidate = validate([token for token in compact[:n_steps]])
    if candidate:
        return candidate
    candidate = validate([token for token in compact[-n_steps:]])
    return candidate


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
    """Macro metrics over the true label set; invalid predictions are errors, not a class."""
    true_labels = sorted(set(y_true))
    if not true_labels:
        return {}
    union_labels = sorted(set(y_true) | set(y_pred))
    macro_p, macro_r, macro_f1, weighted_f1 = _macro_metrics(y_true, y_pred, true_labels)
    _, _, macro_f1_union, _ = _macro_metrics(y_true, y_pred, union_labels)
    return {
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_f1_including_invalid_class": round(macro_f1_union, 6),
        "invalid_predictions": int(sum(1 for pred in y_pred if pred not in true_labels)),
        "label_set_size": len(true_labels),
    }


def _resolve_drvd_image(drvd_root: str, raw_path: object, sample_id: object) -> str:
    """Resolve a DrVD image and fail loudly instead of shrinking the benchmark."""
    rel = str(raw_path or "").strip()
    if not rel:
        raise ValueError(f"DrVD sample {sample_id!r} has no image_paths/image_path field")
    path = Path(rel).expanduser()
    image = path if path.is_absolute() else Path(drvd_root).expanduser() / path
    image = image.resolve()
    if not image.exists():
        raise FileNotFoundError(
            f"DrVD sample {sample_id!r} image does not exist: {image}. "
            "Check --drvd_root and the benchmark image tree."
        )
    return str(image)


def _turn_value(turn: dict[str, Any], *keys: str) -> object:
    for key in keys:
        if key in turn and turn[key] not in (None, ""):
            return turn[key]
    return ""


def _parse_multiturn_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the released DrVD multi-turn schemas to question/options/answer turns."""
    raw_turns = row.get("turns") or row.get("multi_turn") or row.get("conversation")
    turns: list[dict[str, Any]] = []
    if isinstance(raw_turns, list):
        for item in raw_turns:
            if not isinstance(item, dict):
                continue
            question = _turn_value(item, "question", "prompt", "user", "query")
            options = item.get("options", item.get("choices", []))
            answer = _turn_value(item, "answer", "ground_truth", "gt", "label")
            turns.append({"question": str(question).strip(), "options": options, "answer": answer, "level": item.get("level", item.get("task", ""))})
    if not turns and isinstance(row.get("questions", row.get("question")), list):
        questions = row.get("questions", row.get("question", []))
        answers = row.get("answers", row.get("ground_truths", row.get("answer", [])))
        options = row.get("turn_options", row.get("options", row.get("choices", [])))
        if not isinstance(answers, list):
            answers = [answers] * len(questions)
        if not isinstance(options, list):
            options = [options] * len(questions)
        elif options and all(isinstance(item, str) for item in options) and len(options) == len(questions):
            options = [[item] for item in options]
        level_names = ["modality", "body_region", "organ", "lesion", "diagnosis"]
        for index, question in enumerate(questions):
            turns.append({
                "question": str(question).strip(),
                "options": options[index] if index < len(options) else [],
                "answer": answers[index] if index < len(answers) else "",
                "level": level_names[index] if index < len(level_names) else index,
            })
    if not turns:
        # Some mirrors expose one field per clinical level instead of a turns
        # list. Accept both the official five-level schema and older four-level
        # mirrors while preserving the sequential protocol at inference time.
        names = ["modality", "body_region", "organ", "lesion", "diagnosis"]
        for name in names:
            question = _turn_value(row, f"{name}_question", f"{name}_prompt")
            options = row.get(f"{name}_options", [])
            answer = _turn_value(row, f"{name}_answer", f"{name}_gt")
            if question or options or answer:
                turns.append({"question": str(question).strip(), "options": options, "answer": answer, "level": name})
    normalized: list[dict[str, Any]] = []
    level_names = ["modality", "body_region", "organ", "lesion", "diagnosis"]
    for index, item in enumerate(turns):
        options = parse_options(item.get("options", item.get("choices", [])), item)
        answer = parse_answer_letter(str(item.get("answer", "")))
        question = str(item.get("question", "")).strip()
        if not question or not options or not answer:
            raise ValueError(
                "Every DrVD Multi-turn QA turn must contain a question, options, and answer; "
                f"got question={question!r}, options={sorted(options)}, answer={answer!r}"
            )
        if answer not in options:
            raise ValueError(f"Multi-turn answer {answer!r} is not in its option map {sorted(options)}")
        level = str(item.get("level", "")).strip() or (
            level_names[index] if index < len(level_names) else f"turn_{index + 1}"
        )
        normalized.append({"question": question, "options": options, "answer": answer, "level": level})
    if not normalized:
        raise ValueError(
            "Could not find DrVD Multi-turn QA turns. Expected turns/questions lists "
            "or per-level modality/body_region/organ/lesion/diagnosis fields."
        )
    return normalized


def iter_drvd_single(jsonl_path: str, drvd_root: str, max_samples: int, benchmark: str) -> Iterator[Sample]:
    yielded = 0
    seen_ids: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            options = parse_options(row.get("options"), row)
            gt = parse_answer_letter(str(row.get("answer", "")))
            if not options or not gt or gt not in options:
                raise ValueError(
                    f"DrVD {benchmark} line {yielded + 1} has missing options or answer; "
                    "the answer must be one of the declared option labels; refusing to shrink the evaluation denominator"
                )

            rel = _row_image(row)
            q = str(row.get("question", "")).strip()
            sid = _row_id(row, yielded)
            if sid in seen_ids:
                raise ValueError(f"DrVD {benchmark} contains duplicate unique_id {sid!r}")
            seen_ids.add(sid)
            img = _resolve_drvd_image(drvd_root, rel, sid)
            level = _row_level(row, benchmark)
            meta = {
                "task": level,
                "modality": str(row.get("modality", "")).strip(),
                "source_benchmark": benchmark,
            }
            yield Sample(
                sample_id=sid,
                question=q,
                images=[img],
                options=options,
                ground_truth=gt,
                metadata=meta,
                task_type="single",
            )
            yielded += 1
            if max_samples > 0 and yielded >= max_samples:
                break


def iter_drvd_joint(jsonl_path: str, drvd_root: str, max_samples: int) -> Iterator[Sample]:
    yielded = 0
    seen_ids: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rel = _row_image(row)
            sid = _row_id(row, yielded)
            if sid in seen_ids:
                raise ValueError(f"DrVD joint_qa contains duplicate unique_id {sid!r}")
            seen_ids.add(sid)
            img = _resolve_drvd_image(drvd_root, rel, sid)
            level_names = ["modality", "body_region", "organ", "lesion", "diagnosis"]
            level_options = [
                parse_options(row.get(f"{name}_options"), row, f"{name}_")
                for name in level_names
            ]
            # Older DrVD mirrors omit body_region and contain four levels;
            # the official release uses five.  Infer the schema from the
            # explicit body-region fields and retain whichever is present.
            if not level_options[1] and not any(
                key in row for key in ("body_region_options", "body_region_answer")
            ):
                level_names = ["modality", "organ", "lesion", "diagnosis"]
                level_options = [
                    parse_options(row.get(f"{name}_options"), row, f"{name}_")
                    for name in level_names
                ]
            if not all(level_options):
                raise ValueError(
                    f"DrVD joint_qa sample {sid!r} has an incomplete {len(level_names)}-level option set"
                )

            gt_letters = [
                parse_answer_letter(str(row.get(f"{name}_answer", "")))
                for name in level_names
            ]
            if not all(gt_letters) or any(gt not in options for gt, options in zip(gt_letters, level_options)):
                raise ValueError(
                    f"DrVD joint_qa sample {sid!r} has an incomplete or illegal level answer"
                )

            joint_prompt = str(row.get("joint_prompt", "")).strip()
            if not joint_prompt:
                raise ValueError(f"DrVD joint_qa sample {sid!r} has an empty joint_prompt")
            valid_per_step = [set(options.keys()) for options in level_options]

            question_by_level = {
                "modality": "What imaging modality is used in this image?",
                "body_region": "Which body region is shown in this image?",
                "organ": "Which organ appears to be abnormal in this image?",
                "lesion": "Based on the abnormal organ, what lesion or finding is most clearly visible?",
                "diagnosis": "Considering all the above findings, what is the most likely diagnosis?",
            }
            step_names = level_names
            step_questions = [question_by_level[name] for name in level_names]
            step_options = level_options

            yield Sample(
                sample_id=sid,
                question=joint_prompt,
                images=[img],
                options={},
                ground_truth=",".join(gt_letters),
                metadata={
                    "gt_letters": gt_letters,
                    "valid_per_step": [sorted(list(s)) for s in valid_per_step],
                    "step_names": step_names,
                    "step_questions": step_questions,
                    "step_options": step_options,
                    "modality": str(row.get("modality", "")).strip(),
                    "source_benchmark": "drvd_joint_qa",
                },
                task_type="joint",
            )
            yielded += 1
            if max_samples > 0 and yielded >= max_samples:
                break


def iter_drvd_report(jsonl_path: str, drvd_root: str, max_samples: int) -> Iterator[Sample]:
    yielded = 0
    seen_ids: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rel = _row_image(row)
            sid = _row_id(row, yielded)
            if sid in seen_ids:
                raise ValueError(f"DrVD report_generation contains duplicate unique_id {sid!r}")
            seen_ids.add(sid)
            img = _resolve_drvd_image(drvd_root, rel, sid)
            q = str(row.get("question", "Generate a clinical report for this image.")).strip()
            gt = str(row.get("answer", "")).strip()
            if not gt:
                raise ValueError(
                    f"DrVD report-generation sample {sid!r} has an empty answer"
                )

            yield Sample(
                sample_id=sid,
                question=q,
                images=[img],
                options={},
                ground_truth=gt,
                metadata={
                    "task": str(row.get("task", "report_generation")).strip(),
                    "modality": str(row.get("modality", "")).strip(),
                    "source_benchmark": "drvd_report_generation",
                },
                task_type="report",
            )
            yielded += 1
            if max_samples > 0 and yielded >= max_samples:
                break


def iter_drvd_multiturn(jsonl_path: str, drvd_root: str, max_samples: int) -> Iterator[Sample]:
    """Load sequential DrVD Multi-turn QA conversations."""
    yielded = 0
    seen_ids: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Multi-turn JSONL line {line_no} is not an object")
            sid = _row_id(row, yielded)
            if sid in seen_ids:
                raise ValueError(f"DrVD multi_turn_qa contains duplicate unique_id {sid!r}")
            seen_ids.add(sid)
            image = _resolve_drvd_image(drvd_root, _row_image(row), sid)
            turns = _parse_multiturn_turns(row)
            yield Sample(
                sample_id=sid,
                question=turns[0]["question"],
                images=[image],
                options={},
                ground_truth=",".join(turn["answer"] for turn in turns),
                metadata={
                    "turn_names": [str(turn.get("level", index)) for index, turn in enumerate(turns)],
                    "turn_questions": [turn["question"] for turn in turns],
                    "turn_options": [turn["options"] for turn in turns],
                    "gt_letters": [turn["answer"] for turn in turns],
                    "modality": str(row.get("modality", "")).strip(),
                    "task": str(row.get("task", row.get("level", "multi_turn"))).strip(),
                    "source_benchmark": "drvd_multi_turn_qa",
                },
                task_type="multi_turn",
            )
            yielded += 1
            if max_samples > 0 and yielded >= max_samples:
                break


def iter_drvd_multiturn_from_joint(jsonl_path: str, drvd_root: str, max_samples: int) -> Iterator[Sample]:
    """Derive Multi-turn QA from ``joint_qa.jsonl``.

    The official DrVD-Bench release ships four JSONL files.  Multi-turn QA asks
    the hierarchical levels of the joint records one at a time, carrying the
    previous turns in the conversation; the questions, options, and answers
    are therefore those of the joint records.
    """
    for sample in iter_drvd_joint(jsonl_path, drvd_root, max_samples):
        md = sample.metadata
        yield Sample(
            sample_id=sample.sample_id,
            question=str(md["step_questions"][0]),
            images=list(sample.images),
            options={},
            ground_truth=",".join(md["gt_letters"]),
            metadata={
                "turn_names": list(md["step_names"]),
                "turn_questions": list(md["step_questions"]),
                "turn_options": list(md["step_options"]),
                "gt_letters": list(md["gt_letters"]),
                "modality": str(md.get("modality", "")),
                "task": "multi_turn",
                "source_benchmark": "drvd_multi_turn_qa",
                "multi_turn_source": "derived_from_joint_qa",
            },
            task_type="multi_turn",
        )


def load_rgb_images(items: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in items:
        with Image.open(item) as img:
            images.append(img.convert("RGB"))
    return images


def run_chat_generation(
    model: torch.nn.Module,
    processor: Any,
    conversation: list[dict],
    images: list[Image.Image],
    device: str,
    max_new_tokens: int,
) -> str:
    return run_chat_generation_generic(model, processor, conversation, images, device, max_new_tokens)


def run_single_mcq(
    model: torch.nn.Module,
    processor: Any,
    images: list[Image.Image],
    device: str,
    question: str,
    options: dict[str, str],
    task_hint: str,
    style: str,
    max_new_tokens: int,
) -> str:
    option_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

    if style == "two_stage":
        stage1_text = (
            f"{task_hint}Question: {question}\n"
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
                "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": stage1_text}]),
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
            f"{task_hint}Question: {question}\n"
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
                "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": stage2_text}]),
            },
        ]
        return run_chat_generation(
            model=model,
            processor=processor,
            conversation=stage2_conversation,
            images=images,
            device=device,
            max_new_tokens=min(max_new_tokens, 16),
        )

    if style == "cot_brief":
        user_text = (
            f"{task_hint}Question: {question}\n"
            f"Options:\n{option_text}\n"
            "Think briefly (<=2 sentences), then output final letter.\n"
            "Format exactly:\nReason: <short>\nAnswer: <LETTER>"
        )
    else:
        user_text = (
            f"{task_hint}Question: {question}\n"
            f"Options:\n{option_text}\n"
            f"Select the single best option.\n{MCQ_OUTPUT_RULE}"
        )

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": user_text}]),
        },
    ]
    return run_chat_generation(
        model=model,
        processor=processor,
        conversation=conversation,
        images=images,
        device=device,
        max_new_tokens=min(max_new_tokens, 16),
    )


def predict_joint_stepwise(
    model: torch.nn.Module,
    processor: Any,
    sample: Sample,
    device: str,
    max_new_tokens: int,
    step_style: str,
) -> tuple[list[str], list[str]]:
    images = load_rgb_images(sample.images)

    step_names = [str(x) for x in sample.metadata.get("step_names", [])]
    step_questions = [str(x) for x in sample.metadata.get("step_questions", [])]
    step_options_raw = sample.metadata.get("step_options", [])
    step_options: list[dict[str, str]] = []
    for item in step_options_raw:
        if isinstance(item, dict):
            step_options.append({str(k).upper(): str(v) for k, v in item.items()})

    if not (len(step_names) == len(step_questions) == len(step_options) > 0):
        return [], []

    context_lines: list[str] = []
    pred_letters: list[str] = []
    raw_steps: list[str] = []

    for idx in range(len(step_names)):
        options = step_options[idx]
        if not options:
            return [], raw_steps

        task_hint = ""
        if context_lines:
            task_hint = "Previous findings:\n" + "\n".join(context_lines) + "\n\n"

        raw = run_single_mcq(
            model=model,
            processor=processor,
            images=images,
            device=device,
            question=step_questions[idx],
            options=options,
            task_hint=task_hint,
            style=step_style,
            max_new_tokens=max_new_tokens,
        )
        raw_steps.append(raw)

        letter = extract_choice(raw, options.keys())
        if not letter:
            letter = match_option_by_text(raw, options)
        # Keep an empty prediction for an unparseable/illegal answer.  Falling
        # back to the first option silently turns a model failure into a valid
        # prediction and can inflate both step accuracy and joint exact match.
        if letter and letter not in options:
            letter = ""

        pred_letters.append(letter)
        context_lines.append(f"{step_names[idx]}: {letter}. {options.get(letter, '')}")

    return pred_letters, raw_steps


def predict_multi_turn(
    model: torch.nn.Module,
    processor: Any,
    sample: Sample,
    device: str,
    max_new_tokens: int,
    step_style: str,
) -> tuple[list[str], list[str]]:
    """Run Multi-turn QA sequentially while carrying prior turns in context."""
    images = load_rgb_images(sample.images)
    questions = [str(value) for value in sample.metadata.get("turn_questions", [])]
    options_raw = sample.metadata.get("turn_options", [])
    options_list = [
        {str(key).upper(): str(value) for key, value in item.items()}
        if isinstance(item, dict) else {}
        for item in options_raw
    ]
    if len(questions) != len(options_list) or not questions:
        return [], []
    context_lines: list[str] = []
    predictions: list[str] = []
    raw_steps: list[str] = []
    for question, options in zip(questions, options_list):
        if not options:
            predictions.append("")
            raw_steps.append("")
            continue
        task_hint = ""
        if context_lines:
            task_hint = (
                "Conversation history (use it as context, but answer the current question):\n"
                + "\n".join(context_lines)
                + "\n\n"
            )
        raw = run_single_mcq(
            model=model,
            processor=processor,
            images=images,
            device=device,
            question=question,
            options=options,
            task_hint=task_hint,
            style=step_style,
            max_new_tokens=max_new_tokens,
        )
        letter = extract_choice(raw, options.keys()) or match_option_by_text(raw, options)
        if letter not in options:
            letter = ""
        predictions.append(letter)
        raw_steps.append(raw)
        option_text = "; ".join(f"{key}. {value}" for key, value in options.items())
        answer_text = f"{letter}. {options.get(letter, '')}" if letter else "[invalid]"
        context_lines.append(f"Question: {question}\nOptions: {option_text}\nAnswer: {answer_text}")
    return predictions, raw_steps


def predict_one(
    model: torch.nn.Module,
    processor: Any,
    sample: Sample,
    device: str,
    max_new_tokens: int,
    single_style: str,
    joint_style: str,
    joint_step_style: str,
    report_max_new_tokens: int = 512,
) -> dict:
    images = load_rgb_images(sample.images)

    if sample.task_type == "single":
        task_hint = ""
        if isinstance(sample.metadata, dict):
            t = str(sample.metadata.get("task", "")).strip()
            if t:
                task_hint = f"Task: {t}\\n"

        raw = run_single_mcq(
            model=model,
            processor=processor,
            images=images,
            device=device,
            question=sample.question,
            options=sample.options,
            task_hint=task_hint,
            style=single_style,
            max_new_tokens=max_new_tokens,
        )
        return {"raw": raw}

    if sample.task_type == "joint":
        if joint_style == "stepwise":
            pred_letters, raw_steps = predict_joint_stepwise(
                model=model,
                processor=processor,
                sample=sample,
                device=device,
                max_new_tokens=max_new_tokens,
                step_style=joint_step_style,
            )
            return {
                "raw": " | ".join(raw_steps),
                "pred_letters": pred_letters,
                "raw_steps": raw_steps,
            }

        user_text = (
            f"{sample.question}\n\n"
            f"{JOINT_OUTPUT_RULE}"
        )
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
        raw = run_chat_generation(
            model=model,
            processor=processor,
            conversation=conversation,
            images=images,
            device=device,
            max_new_tokens=max(24, min(max_new_tokens, 48)),
        )
        return {"raw": raw}

    if sample.task_type == "multi_turn":
        pred_letters, raw_steps = predict_multi_turn(
            model=model,
            processor=processor,
            sample=sample,
            device=device,
            max_new_tokens=max_new_tokens,
            step_style=joint_step_style,
        )
        return {
            "raw": " | ".join(raw_steps),
            "pred_letters": pred_letters,
            "raw_steps": raw_steps,
        }

    user_text = (
        f"Question: {sample.question}\n"
        "Generate a concise clinical report based only on visible findings in the image."
    )
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
    raw = run_chat_generation(
        model=model,
        processor=processor,
        conversation=conversation,
        images=images,
        device=device,
        max_new_tokens=report_max_new_tokens,
    )
    return {"raw": raw}


def evaluate_samples(
    benchmark: str,
    samples: Iterable[Sample],
    model: torch.nn.Module,
    processor: Any,
    device: str,
    max_new_tokens: int,
    out_jsonl: Path,
    resume: bool,
    single_style: str,
    joint_style: str,
    joint_step_style: str,
    report_metric_mode: str = "official",
    deepseek_api_key: str = "",
    report_bert_score_model: str = REPORT_BERTSCORE_MODEL,
    report_max_new_tokens: int = 512,
) -> dict:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed_ids: set[str] = set()
    total = 0
    correct = 0

    cls_true: list[str] = []
    cls_pred: list[str] = []

    joint_total = 0
    joint_exact = 0
    step_correct: Counter[str] = Counter()
    step_total: Counter[str] = Counter()
    level_correct: Counter[str] = Counter()
    level_total: Counter[str] = Counter()

    def add_level_result(level: object, is_correct: bool) -> None:
        name = str(level or "unknown").strip() or "unknown"
        level_total[name] += 1
        if is_correct:
            level_correct[name] += 1

    mode = "w"
    if resume and out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{out_jsonl} contains malformed JSON in resume data") from exc
                sid = str(row.get("id", ""))
                if not sid:
                    raise ValueError(f"{out_jsonl} contains a prediction without an id")
                if sid in processed_ids:
                    raise ValueError(f"{out_jsonl} contains duplicate prediction id {sid!r}")
                processed_ids.add(sid)

                task_type = str(row.get("task_type", "single"))
                total += 1
                # Report generation has no exact-match accuracy.  Its official
                # metrics are BLEU and BERTScore F1, computed below from the
                # required answer/model_response fields.
                if task_type != "report" and bool(row.get("correct", False)):
                    correct += 1

                if task_type == "single":
                    gt_old = str(row.get("gt", "")).strip().upper()
                    pred_old = str(row.get("pred", "")).strip().upper()
                    if gt_old:
                        cls_true.append(gt_old)
                        cls_pred.append(pred_old if pred_old else "__EMPTY__")
                    add_level_result(row.get("level") or (row.get("metadata") or {}).get("task"), bool(row.get("correct", False)))
                elif task_type == "joint":
                    joint_total += 1
                    gt_letters = row.get("gt_letters", [])
                    pred_letters = row.get("pred_letters", [])
                    names = (row.get("metadata") or {}).get("step_names", [])
                    gt_count = len(gt_letters) if isinstance(gt_letters, list) else 0
                    if not isinstance(names, list) or len(names) != gt_count:
                        default_names = ["modality", "body_region", "organ", "lesion", "diagnosis"] if gt_count == 5 else ["modality", "organ", "lesion", "diagnosis"]
                        names = default_names[:gt_count]
                    if isinstance(gt_letters, list) and names:
                        # A missing, short, or unparsable prediction list counts as
                        # wrong at every level; it never leaves the denominator.
                        preds = pred_letters if isinstance(pred_letters, list) else []
                        hits = [i < len(preds) and preds[i] == gt_letters[i] for i in range(len(gt_letters))]
                        if hits and all(hits):
                            joint_exact += 1
                        for i, name in enumerate(names):
                            step_total[str(name)] += 1
                            if hits[i]:
                                step_correct[str(name)] += 1
                            add_level_result(name, hits[i])
                elif task_type == "multi_turn":
                    gt_letters = row.get("gt_letters", [])
                    pred_letters = row.get("pred_letters", [])
                    names = (row.get("metadata") or {}).get("turn_names", [])
                    if isinstance(gt_letters, list) and isinstance(pred_letters, list):
                        for i, gt_letter in enumerate(gt_letters):
                            add_level_result(names[i] if i < len(names) else i, i < len(pred_letters) and pred_letters[i] == gt_letter)
        mode = "a"
        print(f"[{benchmark}] resume enabled: loaded {len(processed_ids)} existing predictions")

    with out_jsonl.open(mode, encoding="utf-8") as f:
        for sample in tqdm(samples, desc=benchmark):
            sid = str(sample.sample_id)
            if sid in processed_ids:
                continue

            pred_bundle = predict_one(
                model=model,
                processor=processor,
                sample=sample,
                device=device,
                max_new_tokens=max_new_tokens,
                single_style=single_style,
                joint_style=joint_style,
                joint_step_style=joint_step_style,
                report_max_new_tokens=report_max_new_tokens,
            )
            raw_pred = str(pred_bundle.get("raw", ""))

            row = {
                "benchmark": benchmark,
                "id": sid,
                "question": sample.question,
                "modality": str(sample.metadata.get("modality", "Unknown")) if isinstance(sample.metadata, dict) else "Unknown",
                "level": str(sample.metadata.get("task", sample.metadata.get("level", ""))) if isinstance(sample.metadata, dict) else "",
                "pred_raw": raw_pred,
                "gt": sample.ground_truth,
                "metadata": sample.metadata,
                "task_type": sample.task_type,
                "single_style": single_style,
                "joint_style": joint_style,
                "joint_step_style": joint_step_style,
            }

            if sample.task_type == "single":
                gt = sample.ground_truth.upper()
                pred = extract_choice(raw_pred, sample.options.keys())
                if not pred:
                    pred = match_option_by_text(raw_pred, sample.options)
                is_correct = bool(pred) and pred == gt
                row.update({"pred": pred, "correct": is_correct})
                cls_true.append(gt)
                cls_pred.append(pred if pred else "__EMPTY__")
                add_level_result(sample.metadata.get("task", sample.metadata.get("level", "unknown")), is_correct)

            elif sample.task_type == "joint":
                valid_per_step = [set(step) for step in sample.metadata.get("valid_per_step", [])]
                step_names = [str(name) for name in sample.metadata.get("step_names", [])]
                if not step_names or len(valid_per_step) != len(step_names):
                    raise ValueError(f"DrVD joint sample {sample.sample_id!r} has inconsistent level metadata")
                n_steps = len(step_names)

                pred_letters = pred_bundle.get("pred_letters")
                if isinstance(pred_letters, list) and len(pred_letters) == n_steps:
                    # Preserve invalid/empty outputs as invalid predictions.
                    # Do not coerce them to the first option: that makes an
                    # unparsable model response appear correct by chance.
                    normalized_letters: list[str] = []
                    for i, value in enumerate(pred_letters):
                        letter = str(value or "").strip().upper()
                        normalized_letters.append(letter if letter in valid_per_step[i] else "")
                    pred_letters = normalized_letters
                else:
                    pred_letters = extract_joint_letters(raw_pred, valid_per_step)

                gt_letters = list(sample.metadata.get("gt_letters", []))
                is_exact = len(pred_letters) == n_steps and len(gt_letters) == n_steps and all(pred_letters[i] == gt_letters[i] for i in range(n_steps))

                if len(pred_letters) == n_steps and len(gt_letters) == n_steps:
                    step_hits = [pred_letters[i] == gt_letters[i] for i in range(n_steps)]
                else:
                    step_hits = [False] * n_steps

                row.update(
                    {
                        "pred": ",".join(pred_letters) if pred_letters else "",
                        "pred_letters": pred_letters,
                        "gt_letters": gt_letters,
                        "step_hits": step_hits,
                        "correct": is_exact,
                    }
                )
                if "raw_steps" in pred_bundle:
                    row["raw_steps"] = pred_bundle.get("raw_steps")

                joint_total += 1
                if is_exact:
                    joint_exact += 1
                for i in range(n_steps):
                    step_total[step_names[i]] += 1
                    if step_hits[i]:
                        step_correct[step_names[i]] += 1

                is_correct = is_exact

                for index, hit in enumerate(step_hits):
                    add_level_result(step_names[index], hit)

            elif sample.task_type == "multi_turn":
                gt_letters = list(sample.metadata.get("gt_letters", []))
                pred_letters = pred_bundle.get("pred_letters")
                if not isinstance(pred_letters, list):
                    pred_letters = []
                pred_letters = [str(value or "").strip().upper() for value in pred_letters]
                names = list(sample.metadata.get("turn_names", []))
                step_hits = [
                    i < len(pred_letters) and i < len(gt_letters) and pred_letters[i] == str(gt_letters[i]).upper()
                    for i in range(len(gt_letters))
                ]
                is_exact = len(pred_letters) == len(gt_letters) and all(step_hits)
                row.update({
                    "pred": ",".join(pred_letters),
                    "pred_letters": pred_letters,
                    "gt_letters": gt_letters,
                    "step_hits": step_hits,
                    "turn_names": names,
                    "correct": is_exact,
                    "raw_steps": pred_bundle.get("raw_steps", []),
                })
                for index, hit in enumerate(step_hits):
                    add_level_result(names[index] if index < len(names) else index, hit)
                is_correct = is_exact

            else:
                # Keep the exact fields consumed by DrVD-Bench's official
                # report_generation_metric.py.  Do not turn free-form report
                # text into an exact-match accuracy surrogate.
                row.update({
                    "answer": sample.ground_truth,
                    "model_response": raw_pred,
                })
                is_correct = False

            total += 1
            if sample.task_type != "report" and is_correct:
                correct += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if benchmark == "drvd_report_generation":
        # This call validates every row has answer/model_response and computes
        # the official BLEU/BERTScore protocol.  A missing DeepSeek key in
        # official mode fails loudly instead of producing a misleading score.
        result = {
            "benchmark": benchmark,
            "samples": total,
            "output_jsonl": str(out_jsonl),
            **compute_report_generation_metrics(
                out_jsonl,
                deepseek_api_key=deepseek_api_key,
                mode=report_metric_mode,
                bert_score_model=report_bert_score_model,
            ),
        }
        return result

    result = {
        "benchmark": benchmark,
        "samples": total,
        "correct": correct,
        "accuracy": round((correct / total), 6) if total else 0.0,
        "output_jsonl": str(out_jsonl),
        "level_accuracies": {
            level: round(level_correct[level] / level_total[level], 6)
            for level in sorted(level_total)
            if level_total[level]
        },
        "level_counts": {
            level: {"correct": level_correct[level], "total": level_total[level]}
            for level in sorted(level_total)
        },
    }

    if benchmark in {"drvd_visual_evidence_qa", "drvd_independent_qa"} and cls_true and cls_pred and len(cls_true) == len(cls_pred):
        result.update(compute_classification_metrics(cls_true, cls_pred))

    if benchmark == "drvd_joint_qa":
        result.update({
            "joint_exact_accuracy": round((joint_exact / joint_total), 6) if joint_total else 0.0,
            "joint_step_accuracy": round(
                sum(step_correct.values()) / max(1, sum(step_total.values())), 6
            ) if step_total else 0.0,
            "joint_step_accuracies": {
                name: round(step_correct[name] / step_total[name], 6)
                for name in sorted(step_total)
                if step_total[name]
            },
        })
        # Preserve the familiar flat keys while supporting the official
        # five-level schema (which adds body_region).
        for name in sorted(step_total):
            result[f"joint_step_acc_{name}"] = round(
                step_correct[name] / step_total[name], 6
            ) if step_total[name] else 0.0

    if benchmark == "drvd_multi_turn_qa":
        result.update({
            "multi_turn_exact_accuracy": round((correct / total), 6) if total else 0.0,
            "multi_turn_level_average": round(
                sum(result["level_accuracies"].values()) / len(result["level_accuracies"]), 6
            ) if result["level_accuracies"] else 0.0,
        })

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model_path", default="")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "drvd_visual_evidence_qa",
            "drvd_independent_qa",
            "drvd_joint_qa",
            "drvd_multi_turn_qa",
            "drvd_report_generation",
        ],
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--report_max_new_tokens",
        type=int,
        default=512,
        help="Generation budget for DrVD report generation; kept separate from the short MCQ budget.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--single_style", choices=["strict", "cot_brief", "two_stage"], default="strict")
    parser.add_argument("--joint_style", choices=["direct", "stepwise"], default="direct")
    parser.add_argument("--joint_step_style", choices=["strict", "cot_brief", "two_stage"], default="two_stage")
    parser.add_argument(
        "--report_metric_mode",
        choices=["official", "raw"],
        default="official",
        help=(
            "Report-generation metrics. 'official' performs DrVD's DeepSeek "
            "finding extraction and requires DEEPSEEK_API_KEY; 'raw' is an "
            "explicit offline smoke-test mode and is not the paper protocol."
        ),
    )
    parser.add_argument(
        "--deepseek_api_key",
        default=os.environ.get("DEEPSEEK_API_KEY", ""),
        help="DeepSeek key for the official DrVD report-generation metric.",
    )
    parser.add_argument("--report_bert_score_model", default=REPORT_BERTSCORE_MODEL)

    parser.add_argument("--drvd_root", default="${EVAL_ROOT}/DrVD-Bench")
    parser.add_argument("--drvd_visual_jsonl", default="${EVAL_ROOT}/DrVD-Bench/visual_evidence_qa.jsonl")
    parser.add_argument("--drvd_independent_jsonl", default="${EVAL_ROOT}/DrVD-Bench/independent_qa.jsonl")
    parser.add_argument("--drvd_joint_jsonl", default="${EVAL_ROOT}/DrVD-Bench/joint_qa.jsonl")
    parser.add_argument("--drvd_multiturn_jsonl", default="${EVAL_ROOT}/DrVD-Bench/multi_turn_qa.jsonl")
    parser.add_argument(
        "--drvd_multiturn_source",
        choices=["auto", "file", "joint"],
        default="auto",
        help="Multi-turn QA source: a multi-turn JSONL file, the conversations derived from joint_qa.jsonl, "
        "or auto (the file when it exists, otherwise joint_qa.jsonl). The choice is recorded in the summary.",
    )
    parser.add_argument("--drvd_report_jsonl", default="${EVAL_ROOT}/DrVD-Bench/report_generation.jsonl")

    parser.add_argument("--image_max_pixels", type=int, default=262144)
    parser.add_argument("--image_min_pixels", type=int, default=50176)

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

    for benchmark in args.benchmarks:
        multi_turn_source = ""
        if benchmark == "drvd_visual_evidence_qa":
            samples = iter_drvd_single(args.drvd_visual_jsonl, args.drvd_root, args.max_samples, benchmark)
        elif benchmark == "drvd_independent_qa":
            samples = iter_drvd_single(args.drvd_independent_jsonl, args.drvd_root, args.max_samples, benchmark)
        elif benchmark == "drvd_joint_qa":
            samples = iter_drvd_joint(args.drvd_joint_jsonl, args.drvd_root, args.max_samples)
        elif benchmark == "drvd_multi_turn_qa":
            multiturn_path = Path(args.drvd_multiturn_jsonl).expanduser()
            if not multiturn_path.exists():
                alternatives = [
                    multiturn_path.with_name("multi-turn_qa.jsonl"),
                    multiturn_path.with_name("multiturn_qa.jsonl"),
                    multiturn_path.with_name("multi_turn.jsonl"),
                ]
                multiturn_path = next((candidate for candidate in alternatives if candidate.exists()), multiturn_path)
            source = str(args.drvd_multiturn_source)
            if source == "auto":
                source = "file" if multiturn_path.exists() else "joint"
            if source == "file":
                if not multiturn_path.exists():
                    raise FileNotFoundError(
                        f"DrVD Multi-turn QA file not found: {args.drvd_multiturn_jsonl}. "
                        "Provide --drvd_multiturn_jsonl, or use --drvd_multiturn_source joint."
                    )
                samples = iter_drvd_multiturn(str(multiturn_path), args.drvd_root, args.max_samples)
                multi_turn_source = f"file:{multiturn_path.name}"
            else:
                print("[drvd_multi_turn_qa] deriving the conversations from joint_qa.jsonl (official release has no multi-turn file)")
                samples = iter_drvd_multiturn_from_joint(args.drvd_joint_jsonl, args.drvd_root, args.max_samples)
                multi_turn_source = "derived_from_joint_qa"
        elif benchmark == "drvd_report_generation":
            samples = iter_drvd_report(args.drvd_report_jsonl, args.drvd_root, args.max_samples)
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        out_jsonl = output_dir / f"{benchmark}_predictions.jsonl"
        row = (
            evaluate_samples(
                benchmark=benchmark,
                samples=samples,
                model=model,
                processor=processor,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                out_jsonl=out_jsonl,
                resume=bool(args.resume),
                single_style=str(args.single_style),
                joint_style=str(args.joint_style),
                joint_step_style=str(args.joint_step_style),
                report_metric_mode=str(args.report_metric_mode),
                deepseek_api_key=str(args.deepseek_api_key),
                report_bert_score_model=str(args.report_bert_score_model),
                report_max_new_tokens=int(args.report_max_new_tokens),
            )
        )
        if benchmark == "drvd_multi_turn_qa":
            row["multi_turn_source"] = multi_turn_source
        summary.append(row)

    summary_path = output_dir / "drvd_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
