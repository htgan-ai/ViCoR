#!/usr/bin/env python3
"""Prepare the fixed training pools used by the ViCoR experiments.

The paper's data preparation pass is deliberately kept separate from scoring:
this command removes malformed/template rows, removes duplicate image-question
pairs, filters fill-in-the-blank examples, assigns CXR task types, and writes
the fixed-size pools consumed by the public configs.  The raw datasets and the
Qwen3-4B checkpoint are not distributed.  Pass ``--denoiser_model`` and
``--classifier_model`` to run the model-based filtering and CXR task
classification passes; without them, existing labels and deterministic
filters are used for prepared-input checks and audits.

Examples
--------
    # CXR-CounterFact: produce both 40k task-balanced sources
    python scripts/00_preprocess_training_pools.py \
        --dataset cxr --input /data/cxr/raw.jsonl \
        --output_dir /data/CXR-CounterFact --pool_size 40000 --seed 17

    # GEMeX: produce the 60k training pool and 1k geometry reserve
    python scripts/00_preprocess_training_pools.py \
        --dataset gemex --input /data/gemex/raw.jsonl \
        --output_dir /data/GEMeX --pool_size 60000 --reserve_size 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable


LOG = logging.getLogger("vicor.preprocess")
PREPROCESSOR_VERSION = "vicor_appendix_b_qwen3_4b_v2"
_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_PLACEHOLDER_ANSWERS = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "cannot answer",
    "no answer",
    "not available",
}
_FILL_RE = re.compile(
    r"(?:\bfill[\s_-]*in[\s_-]*(?:the[\s_-]*)?blank(?:s)?\b|"
    r"\bfill[\s_-]*the[\s_-]*blank(?:s)?\b|complete the sentence|"
    r"_{2,}|\[\s*blank\s*\])",
    re.IGNORECASE,
)
_NORMAL_RE = re.compile(
    r"(?:no (?:acute )?(?:finding|abnormalit|disease|cardiopulmonary)|"
    r"normal (?:study|chest|radiograph|x[- ]?ray)|negative for|clear lungs?|"
    r"without (?:acute )?(?:finding|abnormalit|disease))",
    re.IGNORECASE,
)
_DISEASE_TERMS = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "effusion",
    "fracture",
    "lesion",
    "opacity",
    "pneumonia",
    "pneumothorax",
    "nodule",
    "support device",
    "enlarged cardiomediastinum",
    "lung disease",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("cxr", "gemex", "schain", "generic"))
    p.add_argument("--input", required=True, help="Raw JSONL file")
    p.add_argument("--output_dir", required=True, help="Directory receiving prepared JSONL files")
    p.add_argument("--pool_size", type=int, default=None, help="Target pool size (CXR/GEMeX/S-Chain defaults: 40000/60000/10000)")
    p.add_argument("--reserve_size", type=int, default=1000, help="GEMeX geometry-valid reserve size")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--image_root", action="append", default=[], help="Additional image root; may be repeated")
    p.add_argument("--require_image", action="store_true", help="Drop rows whose image cannot be resolved")
    p.add_argument("--denoiser_model", default="", help="Optional local Qwen3-4B checkpoint for model-based keep/reject filtering")
    p.add_argument("--denoiser_device", default="auto", help="Denoiser device, e.g. cuda:0, cpu, or auto")
    p.add_argument("--classifier_model", default="", help="Optional local Qwen3-4B/Qwen3-VL checkpoint for CXR task classification")
    p.add_argument("--classifier_device", default="auto", help="Classifier device, e.g. cuda:0, cpu, or auto")
    p.add_argument("--strict", action="store_true", help="Fail on malformed JSON instead of recording and skipping it")
    p.add_argument("--type_field", default="task_type", help="Existing task-label field")
    p.add_argument("--image_field", default="", help="Image field override; defaults to image_path/image/img")
    return p.parse_args()


def norm_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip()).casefold()


def canonical_task(value: Any) -> str:
    value = norm_text(value).replace("_", "-")
    aliases = {
        "single": "single",
        "single-disease": "single",
        "single disease": "single",
        "multi": "multi",
        "multiple": "multi",
        "multi-disease": "multi",
        "multiple-disease": "multi",
        "multiple diseases": "multi",
        "normal": "normal",
        "negative": "normal",
        "no-finding": "normal",
        "no finding": "normal",
        "fill-blank": "fill_blank",
        "fill in the blank": "fill_blank",
        "fill-in-the-blank": "fill_blank",
        "other": "other",
        "open": "other",
        "open-ended": "other",
    }
    return aliases.get(value, "")


def is_fill_blank(row: dict[str, Any]) -> bool:
    # JSONL exports frequently serialize booleans as strings.  In particular,
    # ``bool("false")`` is True, which would silently discard valid examples.
    explicit = row.get("is_fill_blank", False)
    if isinstance(explicit, str):
        explicit = explicit.strip().casefold()
        explicit = explicit in {"1", "true", "yes", "y", "on"}
    elif isinstance(explicit, (int, float)):
        explicit = bool(explicit)
    else:
        explicit = bool(explicit)
    if explicit:
        return True
    text = " ".join(str(row.get(k, "")) for k in ("question", "answer", "reasoning", "task_type", "task"))
    return bool(_FILL_RE.search(text))


def image_value(row: dict[str, Any], image_field: str = "") -> str:
    if image_field:
        return str(row.get(image_field, "") or "").strip()
    return str(row.get("image_path") or row.get("image") or row.get("img") or "").strip()


def resolve_image(raw: str, input_path: Path, image_roots: Iterable[str]) -> str:
    if not raw:
        return ""
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [input_path.parent / path, input_path.parent / "images" / path]
    for root in image_roots:
        root_path = Path(root).expanduser()
        candidates.extend((root_path / path, root_path / path.name))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve()) if candidates else ""


def heuristic_task(row: dict[str, Any], type_field: str = "task_type") -> str:
    for key in (type_field, "task_type", "task", "category", "question_type"):
        task = canonical_task(row.get(key))
        if task:
            return task
    text = " ".join(str(row.get(k, "")) for k in ("question", "answer", "reasoning", "labels", "findings"))
    if is_fill_blank(row):
        return "fill_blank"
    if _NORMAL_RE.search(text):
        return "normal"
    lowered = text.casefold()
    hits = sum(term in lowered for term in _DISEASE_TERMS)
    if hits >= 2 or any(sep in lowered for sep in (",", ";", " and ", " plus ")):
        return "multi"
    return "single"


class QwenTaskClassifier:
    """Optional Qwen3-4B classifier used by the Appendix-B preprocessing pass."""

    def __init__(self, model_path: str, device: str) -> None:
        try:
            import torch
            import transformers
            AutoConfig = getattr(transformers, "AutoConfig")
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("--classifier_model requires torch and transformers") from exc
        self.torch = torch
        # Qwen3 (text) and Qwen3-VL have different AutoModel mappings.  Calling
        # AutoModelForImageTextToText for a text-only Qwen3Config raises before
        # any useful error message on recent Transformers releases, so inspect
        # the checkpoint configuration first.
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = str(getattr(config, "model_type", "")).casefold()
        architectures = " ".join(str(v) for v in (getattr(config, "architectures", None) or [])).casefold()
        self.text_only = not (
            "vl" in model_type
            or "vision" in model_type
            or "visual" in model_type
            or "vl" in architectures
            or "vision" in architectures
        )
        if self.text_only:
            self.processor = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model_loader = getattr(transformers, "AutoModelForCausalLM")
        else:
            auto_processor = getattr(transformers, "AutoProcessor", None)
            if auto_processor is None:
                raise RuntimeError("A Qwen3-VL checkpoint requires transformers.AutoProcessor")
            self.processor = auto_processor.from_pretrained(model_path, trust_remote_code=True)
            model_loader = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_loader is None:
                model_loader = getattr(transformers, "AutoModelForVision2Seq", None)
            if model_loader is None:
                raise RuntimeError("This Transformers version has no vision-language AutoModel loader")
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = model_loader.from_pretrained(model_path, **kwargs)
        if device != "auto":
            self.model.to(device)
        self.device = device
        self.revision = str(getattr(getattr(self.model, "config", None), "_commit_hash", "") or "")

    def __call__(self, row: dict[str, Any], image_path: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB") if image_path and Path(image_path).exists() else None
        prompt = (
            "Classify this medical VQA example. Return exactly one label: "
            "single, multi, normal, or fill_blank. Use multi only when the answer "
            "contains two or more independent findings.\nQuestion: "
            + str(row.get("question", ""))
            + "\nAnswer: "
            + str(row.get("answer", ""))
        )
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if self.text_only:
            text = prompt
            inputs = self.processor(text, return_tensors="pt")
        else:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
            if image is not None:
                kwargs["images"] = [image]
            inputs = self.processor(**kwargs)
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        prompt_len = inputs["input_ids"].shape[-1]
        tokenizer = self.processor if self.text_only else self.processor.tokenizer
        decoded = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        token = re.search(r"(?:single|multi|normal|fill[ _-]?blank)", decoded.casefold())
        task = canonical_task(token.group(0) if token else decoded)
        return task or heuristic_task(row)


class QwenDenoiser:
    """Run the fixed keep/reject prompt used for model-based row filtering."""

    def __init__(self, model_path: str, device: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("--denoiser_model requires torch and transformers") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if device != "auto":
            self.model.to(device)
        self.revision = str(getattr(getattr(self.model, "config", None), "_commit_hash", "") or "")

    def __call__(self, row: dict[str, Any]) -> bool:
        prompt = (
            "You are a strict medical VQA data-quality filter. Return exactly one token, "
            "KEEP or REJECT. KEEP only if the example has a well-formed question and answer, "
            "a clinically coherent response, and no template/fill-in-the-blank artifact. "
            "REJECT malformed, empty, contradictory, or weakly formatted examples.\n"
            "Question: " + str(row.get("question", "")) + "\nAnswer: " + str(row.get("answer", ""))
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=3, do_sample=False)
        prompt_len = inputs["input_ids"].shape[-1]
        decoded = self.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).casefold()
        match = re.search(r"\b(keep|reject)\b", decoded)
        if not match:
            raise ValueError(f"Qwen denoiser returned no KEEP/REJECT decision: {decoded!r}")
        return match.group(1) == "keep"


def load_rows(path: Path, strict: bool) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    audit: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                if strict:
                    raise
                audit.append({"source_line": line_no, "status": "malformed_json", "detail": str(exc)})
                continue
            if not isinstance(row, dict):
                audit.append({"source_line": line_no, "status": "non_object"})
                continue
            rows.append((line_no, row))
    return rows, audit


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boxes(row: dict[str, Any]) -> list[list[float]]:
    import ast

    result: list[list[float]] = []
    raw_boxes = row.get("grounding_boxes", []) or []
    # GEMeX exports may serialize the whole list as JSON/Python text, e.g.
    # ``"[[1, 2, 30, 40]]"``.  Decode that container before normalizing the
    # bare-box and per-box forms below.
    if isinstance(raw_boxes, str):
        try:
            raw_boxes = json.loads(raw_boxes)
        except json.JSONDecodeError:
            try:
                raw_boxes = ast.literal_eval(raw_boxes)
            except (SyntaxError, ValueError):
                return result
    # Some exports contain one bare [x1, y1, x2, y2] instead of a list of
    # boxes.  Normalize both forms before parsing individual entries.
    if isinstance(raw_boxes, (list, tuple)) and len(raw_boxes) == 4 and all(
        isinstance(value, (int, float)) for value in raw_boxes
    ):
        raw_boxes = [raw_boxes]
    elif isinstance(raw_boxes, dict):
        raw_boxes = raw_boxes.get("boxes", raw_boxes.get("bbox", raw_boxes.get("box", [])))
        if isinstance(raw_boxes, str):
            try:
                raw_boxes = json.loads(raw_boxes)
            except json.JSONDecodeError:
                try:
                    raw_boxes = ast.literal_eval(raw_boxes)
                except (SyntaxError, ValueError):
                    return result
        if isinstance(raw_boxes, (list, tuple)) and len(raw_boxes) == 4 and all(
            isinstance(value, (int, float)) for value in raw_boxes
        ):
            raw_boxes = [raw_boxes]
    if not isinstance(raw_boxes, (list, tuple)):
        return result
    for raw in raw_boxes:
        try:
            value = raw if isinstance(raw, (list, tuple)) else ast.literal_eval(str(raw))
            if isinstance(value, (list, tuple)) and len(value) == 4:
                coords = [float(v) for v in value]
                if coords[2] > coords[0] and coords[3] > coords[1] and all(0 <= v <= 336 for v in coords):
                    result.append(coords)
        except (SyntaxError, ValueError, TypeError):
            continue
    return result


def sample_without_replacement(rows: list[dict[str, Any]], size: int, rng: random.Random, label: str) -> list[dict[str, Any]]:
    if len(rows) < size:
        raise ValueError(f"{label} requires {size} rows, found {len(rows)} after preprocessing")
    return rng.sample(rows, size)


def main() -> int:
    args = parse_args()
    if args.pool_size is None:
        args.pool_size = {"cxr": 40000, "gemex": 60000, "schain": 10000, "generic": 0}[args.dataset]
    if args.pool_size < 0 or args.reserve_size < 0:
        raise ValueError("pool_size and reserve_size must be non-negative")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, audit = load_rows(input_path, args.strict)
    denoiser = QwenDenoiser(args.denoiser_model, args.denoiser_device) if args.denoiser_model else None
    classifier = QwenTaskClassifier(args.classifier_model, args.classifier_device) if args.classifier_model and args.dataset == "cxr" else None
    seen: dict[str, int] = {}
    pair_seen: dict[str, dict[str, Any]] = {}
    conflicting_pair_keys: set[str] = set()
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for source_line, row in raw_rows:
        if denoiser is not None:
            if not denoiser(row):
                audit.append({"source_line": source_line, "status": "qwen_denoiser_reject"})
                continue
        question = norm_text(row.get("question"))
        answer = norm_text(row.get("answer"))
        if not question or answer in _PLACEHOLDER_ANSWERS:
            audit.append({"source_line": source_line, "status": "empty_or_placeholder"})
            continue
        if is_fill_blank(row):
            audit.append({"source_line": source_line, "status": "fill_blank"})
            continue
        raw_image = image_value(row, args.image_field)
        resolved_image = resolve_image(raw_image, input_path, args.image_root)
        if args.require_image and (not resolved_image or not Path(resolved_image).exists()):
            audit.append({"source_line": source_line, "status": "missing_image", "image": raw_image})
            continue
        image_identity = norm_text(resolved_image or raw_image)
        pair_payload = "\n".join((image_identity, question))
        # Without an image identifier we cannot establish that two records
        # refer to the same image.  Keep ordinary answer-level deduplication,
        # but do not falsely label all image-less rows with the same question
        # as an image/question conflict.
        pair_key = hashlib.sha1(pair_payload.encode("utf-8")).hexdigest() if image_identity else ""
        previous_pair = pair_seen.get(pair_key) if pair_key else None
        if pair_key and previous_pair is not None and previous_pair["answer"] != answer:
            # Conflicting labels for the same image-question pair are more
            # dangerous than ordinary duplicate rows.  Exclude both rows from
            # the prepared pools and keep an explicit audit trail.
            previous_pair["conflicting"] = True
            conflicting_pair_keys.add(pair_key)
            previous_audit = audit[previous_pair["audit_index"]]
            previous_audit["status"] = "conflicting_answer_excluded"
            previous_audit["conflicting_source_line"] = source_line
            previous_audit["conflicting_answer"] = answer
            audit.append({
                "source_line": source_line,
                "status": "conflicting_answer_excluded",
                "conflicts_with_source_line": previous_pair["source_line"],
                "conflicting_answer": previous_pair["answer"],
                "dedup_pair_key": pair_key,
            })
            continue
        key_payload = "\n".join((norm_text(resolved_image or raw_image), question, answer))
        key = hashlib.sha1(key_payload.encode("utf-8")).hexdigest()
        if key in seen:
            audit.append({
                "source_line": source_line,
                "status": "duplicate",
                "kept_source_line": seen[key],
                "dedup_key": key,
                "dedup_pair_key": pair_key,
            })
            continue
        seen[key] = source_line
        cleaned = dict(row)
        cleaned["preprocess_source_line"] = source_line
        cleaned["preprocess_id"] = hashlib.sha1(f"{input_path}:{source_line}".encode()).hexdigest()[:16]
        if resolved_image:
            cleaned["image_path"] = resolved_image
        try:
            task = classifier(cleaned, resolved_image) if classifier else heuristic_task(cleaned, args.type_field)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            if classifier is not None:
                raise RuntimeError(
                    f"Qwen task classification failed on source line {source_line}; "
                    "fix the checkpoint/runtime instead of silently changing the Appendix-B preprocessing"
                ) from exc
            LOG.warning("Heuristic task classification failed on source line %s: %s", source_line, exc)
            task = "other"
        cleaned["task_type"] = task
        cleaned["is_fill_blank"] = False
        cleaned["_dedup_pair_key"] = pair_key
        kept.append(cleaned)
        counts[task] = counts.get(task, 0) + 1
        audit.append({"source_line": source_line, "status": "kept", "task_type": task})
        audit[-1].update({"dedup_key": key, "preprocess_id": cleaned["preprocess_id"]})
        if pair_key:
            pair_seen[pair_key] = {
                "source_line": source_line,
                "answer": answer,
                "audit_index": len(audit) - 1,
                "conflicting": False,
            }

    if conflicting_pair_keys:
        conflicting_lines = {
            int(row["preprocess_source_line"])
            for row in kept
            if row.get("_dedup_pair_key") in conflicting_pair_keys
        }
        for item in audit:
            if (
                item.get("source_line") in conflicting_lines
                or item.get("dedup_pair_key") in conflicting_pair_keys
            ) and item.get("status") in {"kept", "duplicate"}:
                item["status"] = "conflicting_answer_excluded"
        kept = [row for row in kept if row.get("_dedup_pair_key") not in conflicting_pair_keys]
    for row in kept:
        row.pop("_dedup_pair_key", None)
    counts = {}
    for row in kept:
        task = str(row.get("task_type", "other"))
        counts[task] = counts.get(task, 0) + 1

    rng = random.Random(args.seed)
    if args.dataset == "cxr":
        usable = [r for r in kept if r.get("task_type") in {"single", "multi", "normal"}]
        single = [r for r in usable if r["task_type"] == "single"]
        multi = [r for r in usable if r["task_type"] == "multi"]
        normal = [r for r in usable if r["task_type"] == "normal"]
        def balanced(disease: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
            if not disease:
                raise ValueError(f"{name}: no disease rows remain after preprocessing")
            # Appendix B constructs each fixed 40k source from disease rows
            # plus randomly sampled normal rows.  In the usual case all
            # retained disease rows are kept and normals fill the remainder.
            # If filtering leaves more disease rows than fit in the fixed
            # target, downsample both groups while preserving the observed
            # disease/normal mixture; never silently emit a disease-only pool.
            if len(disease) >= args.pool_size:
                if not normal:
                    raise ValueError(f"{name}: disease rows fill the target but no normal rows are available")
                normal_fraction = len(normal) / float(len(normal) + len(disease))
                normal_count = max(1, min(len(normal), int(round(args.pool_size * normal_fraction))))
                disease_count = args.pool_size - normal_count
                chosen = sample_without_replacement(disease, disease_count, rng, f"{name} disease rows")
                chosen += sample_without_replacement(normal, normal_count, rng, f"{name} normal rows")
            else:
                normal_count = args.pool_size - len(disease)
                chosen = list(disease) + sample_without_replacement(normal, normal_count, rng, f"{name} normal rows")
            rng.shuffle(chosen)
            return chosen
        single_pool = balanced(single, "single")
        multi_pool = balanced(multi, "multi")
        write_jsonl(output_dir / "single_disease.jsonl", single_pool)
        write_jsonl(output_dir / "multiple_diseases.jsonl", multi_pool)
        write_jsonl(output_dir / "cxr_preprocessed.jsonl", usable)
        outputs = {"single_disease": len(single_pool), "multiple_diseases": len(multi_pool), "preprocessed": len(usable)}
    elif args.dataset == "gemex":
        # Appendix B reserves geometry-valid examples first.  The training
        # pool is sampled only from the remaining rows, which makes the split
        # order explicit and prevents reserve membership from depending on a
        # later overlap repair.
        reserve_candidates = [r for r in kept if boxes(r)]
        reserve = sample_without_replacement(reserve_candidates, args.reserve_size, rng, "GEMeX geometry reserve")
        reserve_ids = {r.get("preprocess_id") for r in reserve}
        remaining = [r for r in kept if r.get("preprocess_id") not in reserve_ids]
        clean_pool = sample_without_replacement(remaining, args.pool_size, rng, "GEMeX training pool after reserve")
        write_jsonl(output_dir / "gemex_train_60k.jsonl", clean_pool)
        write_jsonl(output_dir / "gemex_grounding_reserve_1k.jsonl", reserve)
        write_jsonl(output_dir / "gemex_preprocessed.jsonl", kept)
        outputs = {"gemex_train_60k": len(clean_pool), "gemex_grounding_reserve_1k": len(reserve), "preprocessed": len(kept)}
    elif args.dataset == "schain":
        pool = sample_without_replacement(kept, args.pool_size, rng, "S-Chain training pool")
        write_jsonl(output_dir / "schain_train_10k.jsonl", pool)
        write_jsonl(output_dir / "schain_preprocessed.jsonl", kept)
        outputs = {"schain_train_10k": len(pool), "preprocessed": len(kept)}
    else:
        write_jsonl(output_dir / "preprocessed.jsonl", kept)
        outputs = {"preprocessed": len(kept)}

    write_jsonl(output_dir / "preprocess_audit.jsonl", audit)
    manifest = {
        "preprocessor_version": PREPROCESSOR_VERSION,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_dir": str(output_dir),
        "dataset": args.dataset,
        "seed": args.seed,
        "classifier": "qwen_checkpoint" if args.classifier_model else "existing_labels_or_deterministic_heuristic",
        "classifier_model": str(Path(args.classifier_model).expanduser().resolve()) if args.classifier_model else "",
        "classifier_revision": getattr(classifier, "revision", "") if classifier else "",
        "denoiser": "qwen_checkpoint" if args.denoiser_model else "deterministic_filters_only",
        "denoiser_model": str(Path(args.denoiser_model).expanduser().resolve()) if args.denoiser_model else "",
        "denoiser_revision": getattr(denoiser, "revision", "") if denoiser else "",
        "raw_rows": len(raw_rows),
        "kept_rows": len(kept),
        "task_counts": counts,
        "audit_counts": {status: sum(item.get("status") == status for item in audit) for status in sorted({item.get("status") for item in audit})},
        "outputs": outputs,
        "pool_size": args.pool_size,
        "reserve_size": args.reserve_size if args.dataset == "gemex" else 0,
        "image_roots": [str(Path(p).expanduser().resolve()) for p in args.image_root],
    }
    output_paths = [output_dir / "preprocess_audit.jsonl"]
    if args.dataset == "cxr":
        output_paths.extend(output_dir / name for name in ("single_disease.jsonl", "multiple_diseases.jsonl", "cxr_preprocessed.jsonl"))
    elif args.dataset == "gemex":
        output_paths.extend(output_dir / name for name in ("gemex_train_60k.jsonl", "gemex_grounding_reserve_1k.jsonl", "gemex_preprocessed.jsonl"))
    elif args.dataset == "schain":
        output_paths.extend(output_dir / name for name in ("schain_train_10k.jsonl", "schain_preprocessed.jsonl"))
    else:
        output_paths.append(output_dir / "preprocessed.jsonl")
    manifest["output_sha256"] = {path.name: sha256_file(path) for path in output_paths if path.exists()}
    (output_dir / "preprocess_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
