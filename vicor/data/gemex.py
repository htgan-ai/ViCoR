from __future__ import annotations
import os
import json
import signal
import re
from typing import Dict, Any, List
import warnings
from torch.utils.data import Dataset
from PIL import Image


_ANSWER_MARKER_RE = re.compile(r"(?im)^[ \t]*(?:\*\*)?[ \t]*(?:answer|final answer|conclusion)[ \t]*(?:\*\*)?[ \t]*:[ \t]*")


def clean_reasoning_text(reasoning: str, answer: str) -> tuple[str, bool]:
    """Remove a duplicated final-answer section from reasoning when exact.

    Some GEMeX rows include ``**Answer:** <answer>`` inside ``reasoning`` and
    also provide the same value in the separate ``answer`` field.  We remove
    only an answer-marker suffix whose normalized content exactly matches the
    answer; unrelated mentions inside a rationale are preserved.
    """
    text = str(reasoning or "").strip()
    target = str(answer or "").strip()
    if not text or not target:
        return text, False
    matches = list(_ANSWER_MARKER_RE.finditer(text))
    for match in reversed(matches):
        suffix = text[match.end() :].strip()
        if suffix and " ".join(suffix.split()).casefold() == " ".join(target.split()).casefold():
            cleaned = text[: match.start()].rstrip()
            return cleaned, cleaned != text
    return text, False


class GEMeXDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        jsonl_name: str = "data.jsonl",
        use_cot: bool = True,
        strict_image_paths: bool = False,
        image_roots: List[str] | None = None,
        skip_image_exists_scan: bool = True,
        force_mean_image: bool = False,
        image_open_timeout_sec: float = 0.0,
        clean_reasoning: bool = False,
    ):
        """
        GEMeX dataset loader.
        Expects:
          data_root/
            data.jsonl
            images/
        Each line in data.jsonl contains: image_path, question, reasoning, answer
        """
        self.data_root = data_root
        self.strict_image_paths = strict_image_paths
        self.image_roots = [p for p in (image_roots or []) if isinstance(p, str) and p.strip()]
        self.skip_image_exists_scan = bool(skip_image_exists_scan)
        self.force_mean_image = bool(force_mean_image)
        self.image_open_timeout_sec = float(image_open_timeout_sec)
        self.clean_reasoning = bool(clean_reasoning)
        if self.image_open_timeout_sec < 0:
            raise ValueError(f"image_open_timeout_sec must be >= 0, got {self.image_open_timeout_sec}")
        jsonl_path = os.path.join(data_root, jsonl_name)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"GEMeX jsonl not found: {jsonl_path}")

        self.data: List[Dict[str, Any]] = []
        missing_count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["id"] = str(idx)
                resolved_path = "" if self.force_mean_image else self._resolve_image_path(item)
                item["_resolved_image_path"] = resolved_path
                if self.force_mean_image:
                    item["_image_exists"] = False
                elif self.skip_image_exists_scan and not self.strict_image_paths:
                    # Fast-path mode for large remote datasets:
                    # skip expensive per-sample os.path.exists scan during dataset init.
                    # Real image readability is still validated lazily in __getitem__.
                    item["_image_exists"] = True
                else:
                    item["_image_exists"] = bool(resolved_path) and os.path.exists(resolved_path)
                    if not item["_image_exists"]:
                        missing_count += 1
                if use_cot:
                    reasoning = str(item.get("reasoning", ""))
                    answer = item.get("answer", "")
                    item["raw_reasoning"] = reasoning
                    if self.clean_reasoning:
                        reasoning, was_cleaned = clean_reasoning_text(reasoning, str(answer))
                        item["reasoning"] = reasoning
                        item["reasoning_cleaned"] = bool(was_cleaned)
                    if reasoning:
                        item["answer_cot"] = f"<think>{reasoning}</think>{answer}"
                self.data.append(item)

        if self.data and not (self.skip_image_exists_scan and not self.strict_image_paths):
            ratio = missing_count / len(self.data)
            if missing_count > 0:
                msg = (
                    f"{missing_count}/{len(self.data)} samples have missing/unreadable image paths "
                    f"in {jsonl_path} (rate={ratio:.2%}). "
                    "Scoring with mean-image fallback images can change visual-necessity estimates."
                )
                if self.strict_image_paths:
                    raise FileNotFoundError(msg)
                warnings.warn(msg)

    def _resolve_image_path(self, item: Dict[str, Any]) -> str:
        """Resolve image path across schema variants: image_path / image / img."""
        raw_path = str(item.get("image_path") or item.get("image") or item.get("img") or "").strip()
        if not raw_path:
            return ""

        candidates: List[str] = []
        if os.path.isabs(raw_path):
            candidates.append(raw_path)
        else:
            candidates.append(os.path.join(self.data_root, raw_path))
            candidates.append(os.path.join(self.data_root, "images", raw_path))
            for root in self.image_roots:
                candidates.append(os.path.join(root, raw_path))
                candidates.append(os.path.join(root, os.path.basename(raw_path)))

        if self.skip_image_exists_scan and not self.strict_image_paths:
            if os.path.isabs(raw_path):
                return raw_path
            # For bare filenames, prefer data_root/images/<name> as default path.
            if "/" not in raw_path and len(candidates) >= 2:
                return candidates[1]
            return candidates[0] if candidates else ""

        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0] if candidates else ""

    @staticmethod
    def _mean_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
        return Image.new("RGB", size, (123, 117, 104))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = item.get("_resolved_image_path", "")

        if self.force_mean_image:
            image = self._mean_image()
        else:
            try:
                timeout_sec = self.image_open_timeout_sec
                if timeout_sec > 0 and hasattr(signal, "SIGALRM"):
                    def _timeout_handler(signum, frame):
                        raise TimeoutError(f"Image open timed out after {timeout_sec}s: {image_path}")

                    old_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
                    try:
                        image = Image.open(image_path).convert("RGB")
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    image = Image.open(image_path).convert("RGB")
            except Exception:
                if self.strict_image_paths:
                    raise FileNotFoundError(f"Image not readable for sample id={item.get('id', idx)} path={image_path}")
                image = self._mean_image()

        return {
            "id": item.get("id", str(idx)),
            "image": image,
            "image_path": image_path,
            "question": item.get("question", ""),
            "reasoning": item.get("reasoning", ""),
            "answer": str(item.get("answer", "")),
            "answer_cot": str(item.get("answer_cot", item.get("answer", ""))),
            "type": "OPEN",
        }
