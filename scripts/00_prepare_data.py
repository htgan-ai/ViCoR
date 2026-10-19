import argparse
import os
import json
import sys
import logging
from typing import List, Tuple


def setup_logger(name: str = "prepare_data", level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Check and prepare dataset structure")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gemex", "s-chain", "schain", "cxr-counterfact", "custom_jsonl"],
        help="Dataset name",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--jsonl_name", type=str, default="data.jsonl", help="JSONL file name for jsonl-based datasets")
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=[],
        help="Optional extra roots for resolving relative image paths",
    )
    parser.add_argument(
        "--min_image_exists_ratio",
        type=float,
        default=None,
        help="Optional minimum image-availability ratio in [0,1]",
    )
    parser.add_argument(
        "--min_image_exists_count",
        type=int,
        default=None,
        help="Optional minimum number of existing images",
    )
    return parser.parse_args()

def _resolve_image_path(data_root: str, item: dict, image_roots: List[str]) -> str:
    raw_path = str(item.get("image_path") or item.get("image") or item.get("img") or "").strip()
    if not raw_path:
        return ""

    candidates: List[str] = []
    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        candidates.append(os.path.join(data_root, raw_path))
        candidates.append(os.path.join(data_root, "images", raw_path))
        for root in image_roots:
            root = str(root).strip()
            if root:
                candidates.append(os.path.join(root, raw_path))
                candidates.append(os.path.join(root, os.path.basename(raw_path)))

    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0] if candidates else ""


def check_jsonl_images(
    data_root: str,
    jsonl_name: str,
    image_roots: List[str],
    logger,
    min_image_exists_ratio: float | None = None,
    min_image_exists_count: int | None = None,
) -> bool:
    jsonl_path = os.path.join(data_root, jsonl_name)
    if not os.path.exists(jsonl_path):
        logger.error(f"Missing file: {jsonl_path}")
        return False

    if min_image_exists_ratio is not None and not (0.0 <= float(min_image_exists_ratio) <= 1.0):
        logger.error(f"Invalid min_image_exists_ratio={min_image_exists_ratio}, should be in [0,1]")
        return False
    if min_image_exists_count is not None and int(min_image_exists_count) < 0:
        logger.error(f"Invalid min_image_exists_count={min_image_exists_count}, should be >= 0")
        return False

    total = 0
    exists = 0
    missing_examples: List[Tuple[int, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            resolved = _resolve_image_path(data_root, item, image_roots)
            ok = bool(resolved) and os.path.exists(resolved)
            if ok:
                exists += 1
            elif len(missing_examples) < 5:
                raw_path = str(item.get("image_path") or item.get("image") or item.get("img") or "")
                missing_examples.append((idx, raw_path))

    if total == 0:
        logger.error(f"JSONL is empty: {jsonl_path}")
        return False

    ratio = exists / total
    logger.info(f"JSONL file: {jsonl_path}")
    logger.info(f"Image availability: {exists}/{total} ({ratio:.2%})")
    if image_roots:
        logger.info(f"Extra image roots: {image_roots}")

    if missing_examples:
        logger.warning("Examples of unresolved image paths (line, raw_path):")
        for line_no, raw in missing_examples:
            logger.warning(f"  - {line_no}: {raw}")

    ok = True
    if min_image_exists_ratio is not None and ratio < float(min_image_exists_ratio):
        logger.error(
            f"Failed min_image_exists_ratio: {ratio:.2%} < {float(min_image_exists_ratio):.2%}"
        )
        ok = False
    if min_image_exists_count is not None and exists < int(min_image_exists_count):
        logger.error(
            f"Failed min_image_exists_count: {exists} < {int(min_image_exists_count)}"
        )
        ok = False

    return ok

def main():
    args = parse_args()
    logger = setup_logger("prepare_data")
    
    logger.info(f"Checking {args.dataset} at {args.data_root}")
    
    if args.dataset in {"gemex", "s-chain", "schain", "cxr-counterfact", "custom_jsonl"}:
        ok = check_jsonl_images(
            data_root=args.data_root,
            jsonl_name=args.jsonl_name,
            image_roots=args.image_roots,
            logger=logger,
            min_image_exists_ratio=args.min_image_exists_ratio,
            min_image_exists_count=args.min_image_exists_count,
        )
    else:
        logger.error(f"Unsupported dataset: {args.dataset}")
        ok = False
        
    if ok:
        logger.info("Dataset structure verification passed!")
    else:
        logger.error("Dataset structure verification failed. Please check missing files.")
        sys.exit(1)

if __name__ == "__main__":
    main()
