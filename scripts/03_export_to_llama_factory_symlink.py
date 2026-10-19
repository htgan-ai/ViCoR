from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vicor.config import load_config
from vicor.data.gemex import GEMeXDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--subset", type=str, required=True, help="Path to selected_ids.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LLaMA-Factory dataset")
    parser.add_argument("--dataset_name", type=str, default="", help="Optional dataset key override for dataset_info.json")
    return parser.parse_args()


def _first_nonempty(*vals: object) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _build_conversation(item: dict) -> list[dict[str, str]] | None:
    question = _first_nonempty(item.get("question"))
    answer_cot = _first_nonempty(item.get("answer_cot"))
    answer = _first_nonempty(item.get("answer"))
    reasoning = _first_nonempty(item.get("reasoning"))

    if not question:
        conversations = item.get("conversations")
        if isinstance(conversations, list):
            for msg in conversations:
                if isinstance(msg, dict) and str(msg.get("from", "")).lower() in {"human", "user"}:
                    question = str(msg.get("value", "")).replace("<image>", "").strip()
                    break

    assistant = ""
    if reasoning and answer:
        assistant = f"<think>{reasoning}</think>{answer}" if reasoning else answer
    elif answer_cot:
        assistant = answer_cot
    else:
        assistant = answer

    if not assistant:
        conversations = item.get("conversations")
        if isinstance(conversations, list):
            for msg in conversations:
                if isinstance(msg, dict) and str(msg.get("from", "")).lower() in {"gpt", "assistant"}:
                    assistant = str(msg.get("value", "")).strip()
                    break

    if not question or not assistant:
        return None

    return [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": assistant},
    ]


def _load_dataset(config: dict):
    dataset_name = str(config.get("dataset_name", "pathvqa")).lower()
    data_root = config["data_root"]
    data_cfg = config.get("data", {})
    use_cot = data_cfg.get("use_cot", True)
    strict_image_paths = bool(data_cfg.get("strict_image_paths", False))
    skip_image_exists_scan = bool(data_cfg.get("skip_image_exists_scan", True))
    force_mean_image = bool(data_cfg.get("force_mean_image", False))
    image_open_timeout_sec = float(data_cfg.get("image_open_timeout_sec", 0.0))
    image_roots_cfg = data_cfg.get("image_roots", [])
    if isinstance(image_roots_cfg, str):
        image_roots = [image_roots_cfg]
    elif isinstance(image_roots_cfg, list):
        image_roots = [str(p) for p in image_roots_cfg if str(p).strip()]
    else:
        image_roots = []

    if dataset_name in {"gemex", "s-chain", "schain", "cxr-counterfact", "cxr_counterfact", "custom_jsonl"}:
        jsonl_name = data_cfg.get("jsonl_name", "data.jsonl")
        ds = GEMeXDataset(
            data_root=data_root,
            jsonl_name=jsonl_name,
            use_cot=use_cot,
            strict_image_paths=strict_image_paths,
            image_roots=image_roots,
            skip_image_exists_scan=skip_image_exists_scan,
            force_mean_image=force_mean_image,
            image_open_timeout_sec=image_open_timeout_sec,
        )
        iterable = ds.data
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for symlink export.")

    return dataset_name, iterable


def _dataset_key(args: argparse.Namespace, dataset_name: str) -> str:
    if args.dataset_name.strip():
        return args.dataset_name.strip()
    return f"vicor_selected_{dataset_name.replace('-', '_')}"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    with open(args.subset, "r", encoding="utf-8") as f:
        selected_ids = {str(x) for x in json.load(f)}

    dataset_name, iterable = _load_dataset(config)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    llama_data: list[dict[str, object]] = []
    exported = 0
    missing_image = 0
    missing_text = 0

    for item in iterable:
        sample_id = str(item.get("id", ""))
        if sample_id not in selected_ids:
            continue

        conversation = _build_conversation(item)
        if conversation is None:
            missing_text += 1
            raise ValueError(f"Selected sample id={sample_id} has no complete question/answer conversation")

        src = str(item.get("_resolved_image_path", "") or item.get("image_path", "")).strip()
        if not src or not os.path.exists(src):
            missing_image += 1
            raise FileNotFoundError(
                f"Selected sample id={sample_id} has no readable image; "
                "fix image roots before using symlink export"
            )

        ext = Path(src).suffix or ".jpg"
        img_name = f"{sample_id}_{hashlib.md5(src.encode()).hexdigest()[:12]}{ext}"
        dest = images_dir / img_name
        if not dest.exists():
            # Link to the absolute source; a relative target would be resolved
            # against the images directory and break.
            os.symlink(os.path.abspath(src), dest)

        llama_data.append(
            {
                "conversations": conversation,
                "images": [f"images/{img_name}"],
            }
        )
        exported += 1

    if exported != len(selected_ids):
        raise RuntimeError(
            f"Exported {exported} of {len(selected_ids)} selected samples; "
            "paper-faithful export must preserve the selected subset size"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset.json").write_text(json.dumps(llama_data, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        _dataset_key(args, dataset_name): {
            "file_name": "dataset.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        }
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Loaded {len(selected_ids)} selected IDs. exported={exported} "
        f"missing_image={missing_image} missing_text={missing_text} out={out_dir}"
    )


if __name__ == "__main__":
    main()
