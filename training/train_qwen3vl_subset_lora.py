#!/usr/bin/env python3
"""Train a Qwen3-VL LoRA adapter on a selected JSONL subset."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, get_scheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vicor.data.label_masks import assistant_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT training for Qwen3-VL on subset JSONL")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="vicor_qwen3vl")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_from_latest", action="store_true")

    # Runbook-aligned core hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", action="store_true")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--image_max_pixels", type=int, default=262144)
    parser.add_argument("--image_min_pixels", type=int, default=50176)
    parser.add_argument(
        "--allow_mean_image_fallback",
        action="store_true",
        help="Explicitly replace unreadable images with the processor-mean image (changes visual training semantics).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG used by model/dropout code."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a checkpoint RNG snapshot before the first resumed forward."""
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        saved_cuda = state["cuda"]
        if len(saved_cuda) != torch.cuda.device_count():
            raise RuntimeError(
                "Checkpoint CUDA RNG state was recorded for "
                f"{len(saved_cuda)} devices, but this run has {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(saved_cuda)


def load_rng_state(checkpoint_dir: str) -> dict[str, Any] | None:
    path = os.path.join(checkpoint_dir, "rng_state.pth")
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 has no weights_only keyword.
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Invalid RNG state file: {path}")
    return state


def mean_rgb_image(size: tuple[int, int] = (448, 448)) -> Image.Image:
    return Image.new("RGB", size, (123, 117, 104))


def safe_load_rgb_image(image_path: str, timeout_sec: int = 10, allow_mean_image_fallback: bool = False) -> Image.Image:
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out opening image: {image_path}")

    if not image_path:
        if allow_mean_image_fallback:
            return mean_rgb_image()
        raise FileNotFoundError("Training sample has no readable image path")
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            image.load()
        return image
    except Exception:
        if allow_mean_image_fallback:
            return mean_rgb_image()
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class JsonlSubsetDataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: int = 0) -> None:
        self.samples: list[dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            text = f.read()
        rows = json.loads(text) if text.lstrip().startswith("[") else [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
        for row in rows:
            if "conversations" in row:
                conversations = row["conversations"]
                images = row.get("images", [])
                if (len(conversations) != 2 or len(images) != 1
                        or conversations[0].get("from") != "human"
                        or conversations[1].get("from") != "gpt"):
                    raise ValueError("Training export must contain one human/assistant pair and one image")
                row = {
                    "image_path": images[0],
                    "question": conversations[0]["value"].replace("<image>", "").strip(),
                    "answer": conversations[1]["value"],
                }
            self.samples.append(row)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class EpochRandomSampler(Sampler[int]):
    """Stable per-epoch permutations so checkpoint resume skips the same rows."""

    def __init__(self, data_source: Dataset, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


@dataclass
class BatchCollator:
    processor: Any
    data_root: str
    image_pad_id: int | None
    pad_id: int | None
    allow_mean_image_fallback: bool = False

    def apply_chat_template_compat(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        image_token = getattr(self.processor, "image_token", "<image>")
        converted_messages: list[dict[str, str]] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if item.get("type") == "image":
                        parts.append(image_token)
                    elif item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                content = "\n".join([p for p in parts if p]).strip()
            converted_messages.append({"role": msg["role"], "content": str(content)})
        if tokenizer is not None and image_token == "<IMG_CONTEXT>":
            return tokenizer.apply_chat_template(
                converted_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except ValueError:
            if tokenizer is None:
                raise
            return tokenizer.apply_chat_template(
                converted_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def __call__(self, batch_rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        user_texts: list[str] = []
        images: list[Image.Image] = []
        answers: list[str] = []

        for row in batch_rows:
            image_rel = str(row.get("image_path", row.get("image", ""))).strip()
            if not image_rel:
                raise RuntimeError("Missing image_path/image in training sample")

            image_path = os.path.join(self.data_root, image_rel)
            if not os.path.exists(image_path):
                resolved = None
                for _ in range(6):
                    fallback_path = os.path.join(self.data_root, "images", os.path.basename(image_rel))
                    if os.path.exists(image_path):
                        resolved = image_path
                        break
                    if os.path.exists(fallback_path):
                        resolved = fallback_path
                        break
                    time.sleep(0.5)
                if resolved is None:
                    image_path = ""
                else:
                    image_path = resolved

            image = None
            last_blocking_err: BlockingIOError | None = None
            for attempt in range(8):
                try:
                    image = safe_load_rgb_image(
                        image_path,
                        allow_mean_image_fallback=self.allow_mean_image_fallback,
                    )
                    if getattr(self.processor, "image_token", None) == "<IMG_CONTEXT>":
                        image = image.resize((448, 448))
                    break
                except BlockingIOError as e:
                    last_blocking_err = e
                    time.sleep(min(0.2 * (attempt + 1), 1.0))
            if image is None and self.allow_mean_image_fallback:
                image = mean_rgb_image()
            if image is None:
                raise RuntimeError(f"Unable to load training image: {image_path}") from last_blocking_err
            question = str(row.get("question", "")).strip()
            reasoning = str(row.get("reasoning", "")).strip()
            answer = str(row.get("answer", "")).strip()
            assistant_text = f"<think>{reasoning}</think>{answer}" if reasoning else answer

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]

            text = self.apply_chat_template_compat(messages)
            user_text = self.apply_chat_template_compat(
                [messages[0]],
                add_generation_prompt=True,
            )
            texts.append(text)
            user_texts.append(user_text)
            images.append(image)
            answers.append(answer)

        model_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        user_inputs = self.processor(
            text=user_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        user_lens = user_inputs["attention_mask"].sum(dim=1)
        for row_idx, user_len in enumerate(user_lens):
            active = model_inputs["attention_mask"][row_idx].nonzero(as_tuple=True)[0]
            prefix = user_inputs["input_ids"][row_idx][user_inputs["attention_mask"][row_idx].bool()]
            full = model_inputs["input_ids"][row_idx][active]
            n = int(user_len.item())
            if n >= full.numel() or not torch.equal(full[:n], prefix):
                raise ValueError("User template is not a prefix of the full conversation; verify the model chat template and truncation")
        # The paper restricts the *selection score* to final-answer tokens;
        # SFT itself uses the exported supervised answer, including rationale.
        model_inputs["labels"] = assistant_labels(
            model_inputs["input_ids"], model_inputs["attention_mask"],
            user_lens, self.pad_id,
        )
        return model_inputs


def save_json(path: str, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_vlm_model(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    load_errors: list[str] = []
    for model_cls in (AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel):
        try:
            return model_cls.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
        except Exception as e:
            load_errors.append(f"{model_cls.__name__}: {e}")
    raise RuntimeError("Failed to load model with supported auto classes: " + " | ".join(load_errors))


def find_latest_checkpoint(output_dir: str) -> tuple[str | None, int]:
    root = Path(output_dir)
    if not root.exists():
        return None, 0
    candidates: list[tuple[int, str]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("checkpoint-"):
            continue
        step_str = p.name.split("checkpoint-")[-1]
        if not step_str.isdigit():
            continue
        candidates.append((int(step_str), str(p)))
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    step, path = candidates[-1]
    return path, step


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if not getattr(processor, "chat_template", None) and getattr(processor, "tokenizer", None) is not None:
        tokenizer_chat_template = getattr(processor.tokenizer, "chat_template", None)
        if tokenizer_chat_template:
            processor.chat_template = tokenizer_chat_template
    # Keep the direct trainer's processor bounds identical to the release
    # configs and scoring/evaluation loaders.
    if hasattr(processor, "image_processor"):
        for name, value in (("max_pixels", args.image_max_pixels), ("min_pixels", args.image_min_pixels)):
            if hasattr(processor.image_processor, name):
                try:
                    setattr(processor.image_processor, name, value)
                except Exception:
                    pass

    model = load_vlm_model(args.model_path, dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    resume_checkpoint = ""
    resume_step = 0
    if args.resume_from_checkpoint:
        resume_checkpoint = args.resume_from_checkpoint
        ck_name = Path(resume_checkpoint).name
        if ck_name.startswith("checkpoint-") and ck_name.split("checkpoint-")[-1].isdigit():
            resume_step = int(ck_name.split("checkpoint-")[-1])
    elif args.resume_from_latest:
        latest_path, latest_step = find_latest_checkpoint(args.output_dir)
        if latest_path is not None:
            resume_checkpoint = latest_path
            resume_step = latest_step

    resume_rng_state = load_rng_state(resume_checkpoint) if resume_checkpoint else None
    resume_rng_available = resume_rng_state is not None
    if resume_checkpoint and resume_rng_state is None:
        print(f"Warning: {resume_checkpoint} has no rng_state.pth; resumed dropout/augmentation stream cannot be bitwise identical")

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if resume_checkpoint:
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    model = model.to(device)
    model_dtype = next(model.parameters()).dtype
    model.train()

    dataset = JsonlSubsetDataset(args.train_jsonl, max_samples=args.max_train_samples)
    if len(dataset) == 0:
        raise RuntimeError("Empty dataset after loading")

    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pad_id = processor.tokenizer.pad_token_id
    collator = BatchCollator(
        processor=processor,
        data_root=args.data_root,
        image_pad_id=image_pad_id,
        pad_id=pad_id,
        allow_mean_image_fallback=args.allow_mean_image_fallback,
    )

    sampler = EpochRandomSampler(dataset, args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = updates_per_epoch * args.num_train_epochs
    warmup_steps = int(max_train_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    if resume_step > max_train_steps:
        resume_step = max_train_steps

    if resume_checkpoint:
        optimizer_path = os.path.join(resume_checkpoint, "optimizer.pt")
        scheduler_path = os.path.join(resume_checkpoint, "scheduler.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        else:
            for _ in range(resume_step):
                scheduler.step()

    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "train_samples": len(dataset),
                "batch_size": args.per_device_train_batch_size,
                "grad_accum": args.gradient_accumulation_steps,
                "epochs": args.num_train_epochs,
                "max_train_steps": max_train_steps,
                "warmup_steps": warmup_steps,
                "output_dir": args.output_dir,
                "resume_checkpoint": resume_checkpoint,
                "resume_step": resume_step,
            },
            ensure_ascii=False,
        )
    )

    global_step = resume_step
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    start_epoch = min(args.num_train_epochs, global_step // updates_per_epoch)
    resume_updates_in_epoch = global_step % updates_per_epoch
    recent_losses: list[float] = []

    for epoch in range(start_epoch, args.num_train_epochs):
        sampler.set_epoch(epoch)
        skip_micro_batches = 0
        if epoch == start_epoch:
            skip_micro_batches = resume_updates_in_epoch * args.gradient_accumulation_steps
        # Creating a DataLoader iterator consumes a torch seed (and starts
        # workers).  A checkpoint written mid-epoch captured the state after
        # that draw, so it is restored right before the first resumed forward;
        # a checkpoint written at an epoch boundary captured the state before
        # the next epoch's iterator existed, so it is restored first.
        restore_pending = resume_rng_state is not None and epoch == start_epoch
        if restore_pending and skip_micro_batches == 0:
            restore_rng_state(resume_rng_state)
            restore_pending = False
            resume_rng_state = None
        train_iterator = iter(train_loader)
        pbar = tqdm(train_iterator, total=len(train_loader), desc=f"{args.run_name} epoch {epoch + 1}/{args.num_train_epochs}")
        for step_idx, batch in enumerate(pbar, start=1):
            if step_idx <= skip_micro_batches:
                continue
            if restore_pending:
                restore_rng_state(resume_rng_state)
                restore_pending = False
                resume_rng_state = None
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(device=device, dtype=model_dtype, non_blocking=True)

            outputs = model(**batch)
            loss = outputs.loss
            window_start = ((step_idx - 1) // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            window_size = min(args.gradient_accumulation_steps, len(train_loader) - window_start)
            (loss / window_size).backward()
            recent_losses.append(float(loss.item()))

            should_step = (
                (step_idx % args.gradient_accumulation_steps == 0)
                or (step_idx == len(train_loader))
            )
            if should_step:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = sum(recent_losses[-args.logging_steps:]) / min(len(recent_losses), args.logging_steps)
                    print(
                        json.dumps(
                            {
                                "run_name": args.run_name,
                                "global_step": global_step,
                                "max_train_steps": max_train_steps,
                                "epoch": epoch + 1,
                                "loss": round(avg_loss, 6),
                                "lr": scheduler.get_last_lr()[0],
                            },
                            ensure_ascii=False,
                        )
                    )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    # Snapshot before serialization hooks so the state maps to
                    # the next forward even if a custom saver uses randomness.
                    rng_snapshot = capture_rng_state()
                    model.save_pretrained(ckpt_dir)
                    processor.save_pretrained(ckpt_dir)
                    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                    torch.save(rng_snapshot, os.path.join(ckpt_dir, "rng_state.pth"))
                    save_json(
                        os.path.join(ckpt_dir, "training_state.json"),
                        {
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "step_idx": step_idx,
                            "max_train_steps": max_train_steps,
                            "sampler_seed": args.seed,
                            "sampler_epoch": epoch,
                            "rng_state_file": "rng_state.pth",
                        },
                    )

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    meta = {
        "run_name": args.run_name,
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "bf16": bool(args.bf16),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": target_modules,
        "max_train_steps": max_train_steps,
        "sampler_seed": args.seed,
        "resume_rng_restored": bool(resume_rng_available),
        "elapsed_sec": time.time() - start_time,
    }
    save_json(os.path.join(args.output_dir, "train_meta.json"), meta)
    print(json.dumps({"status": "done", **meta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
