#!/usr/bin/env bash
set -euo pipefail

# Fine-tune Qwen3-VL on the exported ViCoR subset with the included direct
# LoRA trainer.  Set TRAINER=llamafactory to use the optional LLaMA-Factory
# wrapper instead (requires a separately installed llamafactory-cli).
: "${BASE_MODEL_PATH:=${MODEL_NAME_OR_PATH:-}}"
: "${BASE_MODEL_PATH:?set BASE_MODEL_PATH (or MODEL_NAME_OR_PATH) to the Qwen3-VL checkpoint}"
: "${SELECTION_ROOT:=outputs/gemex_vicor}"   # OUTPUT_ROOT of examples/run_selection_workflow.sh
: "${DATASET_DIR:=$SELECTION_ROOT/gemex_qwen3vl_vicor/llama_factory_data}"
: "${TRAIN_JSONL:=$DATASET_DIR/dataset.json}"
: "${TRAIN_DATA_ROOT:=$DATASET_DIR}"
: "${OUTPUT_DIR:=outputs/gemex_vicor/gemex_qwen3vl_vicor/qwen3_lora}"
: "${SEED:=42}"
: "${PYTHON:=python3}"
: "${DEVICE:=cuda:0}"
: "${NUM_TRAIN_EPOCHS:=3}"
: "${IMAGE_MAX_PIXELS:=262144}"
: "${IMAGE_MIN_PIXELS:=50176}"
: "${TRAINER:=direct}"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Training data not found: $TRAIN_JSONL" >&2
  echo "Run examples/run_selection_workflow.sh first, or set DATASET_DIR to its llama_factory_data directory." >&2
  exit 2
fi

if [[ "$TRAINER" == "llamafactory" ]]; then
  export MODEL_NAME_OR_PATH="$BASE_MODEL_PATH" DATASET_NAME="${DATASET_NAME:-vicor_selected_gemex}" DATASET_DIR OUTPUT_DIR SEED
  exec bash training/train_qwen3vl_llamafactory_lora.sh
fi

exec "$PYTHON" training/train_qwen3vl_subset_lora.py \
  --model_path "$BASE_MODEL_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --data_root "$TRAIN_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "qwen3vl_seed_${SEED}" \
  --seed "$SEED" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --device "$DEVICE" \
  --image_max_pixels "$IMAGE_MAX_PIXELS" \
  --image_min_pixels "$IMAGE_MIN_PIXELS" \
  --bf16
