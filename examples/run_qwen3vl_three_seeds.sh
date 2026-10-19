#!/usr/bin/env bash
set -euo pipefail

# Run the included Qwen3-VL trainer and evaluation chain for independent seeds,
# then aggregate the resulting benchmark summaries. Comparison-model runs are
# intentionally outside this Qwen-only release.
: "${BASE_MODEL_PATH:?set BASE_MODEL_PATH}"
: "${EVAL_ROOT:?set EVAL_ROOT}"
: "${SELECTION_ROOT:=outputs/gemex_vicor}"   # OUTPUT_ROOT of examples/run_selection_workflow.sh
: "${DATASET_DIR:=$SELECTION_ROOT/gemex_qwen3vl_vicor/llama_factory_data}"
: "${TRAIN_JSONL:=$DATASET_DIR/dataset.json}"
: "${TRAIN_DATA_ROOT:=$DATASET_DIR}"
: "${OUTPUT_ROOT:=outputs/qwen3vl_three_seeds}"
: "${PYTHON:=python3}"
: "${DEVICE:=cuda:0}"
: "${TRAIN_DEVICE:=$DEVICE}"
: "${SEEDS:=17 42 73}"
: "${MAX_SAMPLES:=0}"
: "${MAX_NEW_TOKENS:=16}"
: "${REPORT_MAX_NEW_TOKENS:=512}"
: "${NUM_TRAIN_EPOCHS:=3}"
: "${IMAGE_MAX_PIXELS:=262144}"
: "${IMAGE_MIN_PIXELS:=50176}"

read -r -a SEED_LIST <<< "$SEEDS"
if [[ "${#SEED_LIST[@]}" -ne 3 ]]; then
  echo "SEEDS must contain exactly three independent integer seeds (got ${#SEED_LIST[@]})" >&2
  exit 2
fi

if [[ "${SEED_LIST[0]}" == "${SEED_LIST[1]}" || "${SEED_LIST[0]}" == "${SEED_LIST[2]}" || "${SEED_LIST[1]}" == "${SEED_LIST[2]}" ]]; then
  echo "SEEDS must contain three distinct values" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
RUN_DIRS=()
for seed in "${SEED_LIST[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 2
  fi
  RUN_ROOT="$OUTPUT_ROOT/seed_$seed"
  ADAPTER_DIR="$RUN_ROOT/adapter"
  EVAL_DIR="$RUN_ROOT/evaluation"
  mkdir -p "$RUN_ROOT"

  "$PYTHON" training/train_qwen3vl_subset_lora.py \
    --model_path "$BASE_MODEL_PATH" \
    --train_jsonl "$TRAIN_JSONL" \
    --data_root "$TRAIN_DATA_ROOT" \
    --output_dir "$ADAPTER_DIR" \
    --run_name "qwen3vl_seed_${seed}" \
    --seed "$seed" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --device "$TRAIN_DEVICE" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    --bf16

  "$PYTHON" evaluation/run_qwen3vl_evaluation.py \
    --adapter_path "$ADAPTER_DIR" \
    --base_model_path "$BASE_MODEL_PATH" \
    --eval_root "$EVAL_ROOT" \
    --output_dir "$EVAL_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    --report_max_new_tokens "$REPORT_MAX_NEW_TOKENS" \
    --device "$DEVICE"
  RUN_DIRS+=("$EVAL_DIR")
done

"$PYTHON" evaluation/aggregate_seed_runs.py \
  --run_dirs "${RUN_DIRS[@]}" \
  --output "$OUTPUT_ROOT/aggregate_seed_summary.json"
