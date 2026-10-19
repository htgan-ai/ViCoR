#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME_OR_PATH:?set MODEL_NAME_OR_PATH}"
: "${DATASET_NAME:?set DATASET_NAME}"
: "${DATASET_DIR:?set DATASET_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${LLAMAFACTORY_CLI:=llamafactory-cli}"
: "${IMAGE_MIN_PIXELS:=50176}"
: "${IMAGE_MAX_PIXELS:=262144}"

if ! command -v "$LLAMAFACTORY_CLI" >/dev/null 2>&1 && [[ ! -x "$LLAMAFACTORY_CLI" ]]; then
  cat >&2 <<'EOF'
LLaMA-Factory is an optional external dependency for this entrypoint.
Install and pin the launcher in your training environment, or set
LLAMAFACTORY_CLI to its executable path before running this script.
The direct trainer (training/train_qwen3vl_subset_lora.py) does not require it.
EOF
  exit 127
fi

"$LLAMAFACTORY_CLI" train   --stage sft   --do_train   --model_name_or_path "$MODEL_NAME_OR_PATH"   --dataset "$DATASET_NAME"   --dataset_dir "$DATASET_DIR"   --template "${LLAMAFACTORY_TEMPLATE:-qwen3_vl}"   --finetuning_type lora   --lora_rank 16   --lora_alpha 32   --lora_dropout 0.05   --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj   --output_dir "$OUTPUT_DIR"   --overwrite_output_dir   --learning_rate 1e-4   --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}"   --warmup_ratio 0.03   --lr_scheduler_type cosine   --weight_decay 0.05   --bf16   --per_device_train_batch_size 1   --gradient_accumulation_steps 16   --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS:-4}"   --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}"   --image_min_pixels "$IMAGE_MIN_PIXELS"   --image_max_pixels "$IMAGE_MAX_PIXELS"   --logging_steps 10   --save_steps 500   --save_total_limit 3   --plot_loss   --report_to none   --seed "${SEED:-42}"   --ddp_timeout 180000000
