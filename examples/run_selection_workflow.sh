#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:=configs/gemex_qwen3vl_vicor.yaml}"
: "${BASE_MODEL_PATH:?set BASE_MODEL_PATH to the local or Hugging Face Qwen3-VL checkpoint}"
: "${DATA_ROOT:?set DATA_ROOT to the parent directory containing the configured dataset}"
: "${OUTPUT_ROOT:=outputs/gemex_vicor}"
: "${PYTHON:=python3}"
: "${EXPORT_MODE:=absolute}"
: "${STAGE1_ENABLED:=0}"
: "${STAGE1_CANDIDATE_RATIO:=0}"

# Set RAW_PREPROCESS_ENABLED=1 to make this workflow start from raw JSONL and
# execute the Appendix-B Qwen3-4B denoise/classify/dedup pass before scoring.
# The default remains prepared-input mode because raw data and checkpoints are
# not part of the release package.
: "${RAW_PREPROCESS_ENABLED:=0}"
: "${RAW_INPUT_JSONL:=}"
: "${RAW_PREPARED_DIR:=}"
: "${RAW_DENOISER_MODEL:=}"
: "${RAW_CLASSIFIER_MODEL:=}"
: "${RAW_IMAGE_ROOT:=}"
: "${RAW_POOL_SIZE:=}"
: "${RAW_RESERVE_SIZE:=1000}"

export BASE_MODEL_PATH DATA_ROOT OUTPUT_ROOT
mkdir -p "$OUTPUT_ROOT"

CONFIG_SOURCE="$CONFIG"
if [[ "$RAW_PREPROCESS_ENABLED" == "1" ]]; then
  if [[ -z "$RAW_INPUT_JSONL" ]]; then
    echo "RAW_INPUT_JSONL is required when RAW_PREPROCESS_ENABLED=1" >&2
    exit 2
  fi
  RAW_PREPARED_DIR="${RAW_PREPARED_DIR:-$OUTPUT_ROOT/raw_prepared}"
  RAW_CONFIG="$OUTPUT_ROOT/raw_preprocessed_config.yaml"
  RAW_DATASET="$($PYTHON - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    name = str((yaml.safe_load(f) or {}).get("dataset_name", "")).lower()
if name in {"cxr-counterfact", "cxr_counterfact"}:
    print("cxr")
elif name in {"gemex"}:
    print("gemex")
elif name in {"s-chain", "schain"}:
    print("schain")
else:
    raise SystemExit(f"Raw Appendix-B preprocessing is unsupported for dataset_name={name!r}")
PY
)"
  RAW_PREPARE_CMD=("$PYTHON" scripts/00_preprocess_training_pools.py
    --dataset "$RAW_DATASET"
    --input "$RAW_INPUT_JSONL"
    --output_dir "$RAW_PREPARED_DIR"
    --reserve_size "$RAW_RESERVE_SIZE")
  if [[ -n "$RAW_POOL_SIZE" ]]; then
    RAW_PREPARE_CMD+=(--pool_size "$RAW_POOL_SIZE")
  fi
  if [[ -n "$RAW_IMAGE_ROOT" ]]; then
    RAW_PREPARE_CMD+=(--image_root "$RAW_IMAGE_ROOT")
  fi
  if [[ -n "$RAW_DENOISER_MODEL" ]]; then
    RAW_PREPARE_CMD+=(--denoiser_model "$RAW_DENOISER_MODEL")
  fi
  if [[ -n "$RAW_CLASSIFIER_MODEL" ]]; then
    RAW_PREPARE_CMD+=(--classifier_model "$RAW_CLASSIFIER_MODEL")
  fi
  "${RAW_PREPARE_CMD[@]}"
  # Point a temporary run config at the generated fixed-size pool.  The
  # configured JSONL name already matches the preprocessor output names:
  # gemex_train_60k.jsonl, schain_train_10k.jsonl, or CXR task pools.
  "$PYTHON" - "$CONFIG" "$RAW_PREPARED_DIR" "$RAW_CONFIG" <<'PY'
import sys, yaml
source, prepared, output = sys.argv[1:]
with open(source, encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg["data_root"] = prepared
cfg.setdefault("data", {})["image_roots"] = [prepared]
with open(output, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
  CONFIG_SOURCE="$RAW_CONFIG"
fi

CONFIG_RUN="$CONFIG_SOURCE"
if [[ "$STAGE1_ENABLED" == "1" ]]; then
  STAGE1_CONFIG="$OUTPUT_ROOT/stage1_config.yaml"
  "$PYTHON" scripts/00_stage1_filter.py \
    --config "$CONFIG_SOURCE" \
    --output_config "$STAGE1_CONFIG" \
    --candidate_ratio "$STAGE1_CANDIDATE_RATIO"
  CONFIG_RUN="$STAGE1_CONFIG"
fi

CONFIG_VALUES="$($PYTHON - "$CONFIG_RUN" <<'PY'
import os
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value

cfg = expand(cfg)
data = cfg.get("data", {})
image_roots = data.get("image_roots", [])
if isinstance(image_roots, str):
    image_roots = [image_roots]
print(cfg.get("dataset_name", "custom_jsonl"))
print(cfg.get("data_root", ""))
print(data.get("jsonl_name", "data.jsonl"))
print(cfg.get("out_dir", "outputs/selection"))
print("\t".join(str(p) for p in image_roots if str(p).strip()))
selection = (cfg.get("vicor", {}) or {}).get("selection", {}) or {}
ratio = selection.get("ratio", (cfg.get("vicor", {}) or {}).get("top_ratio"))
if ratio is None:
    raise SystemExit("Config must define vicor.selection.ratio or vicor.top_ratio")
print(ratio)
PY
)"
CONFIG_LINES=()
while IFS= read -r line; do
  CONFIG_LINES+=("$line")
done <<< "$CONFIG_VALUES"
DATASET_NAME="${CONFIG_LINES[0]}"
DATA_ROOT_EXPANDED="${CONFIG_LINES[1]}"
JSONL_NAME="${CONFIG_LINES[2]}"
SELECTION_OUT="${CONFIG_LINES[3]}"
IMAGE_ROOTS_LINE="${CONFIG_LINES[4]:-}"
FINAL_SELECTION_RATIO="${CONFIG_LINES[5]:-}"
if [[ -z "$FINAL_SELECTION_RATIO" ]]; then
  echo "Configuration did not provide vicor.selection.ratio (or vicor.top_ratio)" >&2
  exit 2
fi

PREPARE_CMD=("$PYTHON" scripts/00_prepare_data.py --dataset "$DATASET_NAME" --data_root "$DATA_ROOT_EXPANDED" --jsonl_name "$JSONL_NAME")
if [[ -n "$IMAGE_ROOTS_LINE" ]]; then
  IFS=$'\t' read -r -a IMAGE_ROOTS <<< "$IMAGE_ROOTS_LINE"
  PREPARE_CMD+=(--image_roots "${IMAGE_ROOTS[@]}")
fi
"${PREPARE_CMD[@]}"

"$PYTHON" scripts/01_score_vicor.py --config "$CONFIG_RUN"
"$PYTHON" scripts/02_select_subset.py --scores "$SELECTION_OUT/scores.parquet" --config "$CONFIG_RUN" --ratio "$FINAL_SELECTION_RATIO"

if [[ "$EXPORT_MODE" == "symlink" ]]; then
  EXPORT_SCRIPT=scripts/03_export_to_llama_factory_symlink.py
else
  EXPORT_SCRIPT=scripts/03_export_to_llama_factory_absolute.py
fi
"$PYTHON" "$EXPORT_SCRIPT" --config "$CONFIG_RUN" --subset "$SELECTION_OUT/selected_ids.json" --output_dir "$SELECTION_OUT/llama_factory_data"
