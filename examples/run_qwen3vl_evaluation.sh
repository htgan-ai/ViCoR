#!/usr/bin/env bash
set -euo pipefail

: "${ADAPTER_PATH:?set ADAPTER_PATH}"
: "${BASE_MODEL_PATH:?set BASE_MODEL_PATH}"
: "${EVAL_ROOT:?set EVAL_ROOT to the benchmark root}"
: "${OUTPUT_DIR:=outputs/eval_single}"
: "${PYTHON:=python}"
: "${IMAGE_MAX_PIXELS:=262144}"
: "${IMAGE_MIN_PIXELS:=50176}"
: "${REPORT_METRIC_MODE:=official}"
: "${REPORT_MAX_NEW_TOKENS:=512}"
# Leave empty for the paper-faithful strict schema check.  Set to 1 only for
# an explicitly requested open-answer smoke test on a manifest without
# options/label_space.
: "${ALLOW_FREE_FORM_EXACT:=}"

"$PYTHON" evaluation/run_qwen3vl_evaluation.py \
  --adapter_path "$ADAPTER_PATH" \
  --base_model_path "$BASE_MODEL_PATH" \
  --eval_root "$EVAL_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --image_max_pixels "$IMAGE_MAX_PIXELS" \
  --image_min_pixels "$IMAGE_MIN_PIXELS" \
  --report_metric_mode "$REPORT_METRIC_MODE" \
  --report_max_new_tokens "$REPORT_MAX_NEW_TOKENS" \
  --max_samples 0

if [[ -n "${PERTURBATION_MANIFEST:-}" ]]; then
  for mode in normal no_image wrong_image; do
    "$PYTHON" evaluation/evaluate_visual_perturbations_qwen3vl.py \
      --manifest "$PERTURBATION_MANIFEST" \
      --adapter_path "$ADAPTER_PATH" \
      --base_model_path "$BASE_MODEL_PATH" \
      --mode "$mode" \
      --output "$OUTPUT_DIR/perturbation_${mode}.json" \
      --image_max_pixels "$IMAGE_MAX_PIXELS" \
      --image_min_pixels "$IMAGE_MIN_PIXELS" \
      --max_samples 0
  done
  "$PYTHON" evaluation/aggregate_perturbation_results.py \
    --condition_a "$OUTPUT_DIR/perturbation_normal.json" \
    --condition_b "$OUTPUT_DIR/perturbation_no_image.json" \
    --output "$OUTPUT_DIR/perturbation_drop_no_image.json"
  "$PYTHON" evaluation/aggregate_perturbation_results.py \
    --condition_a "$OUTPUT_DIR/perturbation_normal.json" \
    --condition_b "$OUTPUT_DIR/perturbation_wrong_image.json" \
    --output "$OUTPUT_DIR/perturbation_drop_wrong_image.json"
fi

if [[ -n "${GROUNDING_MANIFEST:-}" ]]; then
  # The shared-correct population is defined from normal-image predictions,
  # not from either masked condition.  Always materialize this normal result
  # so a complete evidence audit can be assembled in one workflow.  To build
  # the paper's 362-ID intersection automatically, provide
  # GROUNDING_NORMAL_RESULTS as a colon-separated list of normal-result JSON
  # files from the other compared models/seeds.  The current run is appended
  # below.  An existing GROUNDING_SHARED_IDS file may be supplied instead.
  grounding_free_form_args=()
  if [[ "$ALLOW_FREE_FORM_EXACT" == "1" ]]; then
    grounding_free_form_args+=(--allow_free_form_exact)
  fi
  "$PYTHON" evaluation/evaluate_visual_perturbations_qwen3vl.py \
    --manifest "$GROUNDING_MANIFEST" \
    --adapter_path "$ADAPTER_PATH" \
    --base_model_path "$BASE_MODEL_PATH" \
    --mode normal \
    --output "$OUTPUT_DIR/grounding_normal.json" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    "${grounding_free_form_args[@]}" \
    --allow_nonpaper_sample_size \
    --max_samples 0

  if [[ -z "${GROUNDING_SHARED_IDS:-}" ]]; then
    : "${GROUNDING_NORMAL_RESULTS:?set GROUNDING_SHARED_IDS or GROUNDING_NORMAL_RESULTS (colon-separated normal JSON files for every other compared model/seed)}"
    normal_result_args=()
    IFS=':' read -r -a external_normal_results <<< "$GROUNDING_NORMAL_RESULTS"
    for normal_result in "${external_normal_results[@]}"; do
      [[ -z "$normal_result" ]] && continue
      # The current result is appended exactly once below, even if a caller
      # accidentally includes that path in GROUNDING_NORMAL_RESULTS.
      [[ "$normal_result" == "$OUTPUT_DIR/grounding_normal.json" ]] && continue
      normal_result_args+=("$normal_result")
    done
    normal_result_args+=("$OUTPUT_DIR/grounding_normal.json")
    shared_id_builder_args=(
      --normal_results "${normal_result_args[@]}"
      --output "$OUTPUT_DIR/grounding_shared_ids.txt"
      --expected_count 362
    )
    if [[ -n "${GROUNDING_EXPECTED_MODELS:-}" ]]; then
      shared_id_builder_args+=(--expected_models "$GROUNDING_EXPECTED_MODELS")
    fi
    "$PYTHON" evaluation/build_shared_correct_ids.py \
      "${shared_id_builder_args[@]}"
    GROUNDING_SHARED_IDS="$OUTPUT_DIR/grounding_shared_ids.txt"
  fi

  for mode in ground_truth_mask background_mask; do
    "$PYTHON" evaluation/evaluate_visual_perturbations_qwen3vl.py \
      --manifest "$GROUNDING_MANIFEST" \
      --adapter_path "$ADAPTER_PATH" \
      --base_model_path "$BASE_MODEL_PATH" \
      --mode "$mode" \
      --output "$OUTPUT_DIR/perturbation_${mode}.json" \
      --image_max_pixels "$IMAGE_MAX_PIXELS" \
      --image_min_pixels "$IMAGE_MIN_PIXELS" \
      "${grounding_free_form_args[@]}" \
      --allow_nonpaper_sample_size \
      --max_samples 0
  done
  "$PYTHON" evaluation/aggregate_perturbation_results.py \
    --condition_a "$OUTPUT_DIR/perturbation_ground_truth_mask.json" \
    --condition_b "$OUTPUT_DIR/perturbation_background_mask.json" \
    --difference b_minus_a \
    --shared_ids "$GROUNDING_SHARED_IDS" \
    --expected_shared_count 362 \
    --output "$OUTPUT_DIR/evidence_region_gap.json"
fi

if [[ -n "${DEPENDENCE_MANIFEST:-}" ]]; then
  "$PYTHON" evaluation/evaluate_visual_dependence_qwen3vl.py \
    --manifest "$DEPENDENCE_MANIFEST" \
    --base_model_path "$BASE_MODEL_PATH" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    --output "$OUTPUT_DIR/pre_finetuning_visual_dependence.json" \
    --max_samples 0
fi
