#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

MODE="${MODE:-epd}"  # epd | baseline | baseline_sweep
RUN_NAME="${RUN_NAME:-flux_dev}"
RUN_DIR="${RUN_DIR:-$(latest_run_dir "${RUN_NAME}" || true)}"
PREDICTOR="${PREDICTOR:-}"
PROMPT_FILE="${PROMPT_FILE:-${EPD_PROMPTS_TXT_DEFAULT}}"
# Keep FLUX sampling aligned with the older SD1.5 / SD3 workflows:
# the default test.txt contains 1000 prompts, so the default seed range is
# 0-999 to produce one image per prompt.
SEEDS="${SEEDS:-0-999}"
OUTDIR="${OUTDIR:-samples/flux}"

MODEL_ID="${MODEL_ID:-${FLUX_MODEL_PATH:-}}"
SAMPLER="${SAMPLER:-euler}"
NUM_STEPS="${NUM_STEPS:-}"
BASELINE_SOLVERS="${BASELINE_SOLVERS:-euler edm dpm2 ipndm}"
BASELINE_EULER_STEPS="${BASELINE_EULER_STEPS:-24}"
BASELINE_EDM_STEPS="${BASELINE_EDM_STEPS:-12}"
BASELINE_DPM2_STEPS="${BASELINE_DPM2_STEPS:-12}"
BASELINE_IPNDM_STEPS="${BASELINE_IPNDM_STEPS:-24}"

resolve_baseline_steps() {
  local sampler="$1"
  case "${sampler}" in
    euler|flux|flowmatch) echo "${BASELINE_EULER_STEPS}" ;;
    edm) echo "${BASELINE_EDM_STEPS}" ;;
    dpm2) echo "${BASELINE_DPM2_STEPS}" ;;
    ipndm) echo "${BASELINE_IPNDM_STEPS}" ;;
    *)
      if [[ -n "${NUM_STEPS}" ]]; then
        echo "${NUM_STEPS}"
      else
        echo "[sample_flux.sh] No default num_steps configured for sampler '${sampler}'." >&2
        return 1
      fi
      ;;
  esac
}

run_epd() {
  if [[ -z "${PREDICTOR}" && -n "${RUN_DIR}" ]]; then
    PREDICTOR="$(resolve_export_predictor "${RUN_DIR}" || true)"
  fi
  if [[ -z "${PREDICTOR}" ]]; then
    echo "[sample_flux.sh] No predictor resolved. Set RUN_DIR or PREDICTOR." >&2
    exit 1
  fi

  if [[ -z "${MODEL_ID}" ]]; then
    MODEL_ID="$(resolve_local_flux_snapshot || true)"
  fi
  MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
  run_flux_runtime_preflight "${MODEL_ID}" "${PREDICTOR}"

  python sample_flux.py \
    --predictor "${PREDICTOR}" \
    --model-id "${MODEL_ID}" \
    --prompt-file "${PROMPT_FILE}" \
    --seeds "${SEEDS}" \
    --max-batch-size 1 \
    --outdir "${OUTDIR}"
}

run_baseline_one() {
  local sampler="$1"
  local steps="$2"
  local sampler_outdir="$3"

  if [[ -z "${MODEL_ID}" ]]; then
    MODEL_ID="$(resolve_local_flux_snapshot || true)"
  fi
  MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
  run_flux_runtime_preflight "${MODEL_ID}"

  python sample_flux_baseline.py \
    --sampler "${sampler}" \
    --num-steps "${steps}" \
    --model-id "${MODEL_ID}" \
    --prompt-file "${PROMPT_FILE}" \
    --seeds "${SEEDS}" \
    --batch 1 \
    --outdir "${sampler_outdir}"
}

case "${MODE}" in
  epd)
    run_epd
    ;;
  baseline)
    if [[ -z "${NUM_STEPS}" ]]; then
      NUM_STEPS="$(resolve_baseline_steps "${SAMPLER}")"
    fi
    run_baseline_one "${SAMPLER}" "${NUM_STEPS}" "${OUTDIR}"
    ;;
  baseline_sweep)
    if [[ -z "${MODEL_ID}" ]]; then
      MODEL_ID="$(resolve_local_flux_snapshot || true)"
    fi
    MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
    run_flux_runtime_preflight "${MODEL_ID}"

    for sampler in ${BASELINE_SOLVERS}; do
      steps="$(resolve_baseline_steps "${sampler}")"
      sampler_outdir="${OUTDIR}_${sampler}"
      python sample_flux_baseline.py \
        --sampler "${sampler}" \
        --num-steps "${steps}" \
        --model-id "${MODEL_ID}" \
        --prompt-file "${PROMPT_FILE}" \
        --seeds "${SEEDS}" \
        --batch 1 \
        --outdir "${sampler_outdir}"
    done
    ;;
  *)
    echo "[sample_flux.sh] Unsupported MODE='${MODE}'. Use epd, baseline, or baseline_sweep." >&2
    exit 1
    ;;
esac
