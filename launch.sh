#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

PROMPT_FILE="${PROMPT_FILE:-${EPD_PROMPTS_TXT_DEFAULT}}"
SEEDS="${SEEDS:-0-999}"
SD15_RUN_NAME="${SD15_RUN_NAME:-sd15}"
SD3_1024_RUN_NAME="${SD3_1024_RUN_NAME:-sd3_1024}"
FLUX_RUN_NAME="${FLUX_RUN_NAME:-flux_dev}"

# RL Train
torchrun --master_port=23123 --nproc_per_node=1 -m training.ppo.launch --config training/ppo/cfgs/sd15.yaml
torchrun --master_port=22222 --nproc_per_node=1 -m training.ppo.launch --config training/ppo/cfgs/sd3_512.yaml
torchrun --master_port=12345 --nproc_per_node=1 -m training.ppo.launch --config training/ppo/cfgs/sd3_1024.yaml
./train_flux.sh

# Sample
SD15_RUN_DIR="${SD15_RUN_DIR:-$(latest_run_dir "${SD15_RUN_NAME}" || true)}"
SD15_PREDICTOR="${SD15_PREDICTOR:-}"
if [[ -z "${SD15_PREDICTOR}" && -n "${SD15_RUN_DIR}" ]]; then
  SD15_PREDICTOR="$(resolve_export_predictor "${SD15_RUN_DIR}" || true)"
fi
if [[ -n "${SD15_PREDICTOR}" ]]; then
  MASTER_PORT=12345 python sample.py \
    --predictor_path "${SD15_PREDICTOR}" \
    --prompt-file "${PROMPT_FILE}" \
    --seeds "${SEEDS}" \
    --batch 16 \
    --outdir samples/sd15
else
  echo "[launch.sh] Skip SD1.5 sample because no predictor was resolved." >&2
fi

SD3_1024_RUN_DIR="${SD3_1024_RUN_DIR:-$(latest_run_dir "${SD3_1024_RUN_NAME}" || true)}"
SD3_1024_PREDICTOR="${SD3_1024_PREDICTOR:-}"
if [[ -z "${SD3_1024_PREDICTOR}" && -n "${SD3_1024_RUN_DIR}" ]]; then
  SD3_1024_PREDICTOR="$(resolve_export_predictor "${SD3_1024_RUN_DIR}" || true)"
fi
if [[ -n "${SD3_1024_PREDICTOR}" ]]; then
  python sample_sd3.py \
    --predictor "${SD3_1024_PREDICTOR}" \
    --seeds "0" \
    --outdir output_images \
    --prompt "..."
else
  echo "[launch.sh] Skip SD3 sample because no predictor was resolved." >&2
fi

FLUX_RUN_DIR="${FLUX_RUN_DIR:-$(latest_run_dir "${FLUX_RUN_NAME}" || true)}"
FLUX_PREDICTOR="${FLUX_PREDICTOR:-}"
if [[ -z "${FLUX_PREDICTOR}" && -n "${FLUX_RUN_DIR}" ]]; then
  FLUX_PREDICTOR="$(resolve_export_predictor "${FLUX_RUN_DIR}" || true)"
fi
if [[ -n "${FLUX_PREDICTOR}" ]]; then
  python sample_flux.py \
    --predictor "${FLUX_PREDICTOR}" \
    --seeds "0" \
    --outdir output_images_flux \
    --prompt "..."
else
  echo "[launch.sh] Skip FLUX sample because no predictor was resolved." >&2
fi

# Evaluation
# score_all_metrics sd15 "${PROMPT_FILE}"
