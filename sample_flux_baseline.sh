#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

SAMPLER="${SAMPLER:-flux}"
MODEL_ID="${MODEL_ID:-${FLUX_MODEL_PATH:-}}"
PROMPT_FILE="${PROMPT_FILE:-${EPD_PROMPTS_TXT_DEFAULT}}"
SEEDS="${SEEDS:-0-3}"
OUTDIR="${OUTDIR:-samples/flux_baseline}"

if [[ -z "${MODEL_ID}" ]]; then
  MODEL_ID="$(resolve_local_flux_snapshot || true)"
fi
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
run_flux_runtime_preflight "${MODEL_ID}"

python sample_flux_baseline.py \
  --sampler "${SAMPLER}" \
  --model-id "${MODEL_ID}" \
  --prompt-file "${PROMPT_FILE}" \
  --seeds "${SEEDS}" \
  --batch 1 \
  --outdir "${OUTDIR}"
