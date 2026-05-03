#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/pipeline_common.sh"

log() {
  printf '\n[train_flux.sh] %s\n' "$*"
}

FLUX_SNAPSHOT_PATH="${FLUX_SNAPSHOT_PATH:-exps/flux/flux-start.pkl}"
FLUX_CONFIG="${FLUX_CONFIG:-training/ppo/cfgs/flux_dev.yaml}"
FLUX_RUN_NAME="${FLUX_RUN_NAME:-flux_dev}"
FLUX_EXPORT_AFTER="${FLUX_EXPORT_AFTER:-1}"
FLUX_EXPORT_CHECKPOINT="${FLUX_EXPORT_CHECKPOINT:-}"

FLUX_MODEL_REF="$(resolve_local_flux_snapshot || true)"

FLUX_MASTER_PORT="${FLUX_MASTER_PORT:-11111}"
FLUX_NPROC_PER_NODE="${FLUX_NPROC_PER_NODE:-1}"
FLUX_CUDA_LAUNCH_BLOCKING="${FLUX_CUDA_LAUNCH_BLOCKING:-1}"
FLUX_TORCH_SHOW_CPP_STACKTRACES="${FLUX_TORCH_SHOW_CPP_STACKTRACES:-1}"
FLUX_PROMPT_CSV="${FLUX_PROMPT_CSV:-}"
FLUX_MAX_STEPS="${FLUX_MAX_STEPS:-}"

log "Using FLUX model reference for preflight: ${FLUX_MODEL_REF:-predictor metadata}"
log "Stage 1: use start table -> ${FLUX_SNAPSHOT_PATH}"
require_file "${FLUX_SNAPSHOT_PATH}"

log "Stage 2: runtime preflight"
run_flux_runtime_preflight "${FLUX_MODEL_REF}" "${FLUX_SNAPSHOT_PATH}"

log "Stage 3: PPO launch"
if [[ "${FLUX_NPROC_PER_NODE}" == "1" ]]; then
  log "Single-GPU launch: python -m training.ppo.launch"
  launch_cmd=(
    python
    -m training.ppo.launch
    --config "${FLUX_CONFIG}"
    --override "run.run_name=${FLUX_RUN_NAME}"
    --override "data.predictor_snapshot=${FLUX_SNAPSHOT_PATH}"
  )
else
  log "Multi-GPU launch: torchrun --nproc_per_node=${FLUX_NPROC_PER_NODE}"
  launch_cmd=(
    torchrun
    --master_port="${FLUX_MASTER_PORT}"
    --nproc_per_node="${FLUX_NPROC_PER_NODE}"
    -m training.ppo.launch
    --config "${FLUX_CONFIG}"
    --override "run.run_name=${FLUX_RUN_NAME}"
    --override "data.predictor_snapshot=${FLUX_SNAPSHOT_PATH}"
  )
fi
if [[ -n "${FLUX_PROMPT_CSV}" ]]; then
  launch_cmd+=(--override "data.prompt_csv=${FLUX_PROMPT_CSV}")
fi
if [[ -n "${FLUX_MAX_STEPS}" ]]; then
  launch_cmd+=(--max-steps "${FLUX_MAX_STEPS}")
fi
CUDA_LAUNCH_BLOCKING="${FLUX_CUDA_LAUNCH_BLOCKING}" \
TORCH_SHOW_CPP_STACKTRACES="${FLUX_TORCH_SHOW_CPP_STACKTRACES}" \
"${launch_cmd[@]}"

if [[ "${FLUX_EXPORT_AFTER}" == "1" ]]; then
  FLUX_RUN_DIR="$(latest_run_dir "${FLUX_RUN_NAME}" || true)"
  if [[ -n "${FLUX_RUN_DIR}" ]]; then
    log "Stage 4: export predictor -> ${FLUX_RUN_DIR}"
    run_export_predictor "${FLUX_RUN_DIR}" "${FLUX_EXPORT_CHECKPOINT}"
    log "Latest run: ${FLUX_RUN_DIR}"
    log "Latest export: $(resolve_export_predictor "${FLUX_RUN_DIR}" || true)"
  else
    echo "[train_flux.sh] Skip export because no run dir matched '*-${FLUX_RUN_NAME}'." >&2
  fi
fi
