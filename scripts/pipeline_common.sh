#!/usr/bin/env bash

if [[ -n "${EPD_PIPELINE_COMMON_SH:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi
EPD_PIPELINE_COMMON_SH=1

EPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPD_PROMPTS_TXT_DEFAULT="${EPD_ROOT}/src/prompts/test.txt"
EPD_PROMPTS_CSV_DEFAULT="${EPD_ROOT}/src/prompts/MS-COCO_val2014_30k_captions.csv"
EPD_RESULTS_DIR_DEFAULT="${EPD_ROOT}/results"
EPD_HF_HOME_DEFAULT="${EPD_ROOT}/weights/hf_cache"

cd "${EPD_ROOT}" || exit 1

export HF_HOME="${HF_HOME:-${EPD_HF_HOME_DEFAULT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

resolve_existing_path() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "${candidate:-}" && -e "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

resolve_local_flux_snapshot() {
    local candidate
    for candidate in \
        "${HUGGINGFACE_HUB_CACHE}/models--black-forest-labs--FLUX.1-dev/snapshots/"* \
        "${HOME}/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/"*; do
        if [[ -f "${candidate}/model_index.json" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "[EPD] Missing file: ${path}" >&2
        return 1
    fi
}

require_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "[EPD] Missing directory: ${path}" >&2
        return 1
    fi
}

latest_run_dir() {
    local suffix="$1"
    find "${EPD_ROOT}/exps" -maxdepth 1 -mindepth 1 -type d -name "*-${suffix}" | sort | tail -n 1
}

latest_policy_checkpoint() {
    local run_dir="$1"
    find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'policy-step*.pt' | sort | tail -n 1
}

resolve_policy_checkpoint() {
    local run_dir="$1"
    local checkpoint="${2:-}"
    if [[ -n "${checkpoint}" ]]; then
        if [[ "${checkpoint}" = /* ]]; then
            printf '%s\n' "${checkpoint}"
        else
            printf '%s/%s\n' "${run_dir}" "${checkpoint}"
        fi
        return 0
    fi
    latest_policy_checkpoint "${run_dir}"
}

resolve_export_predictor() {
    local run_dir="$1"
    local step="${2:-}"
    local digits
    local candidate
    if [[ -n "${step}" ]]; then
        digits="$(printf '%06d' "${step}")"
        candidate="${run_dir}/export/network-snapshot-export-step${digits}.pkl"
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi
    find "${run_dir}/export" -maxdepth 1 -type f -name 'network-snapshot-export-*.pkl' | sort | tail -n 1
}

run_export_predictor() {
    local run_dir="$1"
    local checkpoint="${2:-}"
    require_dir "${run_dir}" || return 1
    if [[ -n "${checkpoint}" ]]; then
        checkpoint="$(resolve_policy_checkpoint "${run_dir}" "${checkpoint}")"
        require_file "${checkpoint}" || return 1
        python -m training.ppo.export_epd_predictor "${run_dir}" --checkpoint "${checkpoint#${run_dir}/}"
    else
        python -m training.ppo.export_epd_predictor "${run_dir}"
    fi
}

prepare_prompt_subset_file() {
    local prompt_file="$1"
    local output_file="$2"
    local seeds="${3:-}"
    local count="${4:-}"

    if [[ -z "${prompt_file}" || -z "${output_file}" ]]; then
        echo "Usage: prepare_prompt_subset_file <prompt_file> <output_file> [seeds] [count]" >&2
        return 1
    fi
    if [[ -n "${seeds}" ]]; then
        python -m training.ppo.scripts.prepare_prompt_subset \
            --prompts "${prompt_file}" \
            --output "${output_file}" \
            --seeds "${seeds}"
    else
        python -m training.ppo.scripts.prepare_prompt_subset \
            --prompts "${prompt_file}" \
            --output "${output_file}" \
            --count "${count}"
    fi
}

run_flux_runtime_preflight() {
    local model_id="${1:-}"
    local predictor="${2:-}"
    local extra=()
    if [[ -n "${model_id}" ]]; then
        extra+=(--model-id "${model_id}")
    fi
    if [[ -n "${predictor}" ]]; then
        extra+=(--predictor "${predictor}")
    fi
    if [[ "${FLUX_ALLOW_REMOTE:-0}" == "1" ]]; then
        extra+=(--allow-remote)
    fi
    python -m training.ppo.scripts.check_flux_runtime "${extra[@]}"
}

score_all_metrics_dir() {
    local image_dir="$1"
    local prompt_file="$2"
    local prefix="$3"
    local results_dir="${4:-${EPD_RESULTS_DIR_DEFAULT}}"
    if [[ -z "${image_dir}" || -z "${prompt_file}" || -z "${prefix}" ]]; then
        echo "Usage: score_all_metrics_dir <image_dir> <prompt_file> <result_prefix> [results_dir]" >&2
        return 1
    fi

    local clip_weights="${EPD_ROOT}/weights/clip"
    local hps_weights
    local aesthetic_weights
    local pickscore_weights
    local imagereward_weights
    local mps_weights

    require_dir "${image_dir}" || return 1
    require_file "${prompt_file}" || return 1

    hps_weights="$(resolve_existing_path "${EPD_ROOT}/weights/HPS_v2.1_compressed.pt")"
    aesthetic_weights="$(resolve_existing_path "${EPD_ROOT}/weights/sac+logos+ava1-l14-linearMSE.pth")"
    mps_weights="$(resolve_existing_path "${EPD_ROOT}/weights/MPS_overall_checkpoint.pth")"
    pickscore_weights="$(resolve_existing_path "${EPD_ROOT}/weights/PickScore_v1" || true)"
    imagereward_weights="$(resolve_existing_path "${EPD_ROOT}/weights/ImageReward.pt" "${EPD_ROOT}/weights/ImageReward-v1.0.pt" || true)"

    require_file "${hps_weights}" || return 1
    require_file "${aesthetic_weights}" || return 1
    require_file "${mps_weights}" || return 1
    mkdir -p "${results_dir}"

    if [[ -z "${pickscore_weights}" ]]; then
        pickscore_weights="${EPD_ROOT}/weights/PickScore_v1"
    fi
    if [[ -z "${imagereward_weights}" ]]; then
        imagereward_weights="${EPD_ROOT}/weights/ImageReward.pt"
    fi

    python -m training.ppo.scripts.score_clip \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${clip_weights}" \
        --output-json "${results_dir}/${prefix}_clip.json"

    python -m training.ppo.scripts.score_hps \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${hps_weights}" \
        --output-json "${results_dir}/${prefix}_hps.json"

    python -m training.ppo.scripts.score_aesthetic \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${aesthetic_weights}" \
        --output-json "${results_dir}/${prefix}_aesthetic.json"

    if [[ ! -d "${EPD_ROOT}/weights/PickScore_v1" ]]; then
        echo "[pipeline_common.sh] Local weights/PickScore_v1 is missing for ${prefix}; falling back to Hugging Face download." >&2
    fi
    python -m training.ppo.scripts.score_pick \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${pickscore_weights}" \
        --output-json "${results_dir}/${prefix}_pick.json"

    python -m training.ppo.scripts.score_imagereward \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${imagereward_weights}" \
        --output-json "${results_dir}/${prefix}_imagereward.json"

    python -m training.ppo.scripts.score_mps \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts "${prompt_file}" \
        --weights "${mps_weights}" \
        --output-json "${results_dir}/${prefix}_mps.json"
}

score_fid_dir() {
    local image_dir="$1"
    local manifest_csv="$2"
    local real_dir="$3"
    local prefix="$4"
    local results_dir="${5:-${EPD_RESULTS_DIR_DEFAULT}}"
    local fid_cache_dir="${EPD_RESULTS_DIR_DEFAULT}/fid_cache"
    if [[ -z "${image_dir}" || -z "${manifest_csv}" || -z "${real_dir}" || -z "${prefix}" ]]; then
        echo "Usage: score_fid_dir <image_dir> <manifest_csv> <real_dir> <result_prefix> [results_dir]" >&2
        return 1
    fi

    require_dir "${image_dir}" || return 1
    require_file "${manifest_csv}" || return 1
    require_dir "${real_dir}" || return 1
    mkdir -p "${results_dir}" "${fid_cache_dir}"

    python -m training.ppo.scripts.score_fid_dir \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --manifest "${manifest_csv}" \
        --real-images "${real_dir}" \
        --cache-dir "${fid_cache_dir}" \
        --mode clean \
        --eval-res 256 \
        --num-workers 2 \
        --output-json "${results_dir}/${prefix}_fid.json"
}

score_fid_npz_dir() {
    local image_dir="$1"
    local ref_npz="$2"
    local prefix="$3"
    local results_dir="${4:-${EPD_RESULTS_DIR_DEFAULT}}"
    if [[ -z "${image_dir}" || -z "${ref_npz}" || -z "${prefix}" ]]; then
        echo "Usage: score_fid_npz_dir <image_dir> <ref_npz> <result_prefix> [results_dir]" >&2
        return 1
    fi

    require_dir "${image_dir}" || return 1
    require_file "${ref_npz}" || return 1
    mkdir -p "${results_dir}"

    python -m training.ppo.scripts.score_fid_npz_dir \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --ref-npz "${ref_npz}" \
        --num-expected 10000 \
        --batch-size 64 \
        --num-workers 2 \
        --device cuda \
        --output-json "${results_dir}/${prefix}_legacy_fid.json"
}

score_all_metrics() {
    local name="$1"
    local prompt_file="${2:-${EPD_PROMPTS_TXT_DEFAULT}}"
    if [[ -z "${name}" ]]; then
        echo "Usage: score_all_metrics <images_subdir_under_samples> [prompt_file]" >&2
        return 1
    fi
    score_all_metrics_dir "${EPD_ROOT}/samples/${name}" "${prompt_file}" "${name}" "${EPD_RESULTS_DIR_DEFAULT}"
}
