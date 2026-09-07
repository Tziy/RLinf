#!/usr/bin/env bash

set -euo pipefail

FDVLA_REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FDVLA_ARTIFACT_HASH_ALGORITHM="file-raw+dir-logical-deref-sha256-v1"
export EMBODIED_PATH="${FDVLA_REPO_PATH}/examples/embodiment"
export PYTHONPATH="$(dirname "${FDVLA_REPO_PATH}"):${FDVLA_REPO_PATH}:${PYTHONPATH:-}"

fdvla_require_inputs() {
    : "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the common initial checkpoint}"
    : "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the frozen GR00T backbone}"
    if [[ "${GR00T_MODEL_PATH}" != /* || "${GR00T_BACKBONE_PATH}" != /* ]]; then
        echo "GR00T_MODEL_PATH and GR00T_BACKBONE_PATH must be absolute." >&2
        return 2
    fi
}

fdvla_checkpoint_sha256() {
    local checkpoint_path=$1
    if [[ -f "${checkpoint_path}" ]]; then
        sha256sum "${checkpoint_path}" | awk '{print $1}'
        return
    fi
    if [[ ! -d "${checkpoint_path}" ]]; then
        echo "Checkpoint does not exist: ${checkpoint_path}" >&2
        return 2
    fi
    local broken_link
    broken_link=$(find "${checkpoint_path}" -xtype l -print -quit)
    if [[ -n "${broken_link}" ]]; then
        echo "Checkpoint contains a broken symlink: ${broken_link}" >&2
        return 2
    fi
    if ! find -L "${checkpoint_path}" -type f -printf '' >/dev/null; then
        echo "Checkpoint directory cannot be resolved completely: ${checkpoint_path}" >&2
        return 2
    fi

    local relative_files=()
    mapfile -d '' -t relative_files < <(
        find -L "${checkpoint_path}" -type f -printf '%P\0' | LC_ALL=C sort -z
    )
    if ((${#relative_files[@]} == 0)); then
        echo "Checkpoint directory contains no resolved files: ${checkpoint_path}" >&2
        return 2
    fi

    # Bind both the logical checkpoint layout and dereferenced file contents.
    # Relative paths make identical snapshots hash equally after relocation;
    # dereferencing is required because Hugging Face snapshots are symlink farms.
    {
        local relative_path file_sha
        for relative_path in "${relative_files[@]}"; do
            file_sha=$(sha256sum -- "${checkpoint_path}/${relative_path}" | awk '{print $1}')
            printf 'path:%s\0sha256:%s\n' "${relative_path}" "${file_sha}"
        done
    } | sha256sum | awk '{print $1}'
}

fdvla_write_resolved_sha256() {
    local output_path=$1
    local artifact_path=$2
    local resolved_sha=${3:-}
    if [[ -n "${resolved_sha}" ]]; then
        if [[ ! "${resolved_sha}" =~ ^[0-9a-f]{64}$ ]]; then
            echo "Invalid precomputed SHA256 for ${artifact_path}: ${resolved_sha}" >&2
            return 2
        fi
        printf '%s\n' "${resolved_sha}" >"${output_path}"
        return
    fi
    fdvla_checkpoint_sha256 "${artifact_path}" >"${output_path}"
}

fdvla_worktree_sha256() {
    local relative_path
    {
        git -C "${FDVLA_REPO_PATH}" diff --binary HEAD -- rlinf examples tests docs
        while IFS= read -r relative_path; do
            printf 'untracked:%s\n' "${relative_path}"
            sha256sum "${FDVLA_REPO_PATH}/${relative_path}"
        done < <(
            git -C "${FDVLA_REPO_PATH}" ls-files --others --exclude-standard \
                -- rlinf examples tests docs | LC_ALL=C sort
        )
    } | sha256sum | awk '{print $1}'
}

fdvla_prepare_run_paths() {
    local config_name=$1
    local method_id=$2
    local timestamp
    timestamp=${FDVLA_RUN_TIMESTAMP:-$(date -u +'%Y%m%d-%H:%M:%S')}
    export FDVLA_RUN_LOG_DIR=${FDVLA_RUN_LOG_DIR:-"${FDVLA_REPO_PATH}/logs/${timestamp}-${config_name}"}
    export RLINF_LOG_DIR=${RLINF_LOG_DIR:-${FDVLA_RUN_LOG_DIR}}
    export FDVLA_METADATA_DIR=${FDVLA_METADATA_DIR:-"${FDVLA_RUN_LOG_DIR}/fdvla_metadata_${method_id}"}
    mkdir -p "${FDVLA_RUN_LOG_DIR}" "${FDVLA_METADATA_DIR}"
}

fdvla_write_run_metadata() {
    local config_name=$1
    local method_id=$2
    local timestamp metadata_dir
    timestamp=$(date -u +'%Y%m%dT%H%M%SZ')
    metadata_dir=${FDVLA_METADATA_DIR:-"${FDVLA_REPO_PATH}/logs/fdvla_metadata/${timestamp}_${method_id}"}
    mkdir -p "${metadata_dir}"

    git -C "${FDVLA_REPO_PATH}" rev-parse HEAD >"${metadata_dir}/git_sha.txt"
    git -C "${FDVLA_REPO_PATH}" status --short >"${metadata_dir}/git_status.txt"
    fdvla_worktree_sha256 >"${metadata_dir}/worktree_sha256.txt"
    printf '%s\n' "${FDVLA_ARTIFACT_HASH_ALGORITHM}" \
        >"${metadata_dir}/artifact_hash_algorithm.txt"
    printf '%s\n' "${GR00T_MODEL_PATH}" >"${metadata_dir}/checkpoint_path.txt"
    fdvla_write_resolved_sha256 \
        "${metadata_dir}/checkpoint_sha256.txt" \
        "${GR00T_MODEL_PATH}" \
        "${FDVLA_RESOLVED_MODEL_SHA256:-}"
    printf '%s\n' "${GR00T_BACKBONE_PATH}" >"${metadata_dir}/backbone_path.txt"
    fdvla_write_resolved_sha256 \
        "${metadata_dir}/backbone_sha256.txt" \
        "${GR00T_BACKBONE_PATH}" \
        "${FDVLA_RESOLVED_BACKBONE_SHA256:-}"
    if [[ -n "${PPO_CKPT_PATH:-}" ]]; then
        printf '%s\n' "${PPO_CKPT_PATH}" >"${metadata_dir}/ppo_checkpoint_path.txt"
        fdvla_write_resolved_sha256 \
            "${metadata_dir}/ppo_checkpoint_sha256.txt" \
            "${PPO_CKPT_PATH}" \
            "${FDVLA_RESOLVED_PPO_SHA256:-}"
    fi
    if [[ -n "${PPO_RESUME_DIR:-}" ]]; then
        if [[ "${PPO_RESUME_DIR}" != /* ]]; then
            echo "PPO_RESUME_DIR must be absolute: ${PPO_RESUME_DIR}" >&2
            return 2
        fi
        printf '%s\n' "${PPO_RESUME_DIR}" \
            >"${metadata_dir}/ppo_resume_checkpoint_path.txt"
        fdvla_write_resolved_sha256 \
            "${metadata_dir}/ppo_resume_checkpoint_sha256.txt" \
            "${PPO_RESUME_DIR}" \
            "${FDVLA_RESOLVED_RESUME_SHA256:-}"
        local resume_full_weights
        resume_full_weights="${PPO_RESUME_DIR}/actor/model_state_dict/full_weights.pt"
        if [[ -f "${resume_full_weights}" ]]; then
            printf '%s\n' "${resume_full_weights}" \
                >"${metadata_dir}/ppo_resume_actor_full_weights_path.txt"
            fdvla_write_resolved_sha256 \
                "${metadata_dir}/ppo_resume_actor_full_weights_sha256.txt" \
                "${resume_full_weights}" \
                "${FDVLA_RESOLVED_RESUME_ACTOR_SHA256:-}"
        fi
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
            --format=csv,noheader >"${metadata_dir}/hardware.csv"
    fi
    python "${FDVLA_REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
        --config-name "${config_name}" --cfg job --resolve \
        >"${metadata_dir}/resolved_config.yaml"
    {
        printf 'semantic_eval_fixed_age_frames=%s\n' "${SEMANTIC_EVAL_FIXED_AGE_FRAMES:--1}"
        printf 'train_execution_horizon=%s\n' "${TRAIN_EXECUTION_HORIZON:--1}"
        printf 'eval_execution_horizon=%s\n' "${EVAL_EXECUTION_HORIZON:--1}"
        printf 'action_prediction_horizon=%s\n' "${ACTION_CHUNK_SIZE:-16}"
        printf 'control_hz=%s\n' "${FDVLA_CONTROL_HZ:-20}"
        printf 'policy_noise_seed=%s\n' "${EVAL_NOISE_SEED:-2026}"
    } >"${metadata_dir}/realtime_condition.env"
    printf '%s\n' "${metadata_dir}"
}

fdvla_apply_protocol_defaults() {
    export ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-16}
    export TRAIN_EXECUTION_HORIZON=${TRAIN_EXECUTION_HORIZON:--1}
    export EVAL_EXECUTION_HORIZON=${EVAL_EXECUTION_HORIZON:--1}
    export FDVLA_CONTROL_HZ=${FDVLA_CONTROL_HZ:-20}
    export DENOISING_STEPS=${DENOISING_STEPS:-4}
    export PPO_ACTOR_LR=${PPO_ACTOR_LR:-1.0e-7}
    export PPO_VALUE_LR=${PPO_VALUE_LR:-2.0e-5}
    export PPO_GLOBAL_BATCH_SIZE=${PPO_GLOBAL_BATCH_SIZE:-384}
    export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
    export PPO_UPDATE_EPOCHS=${PPO_UPDATE_EPOCHS:-1}
    export PPO_GAMMA=${PPO_GAMMA:-0.99}
    export PPO_GAE_LAMBDA=${PPO_GAE_LAMBDA:-0.95}
    export TRAIN_ROLLOUT_EPOCH=${TRAIN_ROLLOUT_EPOCH:-4}
    export TRAIN_ROLLOUT_STEPS=${TRAIN_ROLLOUT_STEPS:-256}
    export TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS:-60}
    export EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-48}
    export TRAIN_SEED=${TRAIN_SEED:-0}
    export EVAL_SEED=${EVAL_SEED:-0}
    export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=${SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES:-0}
    export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=${SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES:-6}
    export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=${SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES:-0}
    export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=${SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES:-6}
    export SEMANTIC_FETCH_TARGET_AGE_FRAMES=${SEMANTIC_FETCH_TARGET_AGE_FRAMES:--1}
    export SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES=${SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES:--1}
    export SEMANTIC_MID_CHUNK_MIN_FRAME=${SEMANTIC_MID_CHUNK_MIN_FRAME:-1}
    export SEMANTIC_PREPROCESS_PROXY=${SEMANTIC_PREPROCESS_PROXY:-false}
    export PPO_MAX_STEPS=${PPO_MAX_STEPS:-50}
    export PPO_VAL_INTERVAL=${PPO_VAL_INTERVAL:-10}
    export PPO_EVAL_BEFORE_TRAINING=${PPO_EVAL_BEFORE_TRAINING:-true}
    export PPO_GROUP_SIZE=1
}
