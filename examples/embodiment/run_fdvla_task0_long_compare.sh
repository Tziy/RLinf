#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the common initial checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the common frozen backbone}"

RUN_ID=${FDVLA_LONG_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_LONG_RUN_ROOT:-"${REPO_DIR}/logs/fdvla_task0_long_compare/${RUN_ID}"}
mkdir -p "${RUN_ROOT}"

# Match the established Task-0 weak1000 pilot protocol. Both methods process
# the same frame budget and PPO batches; only semantic execution/placement differs.
export TRAIN_TASK_ID_FILTER=${TRAIN_TASK_ID_FILTER:-[0]}
export EVAL_TASK_ID_FILTER=${EVAL_TASK_ID_FILTER:-[0]}
export BALANCE_TRAIN_TASK_ASSIGNMENT=${BALANCE_TRAIN_TASK_ASSIGNMENT:-true}
export PRESERVE_TRAIN_TASK_ASSIGNMENT=${PRESERVE_TRAIN_TASK_ASSIGNMENT:-true}
export TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS:-60}
export EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-48}
export TARGET_TRAIN_ENVS=${TARGET_TRAIN_ENVS:-60}
export TARGET_EVAL_ENVS=${TARGET_EVAL_ENVS:-48}
export TRAIN_MAX_EPISODE_STEPS=${TRAIN_MAX_EPISODE_STEPS:-480}
export TRAIN_ROLLOUT_STEPS=${TRAIN_ROLLOUT_STEPS:-256}
export EVAL_MAX_EPISODE_STEPS=${EVAL_MAX_EPISODE_STEPS:-480}
export EVAL_ROLLOUT_STEPS=${EVAL_ROLLOUT_STEPS:-480}
export TRAIN_ROLLOUT_EPOCH=${TRAIN_ROLLOUT_EPOCH:-4}
export PPO_MAX_STEPS=${PPO_MAX_STEPS:-50}
export PPO_GLOBAL_BATCH_SIZE=${PPO_GLOBAL_BATCH_SIZE:-384}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
export PPO_UPDATE_EPOCHS=${PPO_UPDATE_EPOCHS:-1}
export PPO_EVAL_BEFORE_TRAINING=${PPO_EVAL_BEFORE_TRAINING:-true}
export PPO_VAL_INTERVAL=${PPO_VAL_INTERVAL:-10}
export PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:-10}
export PPO_SUCCESS_EARLY_STOP_ENABLED=false
export ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-16}
export EVAL_EXECUTION_HORIZON=${EVAL_EXECUTION_HORIZON:-8}
export DENOISING_STEPS=${DENOISING_STEPS:-4}
export TRAIN_SEED=${TRAIN_SEED:-0}
export EVAL_SEED=${EVAL_SEED:-0}
export EVAL_NOISE_SEED=${EVAL_NOISE_SEED:-2026}
export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=${SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES:-0}
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=${SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES:-6}
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=${SEMANTIC_EVAL_FIXED_AGE_FRAMES:-6}
export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rf_long}

run_arm() {
    local arm=$1
    local launcher=$2
    local run_dir="${RUN_ROOT}/${arm}"

    mkdir -p "${run_dir}"
    export FDVLA_RUN_TIMESTAMP="${RUN_ID}_${arm}"
    export FDVLA_RUN_LOG_DIR="${run_dir}"
    export RLINF_LOG_DIR="${run_dir}"
    export FDVLA_METADATA_DIR="${run_dir}/fdvla_metadata"
    export PPO_EXPERIMENT_NAME="fdvla_task0_long_${arm}"
    export FDVLA_METHOD_ID="${arm}"

    printf 'run_id=%s\narm=%s\nstart_utc=%s\n' \
        "${RUN_ID}" "${arm}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        >"${run_dir}/run.env"
    local start_ns end_ns
    start_ns=$(date +%s%N)
    bash "${launcher}" >"${run_dir}/launcher.log" 2>&1
    end_ns=$(date +%s%N)
    awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
        'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
        >"${run_dir}/process_time.txt"
    printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >>"${run_dir}/run.env"

    python "${REPO_DIR}/examples/analysis/summarize_fdvla_training_run.py" \
        --run-dir "${run_dir}" \
        --run-log "${run_dir}/launcher.log" \
        --output-dir "${run_dir}/summary"
}

# Equal four-GPU budget. Run coupled first to reverse the order of the earlier
# calibration; the fresh decoupled arm follows with the same code and inputs.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ACTOR_GPU_PLACEMENT=0-3
export ROLLOUT_GPU_PLACEMENT=0-3
export ENV_GPU_PLACEMENT=0-3
run_arm C-PPO "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh"

export SEMANTIC_GPU_IDS=0
export DIT_GPU_IDS=1,2,3
export ACTOR_GPU_IDS=1,2,3
export ALLOWED_GPU_IDS=0,1,2,3
export ACTOR_GPU_PLACEMENT=1-3
export ROLLOUT_GPU_PLACEMENT=1-3
export ENV_GPU_PLACEMENT=1-3
run_arm D-PPO "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh"

printf '%s\n' "${RUN_ROOT}"
