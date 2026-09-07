#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the common initial checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the common frozen backbone}"

BENCHMARK_ID=${FDVLA_SPEED_BENCHMARK_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
BENCHMARK_ROOT=${FDVLA_SPEED_BENCHMARK_ROOT:-"${REPO_DIR}/logs/fdvla_minimal_speed/${BENCHMARK_ID}"}
mkdir -p "${BENCHMARK_ROOT}"

# The benchmark measures PPO throughput, not policy quality. Both arms process
# the same Task-0 control frames and optimizer batches. Five updates leave one
# warm-up update and three steady-state samples while keeping the run short.
export TRAIN_TASK_ID_FILTER=${TRAIN_TASK_ID_FILTER:-[0]}
export EVAL_TASK_ID_FILTER=${EVAL_TASK_ID_FILTER:-[0]}
export BALANCE_TRAIN_TASK_ASSIGNMENT=false
export PRESERVE_TRAIN_TASK_ASSIGNMENT=true
export TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS:-12}
export EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-12}
export TRAIN_MAX_EPISODE_STEPS=${TRAIN_MAX_EPISODE_STEPS:-64}
export TRAIN_ROLLOUT_STEPS=${TRAIN_ROLLOUT_STEPS:-64}
export EVAL_MAX_EPISODE_STEPS=${EVAL_MAX_EPISODE_STEPS:-64}
export EVAL_ROLLOUT_STEPS=${EVAL_ROLLOUT_STEPS:-64}
export TRAIN_ROLLOUT_EPOCH=${TRAIN_ROLLOUT_EPOCH:-1}
export PPO_MAX_STEPS=${PPO_MAX_STEPS:-5}
export PPO_GLOBAL_BATCH_SIZE=${PPO_GLOBAL_BATCH_SIZE:-24}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
export PPO_UPDATE_EPOCHS=${PPO_UPDATE_EPOCHS:-1}
export PPO_EVAL_BEFORE_TRAINING=false
export PPO_VAL_INTERVAL=1000
export PPO_SAVE_INTERVAL=-1
export PPO_SUCCESS_EARLY_STOP_ENABLED=false
export ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-4}
export TRAIN_SEED=${TRAIN_SEED:-0}
export EVAL_SEED=${EVAL_SEED:-0}
export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rf}

run_arm() {
    local arm=$1
    local launcher=$2
    local run_dir="${BENCHMARK_ROOT}/${arm}"

    mkdir -p "${run_dir}"
    export FDVLA_RUN_TIMESTAMP="${BENCHMARK_ID}_${arm}"
    export FDVLA_RUN_LOG_DIR="${run_dir}"
    export RLINF_LOG_DIR="${run_dir}"
    export FDVLA_METADATA_DIR="${run_dir}/fdvla_metadata"
    export PPO_EXPERIMENT_NAME="fdvla_minimal_speed_${arm}"
    export FDVLA_METHOD_ID="${arm}"

    printf 'benchmark_id=%s\narm=%s\nstart_utc=%s\n' \
        "${BENCHMARK_ID}" "${arm}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        >"${run_dir}/benchmark.env"
    local start_ns end_ns
    start_ns=$(date +%s%N)
    bash "${launcher}" >"${run_dir}/launcher.log" 2>&1
    end_ns=$(date +%s%N)
    awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
        'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
        >"${run_dir}/process_time.txt"
    printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        >>"${run_dir}/benchmark.env"

    python "${REPO_DIR}/examples/analysis/summarize_fdvla_training_run.py" \
        --run-dir "${run_dir}" \
        --run-log "${run_dir}/launcher.log" \
        --output-dir "${run_dir}/summary"
}

# Equal total hardware budget: C-PPO uses all four GPUs for the coupled model;
# D-PPO reserves GPU 0 for the frozen semantic server and trains on GPUs 1-3.
# The order is reversed relative to the initial short calibration to reduce
# systematic page-cache and launch-order bias.
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

printf '%s\n' "${BENCHMARK_ROOT}"
