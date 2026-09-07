#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
GR00T_REPO=$(cd "${REPO_DIR}/.." && pwd)

RUN_ID=${FDVLA_CLEAN_C80_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_CLEAN_C80_RUN_ROOT:-"${REPO_DIR}/logs/fdvla_coupled_clean80/${RUN_ID}"}
RUN_DIR=${RUN_ROOT}/C-PPO-clean

export GR00T_MODEL_PATH=${GR00T_MODEL_PATH:-"${GR00T_REPO}/checkpoints/GR00T-N1.7-LIBERO/libero_10"}
export GR00T_BACKBONE_PATH=${GR00T_BACKBONE_PATH:-"${GR00T_REPO}/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"}

for required_path in "${GR00T_MODEL_PATH}" "${GR00T_BACKBONE_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required input does not exist: ${required_path}" >&2
        exit 2
    fi
done

# Match the completed D-PPO Task-0 protocol. Each update processes
# 60 envs * 256 frames * 4 rollout epochs = 61,440 control frames.
export TRAIN_TASK_ID_FILTER='[0]'
export EVAL_TASK_ID_FILTER='[0]'
export TRAIN_NUM_ENVS=60
export EVAL_NUM_ENVS=48
export TRAIN_MAX_EPISODE_STEPS=256
export TRAIN_ROLLOUT_STEPS=256
export EVAL_MAX_EPISODE_STEPS=480
export EVAL_ROLLOUT_STEPS=480
export TRAIN_ROLLOUT_EPOCH=4
export BALANCE_TRAIN_TASK_ASSIGNMENT=false
export PRESERVE_TRAIN_TASK_ASSIGNMENT=true
export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false
export TRAIN_SEED=0
export EVAL_SEED=${EVAL_SEED:-2026}

export PPO_MAX_EPOCHS=80
export PPO_MAX_STEPS=80
export PPO_GLOBAL_BATCH_SIZE=960
export PPO_MICRO_BATCH_SIZE=2
export PPO_UPDATE_EPOCHS=2
export PPO_ACTOR_LR=5.0e-6
export PPO_VALUE_LR=1.0e-4
export PPO_CRITIC_WARMUP_STEPS=0
export PPO_GAMMA=0.99
export PPO_GAE_LAMBDA=0.95
export PPO_ACTION_NOISE_SCALE=0.0
export PPO_EVAL_BEFORE_TRAINING=true
export PPO_VAL_INTERVAL=10
export PPO_SAVE_INTERVAL=10
export PPO_SUCCESS_EARLY_STOP_ENABLED=false

export ACTION_CHUNK_SIZE=16
export TRAIN_EXECUTION_HORIZON=8
export EVAL_EXECUTION_HORIZON=8
export DENOISING_STEPS=4
export ROLLOUT_INFERENCE_MICRO_BATCH_SIZE=20
export DETERMINISTIC_EVAL_NOISE=true
export EVAL_NOISE_SEED=${EVAL_NOISE_SEED:-2026}
export VALUE_HEAD_INIT_SEED=0

# Clean native coupled GR00T: current-frame local VLM -> DiT. Do not construct
# any packet-age or recent-action semantic adapter. The VLM remains frozen;
# PPO trains the full DiT core and value head, matching D's common trainable set.
export REQUIRE_PACKET_AGE_INPUT=false
export INITIALIZE_PACKET_AGE_ADAPTER=false
export ACTION_HISTORY_LENGTH=0
export ZERO_INIT_NEW_DELAY_ADAPTERS=false
export SEMANTIC_TEXT_PADDING_TOKENS=570
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=-1
export POSTHOC_SEMANTIC_DELAY_ENABLED=false
export POSTHOC_SEMANTIC_DELAY_BANK_ENABLED=false

# Equal four-A100 budget. Coupled uses the best legal native placement: four
# local GR00T rollout replicas and four-way FSDP actor sharding.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ACTOR_GPU_PLACEMENT=0-3
export ROLLOUT_GPU_PLACEMENT=0-3
export ENV_GPU_PLACEMENT=0-3
export ACTOR_ENABLE_OFFLOAD=true
export PPO_CLEAR_MEMORY_AFTER_UPDATE=false
export DIT_LIMIT_ALL_GATHERS=true
export DIT_CPU_OFFLOAD=false
export FDVLA_ACTOR_MODEL_PRECISION=bf16
export FDVLA_ROLLOUT_MODEL_PRECISION=bf16
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/fc80}

export FDVLA_METHOD_ID=C-PPO-clean
export FDVLA_RUN_TIMESTAMP="${RUN_ID}_C-PPO-clean"
export FDVLA_RUN_LOG_DIR="${RUN_DIR}"
export RLINF_LOG_DIR="${RUN_DIR}"
export FDVLA_METADATA_DIR="${RUN_DIR}/fdvla_metadata"
export PPO_EXPERIMENT_NAME=fdvla_task0_coupled_clean80_seed0

mkdir -p "${RUN_DIR}"
{
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'arm=C-PPO-clean\n'
    printf 'initial_checkpoint=%s\n' "${GR00T_MODEL_PATH}"
    printf 'start_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'train_envs=60\n'
    printf 'eval_envs=48\n'
    printf 'frames_per_update=61440\n'
    printf 'max_updates=80\n'
    printf 'adapter_policy=packet_age_absent,action_history_absent\n'
} >"${RUN_DIR}/run.env"

start_ns=$(date +%s%N)
set +e
bash "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh" >"${RUN_DIR}/launcher.log" 2>&1
status=$?
set -e
end_ns=$(date +%s%N)
awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
    'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
    >"${RUN_DIR}/process_time.txt"
{
    printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'exit_status=%s\n' "${status}"
} >>"${RUN_DIR}/run.env"

if [[ -f "${REPO_DIR}/examples/analysis/summarize_fdvla_training_run.py" ]]; then
    python "${REPO_DIR}/examples/analysis/summarize_fdvla_training_run.py" \
        --run-dir "${RUN_DIR}" \
        --run-log "${RUN_DIR}/launcher.log" \
        --output-dir "${RUN_DIR}/summary" || true
fi

exit "${status}"
