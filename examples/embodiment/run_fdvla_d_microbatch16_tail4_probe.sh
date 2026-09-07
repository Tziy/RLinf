#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
RUN_ID=${FDVLA_MICRO16_RUN_ID:-20260904_task0_k2_sem8_bf16_lr5e7_kl0_tail4_micro16_10u_v1}
RUN_DIR=${FDVLA_MICRO16_RUN_DIR:-${REPO_DIR}/logs/fdvla_ppo_stability/${RUN_ID}/D-PPO}
MODEL_PATH=${FDVLA_MICRO16_MODEL_PATH:-${REPO_DIR}/logs/fdvla_d_sft_recalibration/20260902_seed26026_from_weak1000_50x10_v1/sft_grid/D/libero10_decoupled_delay0to8_s4_a4_grid50/checkpoint-40}
BACKBONE_PATH=${GR00T_BACKBONE_PATH:-${REPO_DIR}/../.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}

if [[ -e "${RUN_DIR}/run.env" || -e "${RUN_DIR}/launcher.log" ]]; then
    echo "Refusing to overwrite an existing micro-batch probe: ${RUN_DIR}" >&2
    exit 2
fi
for required in "${MODEL_PATH}" "${BACKBONE_PATH}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required checkpoint: ${required}" >&2
        exit 2
    fi
done
mkdir -p "${RUN_DIR}"

export GR00T_MODEL_PATH=${MODEL_PATH}
export GR00T_BACKBONE_PATH=${BACKBONE_PATH}
unset PPO_RESUME_DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ALLOWED_GPU_IDS=0,1,2,3
export SEMANTIC_GPU_IDS=0,1,2,3
export DIT_GPU_IDS=0,1,2,3
export ACTOR_GPU_IDS=0,1,2,3
export ACTOR_GPU_PLACEMENT=0-3
export ROLLOUT_GPU_PLACEMENT=0-3
export ENV_GPU_PLACEMENT=0-3
export DIT_REPLICAS_PER_GPU=1
export ENV_WORKERS_PER_GPU=1
export SEMANTIC_BATCH_MAX_REQUESTS=1
export SEMANTIC_BATCH_TARGET_REQUESTS=1
export SEMANTIC_BATCH_TARGET_ENVS=0
export SEMANTIC_BATCH_WAIT_MS=0
export SEMANTIC_BOOTSTRAP_TARGET_ENVS=0
export SEMANTIC_BOOTSTRAP_WAIT_MS=30000
export SEMANTIC_PREPROCESS_PROXY=false
export SEMANTIC_PREPROCESS_WORKERS=12
export SEMANTIC_OMP_NUM_THREADS=1
export SEMANTIC_RPC_BATCH_WAIT_MS=2
export COLOCATED_SEMANTIC_FETCH_PAUSE_MS=0

export TRAIN_TASK_ID_FILTER='[0]'
export EVAL_TASK_ID_FILTER='[0]'
export BALANCE_TRAIN_TASK_ASSIGNMENT=true
export PRESERVE_TRAIN_TASK_ASSIGNMENT=true
export UNIQUE_TRAIN_TRIAL_ASSIGNMENT=true
export TRAIN_NUM_ENVS=60
export EVAL_NUM_ENVS=48
export TARGET_TRAIN_ENVS=60
export TARGET_EVAL_ENVS=48
export TRAIN_AUTO_RESET=true
export EVAL_AUTO_RESET=false
export TRAIN_MAX_EPISODE_STEPS=480
export TRAIN_ROLLOUT_STEPS=256
export EVAL_MAX_EPISODE_STEPS=480
export EVAL_ROLLOUT_STEPS=480
export TRAIN_ROLLOUT_EPOCH=4
export TRAIN_SEED=0
export EVAL_SEED=26026
export EVAL_NOISE_SEED=26026
export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false

export ACTION_CHUNK_SIZE=16
export TRAIN_EXECUTION_HORIZON=2
export EVAL_EXECUTION_HORIZON=2
export DENOISING_STEPS=4
export SEMANTIC_PUBLISH_INTERVAL_FRAMES=8
export SEMANTIC_MID_CHUNK_PUBLISH=false
export SEMANTIC_MID_CHUNK_FRAME=4
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=-1
export SEMANTIC_FETCH_TARGET_AGE_FRAMES=-1
export SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES=-1
export SEMANTIC_TEXT_PADDING_TOKENS=160

export PPO_ACTOR_LR=5.0e-7
export PPO_VALUE_LR=2.0e-5
export PPO_KL_BETA=0.0
export FDVLA_ACTOR_MODEL_PRECISION=bf16
export FDVLA_ROLLOUT_MODEL_PRECISION=bf16
export DIT_TRAIN_LAST_N_BLOCKS=4
export SEMANTIC_ADAPTER_ONLY_TRAIN=false
export TRAIN_VALUE_HEAD_WITH_DIT_ONLY=true
export PPO_GLOBAL_BATCH_SIZE=960
export PPO_MICRO_BATCH_SIZE=16
export PPO_UPDATE_EPOCHS=1
export PPO_GAMMA=0.99
export PPO_GAE_LAMBDA=0.95
export PPO_MAX_STEPS=10
export PPO_EVAL_BEFORE_TRAINING=true
export PPO_VAL_INTERVAL=5
export PPO_SAVE_INTERVAL=5
export PPO_SUCCESS_EARLY_STOP_ENABLED=false

export FDVLA_RUN_TIMESTAMP=${RUN_ID}_D-PPO
export FDVLA_RUN_LOG_DIR=${RUN_DIR}
export RLINF_LOG_DIR=${RUN_DIR}
export FDVLA_METADATA_DIR=${RUN_DIR}/fdvla_metadata
export FDVLA_METHOD_ID=D-PPO
export PPO_EXPERIMENT_NAME=fdvla_k2_sem8_bf16_lr5e7_kl0_tail4_micro16_10u_D-PPO
export RAY_TMPDIR=${FDVLA_MICRO16_RAY_TMPDIR:-/dev/shm/fdm16v1}

printf 'start_utc=%s\nsource_checkpoint=%s\ntarget_global_step=10\nppo_micro_batch_size=16\ndit_train_last_n_blocks=4\ncomparison_run=%s\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${GR00T_MODEL_PATH}" \
    "${REPO_DIR}/logs/fdvla_ppo_stability/20260903_task0_k2_sem8_bf16_lr5e7_kl0_tail4_10u_v1/D-PPO" \
    >"${RUN_DIR}/run.env"
start_ns=$(date +%s%N)
bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh" >"${RUN_DIR}/launcher.log" 2>&1
end_ns=$(date +%s%N)
awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
    'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
    >"${RUN_DIR}/process_time.txt"
printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >>"${RUN_DIR}/run.env"
python "${REPO_DIR}/examples/analysis/summarize_fdvla_training_run.py" \
    --run-dir "${RUN_DIR}" \
    --run-log "${RUN_DIR}/launcher.log" \
    --output-dir "${RUN_DIR}/summary"
