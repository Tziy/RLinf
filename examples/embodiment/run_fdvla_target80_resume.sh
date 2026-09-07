#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the preregistered weak1000 checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the frozen backbone}"
: "${PPO_RESUME_DIR:?Set PPO_RESUME_DIR to the preregistered D-PPO global_step_50}"

REGISTERED_MODEL_PATH=/vepfs-mlp2/c20250301/240403026/async_vla/async_libero_runs/libero10_decoupled_sft_0to8_rebuilt/W1000_libero10_decoupled_0to8_scalar_age_s4_a4/libero10_delay0to8_s4_a4_weak1000/checkpoint-1000
REGISTERED_PARENT_PATH=${REPO_DIR}/logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/fdvla_binary_d_ppo_weak1000_task0_seed0/checkpoints/global_step_50
REGISTERED_MODEL_SHA256=edff533a5c7b9472739577f0b4de8c0bb39f6a93815aad30ec40784538d1adbf
REGISTERED_PARENT_ACTOR_SHA256=ade2858d1bf13e062947e227d00d12159c4dbdcc08b1464f7d087497cbafca8f

if [[ "${GR00T_MODEL_PATH}" != "${REGISTERED_MODEL_PATH}" ]]; then
    echo "GR00T_MODEL_PATH differs from the preregistered weak1000 checkpoint." >&2
    exit 2
fi
if [[ "${PPO_RESUME_DIR}" != "${REGISTERED_PARENT_PATH}" ]]; then
    echo "PPO_RESUME_DIR differs from the preregistered global_step_50 checkpoint." >&2
    exit 2
fi
parent_actor=${PPO_RESUME_DIR}/actor/model_state_dict/full_weights.pt
observed_parent_sha=$(sha256sum "${parent_actor}" | awk '{print $1}')
if [[ "${observed_parent_sha}" != "${REGISTERED_PARENT_ACTOR_SHA256}" ]]; then
    echo "Preregistered parent actor SHA256 mismatch: ${observed_parent_sha}" >&2
    exit 2
fi
export FDVLA_RESOLVED_MODEL_SHA256=${REGISTERED_MODEL_SHA256}
export FDVLA_RESOLVED_RESUME_ACTOR_SHA256=${REGISTERED_PARENT_ACTOR_SHA256}

PLACEMENT=${1:-actor4_rollout3}
case "${PLACEMENT}" in
    actor4_rollout3)
        export SEMANTIC_GPU_IDS=0
        export DIT_GPU_IDS=1,2,3
        export ACTOR_GPU_IDS=0,1,2,3
        ;;
    colocated4)
        export SEMANTIC_GPU_IDS=0
        export DIT_GPU_IDS=0,1,2,3
        export ACTOR_GPU_IDS=0,1,2,3
        export COLOCATED_SEMANTIC_FETCH_PAUSE_MS=${COLOCATED_SEMANTIC_FETCH_PAUSE_MS:-0}
        ;;
    *)
        echo "Unknown placement: ${PLACEMENT}" >&2
        exit 2
        ;;
esac
export ALLOWED_GPU_IDS=0,1,2,3

# Locked development continuation protocol from fdvla_binary_run_manifest.yaml.
export TRAIN_TASK_ID_FILTER='[0]'
export EVAL_TASK_ID_FILTER='[0]'
export BALANCE_TRAIN_TASK_ASSIGNMENT=true
export PRESERVE_TRAIN_TASK_ASSIGNMENT=true
export TRAIN_NUM_ENVS=60
export EVAL_NUM_ENVS=48
export TARGET_TRAIN_ENVS=60
export TARGET_EVAL_ENVS=48
export TRAIN_MAX_EPISODE_STEPS=480
export TRAIN_ROLLOUT_STEPS=256
export EVAL_MAX_EPISODE_STEPS=480
export EVAL_ROLLOUT_STEPS=480
export TRAIN_ROLLOUT_EPOCH=4
export PPO_GLOBAL_BATCH_SIZE=384
export PPO_MICRO_BATCH_SIZE=2
export PPO_UPDATE_EPOCHS=1
export PPO_MAX_STEPS=150
export PPO_EVAL_BEFORE_TRAINING=true
export PPO_VAL_INTERVAL=10
export PPO_SAVE_INTERVAL=10
export PPO_SUCCESS_EARLY_STOP_ENABLED=true
export PPO_SUCCESS_EARLY_STOP_METRIC=eval/success_once
export PPO_SUCCESS_EARLY_STOP_THRESHOLD=0.75
export PPO_SUCCESS_EARLY_STOP_BAND_LOW=0.75
export PPO_SUCCESS_EARLY_STOP_BAND_HIGH=0.85
export PPO_SUCCESS_EARLY_STOP_PROTOCOL_LABEL=weak1000_seed0_continue_from_step50_target80
export ACTION_CHUNK_SIZE=16
export EVAL_EXECUTION_HORIZON=8
export DENOISING_STEPS=4
export TRAIN_SEED=0
export EVAL_SEED=0
export EVAL_NOISE_SEED=13026
export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=${SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES:-0}
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=${SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES:-6}
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=6
export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_TEXT_PADDING_TOKENS=160

RUN_ID=${FDVLA_TARGET80_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_TARGET80_RUN_ROOT:-${REPO_DIR}/logs/fdvla_target80_resume/${RUN_ID}}
RUN_DIR=${RUN_ROOT}/${PLACEMENT}
mkdir -p "${RUN_DIR}"
export FDVLA_RUN_TIMESTAMP=${RUN_ID}_${PLACEMENT}
export FDVLA_RUN_LOG_DIR=${RUN_DIR}
export RLINF_LOG_DIR=${RUN_DIR}
export FDVLA_METADATA_DIR=${RUN_DIR}/fdvla_metadata
export FDVLA_METHOD_ID=D-PPO
export PPO_EXPERIMENT_NAME=fdvla_target80_${PLACEMENT}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/fdvla_target80_${PLACEMENT}}

printf 'run_id=%s\nplacement=%s\nstart_utc=%s\n' \
    "${RUN_ID}" "${PLACEMENT}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
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
printf '%s\n' "${RUN_DIR}"
