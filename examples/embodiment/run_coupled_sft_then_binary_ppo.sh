#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLINF_REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
GR00T_REPO=$(cd "${RLINF_REPO}/.." && pwd)

# This is the matched initialization repair for the coupled arm.  It starts
# from the same official LIBERO-10 checkpoint as the decoupled weak1000 SFT,
# uses the same demonstrations and optimizer budget, and changes only the SFT
# temporal context: coupled always consumes the current frame (age zero).
RUN_ID=${FDVLA_C_SFT_PPO_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_C_SFT_PPO_RUN_ROOT:-${RLINF_REPO}/logs/fdvla_c_sft_then_ppo/${RUN_ID}}
RAY_TMP_ROOT=${C_SFT_PPO_RAY_TMPDIR:-/tmp/fcspp}

BASE_MODEL_PATH=${C_SFT_BASE_MODEL_PATH:-${GR00T_REPO}/checkpoints/GR00T-N1.7-LIBERO/libero_10}
BACKBONE_PATH=${GR00T_BACKBONE_PATH:-${GR00T_REPO}/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}
DATASET_PATH=${C_SFT_DATASET_PATH:-${GR00T_REPO}/examples/LIBERO/nvidia/LIBERO_LeRobot_v3/libero_10}

SFT_STEPS=${C_SFT_MAX_STEPS:-1000}
SFT_EXPERIMENT_NAME=${C_SFT_EXPERIMENT_NAME:-libero10_coupled_current_s4_a4_sft${SFT_STEPS}}
SFT_OUTPUT_ROOT=${C_SFT_OUTPUT_ROOT:-${RUN_ROOT}/sft}
SFT_CHECKPOINT=${SFT_OUTPUT_ROOT}/${SFT_EXPERIMENT_NAME}/checkpoint-${SFT_STEPS}

PPO_STEPS=${C_PPO_MAX_STEPS:-50}
PPO_EXPERIMENT_NAME_FINAL=${C_PPO_EXPERIMENT_NAME:-fdvla_c_sft${SFT_STEPS}_binary_ppo_seed0}
PPO_RUN_DIR=${RUN_ROOT}/ppo
PPO_CHECKPOINT=${PPO_RUN_DIR}/${PPO_EXPERIMENT_NAME_FINAL}/checkpoints/global_step_${PPO_STEPS}/actor/model_state_dict/full_weights.pt

for required_path in "${BASE_MODEL_PATH}" "${BACKBONE_PATH}" "${DATASET_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required input does not exist: ${required_path}" >&2
        exit 2
    fi
done
if [[ "${RUN_ROOT}" != /* ]]; then
    echo "RUN_ROOT must be absolute: ${RUN_ROOT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"

{
    printf 'protocol_label=matched_coupled_current_frame_sft_then_binary_ppo\n'
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'git_sha=%s\n' "$(git -C "${RLINF_REPO}" rev-parse HEAD)"
    printf 'base_model=%s\n' "${BASE_MODEL_PATH}"
    printf 'backbone=%s\n' "${BACKBONE_PATH}"
    printf 'dataset=%s\n' "${DATASET_PATH}"
    printf 'sft_steps=%s\n' "${SFT_STEPS}"
    printf 'sft_global_batch=320\n'
    printf 'sft_learning_rate=1e-4\n'
    printf 'sft_delay_augmentation=false\n'
    printf 'sft_semantic_age=0\n'
    printf 'sft_state_history=4\n'
    printf 'sft_action_history=4\n'
    printf 'sft_tune_llm=false\n'
    printf 'sft_tune_visual=false\n'
    printf 'sft_tune_projector=false\n'
    printf 'sft_tune_diffusion_model=true\n'
    printf 'sft_tune_delay_adapter=true\n'
    printf 'sft_tune_vlln=false\n'
    printf 'ppo_reward=binary_terminal_success_once\n'
    printf 'ppo_steps=%s\n' "${PPO_STEPS}"
    printf 'ppo_train_envs=60\n'
    printf 'ppo_rollout_epochs_per_update=4\n'
    printf 'ppo_frames_per_rollout_epoch=256\n'
    printf 'ppo_global_batch=384\n'
    printf 'ppo_actor_lr=1e-7\n'
    printf 'ppo_value_lr=2e-5\n'
    printf 'eval_task=0\n'
    printf 'eval_trials=48\n'
    printf 'eval_prediction_horizon=16\n'
    printf 'eval_execution_horizon=2\n'
    printf 'eval_noise_seed=13026\n'
} >"${RUN_ROOT}/protocol.env"
git -C "${RLINF_REPO}" status --short >"${RUN_ROOT}/git_status_before.txt"
sha256sum "${BASE_MODEL_PATH}/model.safetensors.index.json" \
    >"${RUN_ROOT}/base_model_index.sha256"

run_sft() {
    mkdir -p "${SFT_OUTPUT_ROOT}"
    printf 'start_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        >"${RUN_ROOT}/sft_stage.env"
    local start_ns end_ns
    start_ns=$(date +%s%N)
    (
        cd "${GR00T_REPO}"
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export NUM_GPUS=4
        export MASTER_PORT=${C_SFT_MASTER_PORT:-29641}
        export SAVE_STEPS=${SFT_STEPS}
        export MAX_STEPS=${SFT_STEPS}
        export USE_WANDB=0
        export DATALOADER_NUM_WORKERS=${C_SFT_DATA_WORKERS:-4}
        export GLOBAL_BATCH_SIZE=320
        export LEARNING_RATE=1e-4
        export SHARD_SIZE=1024
        export NUM_SHARDS_PER_EPOCH=100000
        export EPISODE_SAMPLING_RATE=1.0
        bash examples/finetune.sh \
            --base-model-path "${BASE_MODEL_PATH}" \
            --dataset-path "${DATASET_PATH}" \
            --embodiment-tag libero_sim \
            --output-dir "${SFT_OUTPUT_ROOT}" \
            --experiment-name "${SFT_EXPERIMENT_NAME}" \
            --state-dropout-prob 0.2 \
            --use-percentiles true \
            --save-only-model \
            -- \
            --backbone-model-path "${BACKBONE_PATH}" \
            --no-tune-projector \
            --tune-delay-adapter \
            --no-tune-sampler-dt-adapter \
            --tune-diffusion-model \
            --no-tune-vlln \
            --no-delay-augmentation-enabled \
            --use-packet-age-embedding \
            --packet-age-embedding-type scalar \
            --packet-age-input-features packet_age \
            --packet-age-normalization-ms 400 \
            --state-history-length 4 \
            --action-history-length 4 \
            --action-history-noise-std 0.01
    ) >"${RUN_ROOT}/sft_launcher.log" 2>&1
    end_ns=$(date +%s%N)
    awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
        'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
        >"${RUN_ROOT}/sft_process_time.txt"
    printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        >>"${RUN_ROOT}/sft_stage.env"
}

apply_eval_protocol() {
    export ACTION_CHUNK_SIZE=16
    export EVAL_EXECUTION_HORIZON=2
    export DENOISING_STEPS=4
    export EVAL_TASK_ID_FILTER='[0]'
    export EVAL_NUM_ENVS=48
    export TARGET_EVAL_ENVS=48
    export EVAL_MAX_EPISODE_STEPS=480
    export EVAL_ROLLOUT_STEPS=480
    export EVAL_ROLLOUT_EPOCH=1
    export EVAL_AUTO_RESET=false
    export EVAL_IGNORE_TERMINATIONS=false
    export EVAL_SEED=0
    export EVAL_NOISE_SEED=13026
    export DETERMINISTIC_EVAL_NOISE=true
    export SEMANTIC_TEXT_PADDING_TOKENS=160
    export SEMANTIC_EVAL_FIXED_AGE_FRAMES=-1
    export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
    export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
    export SEMANTIC_FETCH_TARGET_AGE_FRAMES=-1
    export SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES=-1
}

run_coupled_eval() {
    local label=$1
    local policy_checkpoint=${2:-}
    local eval_dir="${RUN_ROOT}/${label}"
    mkdir -p "${eval_dir}"
    apply_eval_protocol
    export ONLY_EVAL=true
    export GR00T_MODEL_PATH="${SFT_CHECKPOINT}"
    export GR00T_BACKBONE_PATH="${BACKBONE_PATH}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export ACTOR_GPU_PLACEMENT=0-3
    export ROLLOUT_GPU_PLACEMENT=0-3
    export ENV_GPU_PLACEMENT=0-3
    export FDVLA_METHOD_ID="${label}"
    export FDVLA_RUN_TIMESTAMP="${RUN_ID}_${label}"
    export FDVLA_RUN_LOG_DIR="${eval_dir}"
    export RLINF_LOG_DIR="${eval_dir}"
    export FDVLA_METADATA_DIR="${eval_dir}/fdvla_metadata"
    export PPO_EXPERIMENT_NAME="${label}"
    export RAY_TMPDIR="${RAY_TMP_ROOT}"
    unset PPO_RESUME_DIR
    if [[ -n "${policy_checkpoint}" ]]; then
        export PPO_CKPT_PATH="${policy_checkpoint}"
    else
        unset PPO_CKPT_PATH
    fi
    bash "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh" \
        >"${eval_dir}/launcher.log" 2>&1
    test -s "${eval_dir}/eval_trials.jsonl"
    test "$(wc -l <"${eval_dir}/eval_trials.jsonl")" -eq 48
}

run_coupled_ppo() {
    mkdir -p "${PPO_RUN_DIR}"
    apply_eval_protocol
    export ONLY_EVAL=false
    export GR00T_MODEL_PATH="${SFT_CHECKPOINT}"
    export GR00T_BACKBONE_PATH="${BACKBONE_PATH}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export ACTOR_GPU_PLACEMENT=0-3
    export ROLLOUT_GPU_PLACEMENT=0-3
    export ENV_GPU_PLACEMENT=0-3
    export TRAIN_TASK_ID_FILTER='[0]'
    export BALANCE_TRAIN_TASK_ASSIGNMENT=true
    export PRESERVE_TRAIN_TASK_ASSIGNMENT=true
    export TRAIN_NUM_ENVS=60
    export TARGET_TRAIN_ENVS=60
    export TRAIN_MAX_EPISODE_STEPS=480
    export TRAIN_ROLLOUT_STEPS=256
    export TRAIN_ROLLOUT_EPOCH=4
    export TRAIN_SEED=0
    export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false
    export PPO_MAX_STEPS="${PPO_STEPS}"
    export PPO_GLOBAL_BATCH_SIZE=384
    export PPO_MICRO_BATCH_SIZE=2
    export PPO_UPDATE_EPOCHS=1
    export PPO_ACTOR_LR=1e-7
    export PPO_VALUE_LR=2e-5
    export PPO_EVAL_BEFORE_TRAINING=true
    export PPO_VAL_INTERVAL=10
    export PPO_SAVE_INTERVAL=10
    export PPO_SUCCESS_EARLY_STOP_ENABLED=false
    export FDVLA_METHOD_ID=C-PPO-after-C-SFT
    export FDVLA_RUN_TIMESTAMP="${RUN_ID}_C-PPO"
    export FDVLA_RUN_LOG_DIR="${PPO_RUN_DIR}"
    export RLINF_LOG_DIR="${PPO_RUN_DIR}"
    export FDVLA_METADATA_DIR="${PPO_RUN_DIR}/fdvla_metadata"
    export PPO_EXPERIMENT_NAME="${PPO_EXPERIMENT_NAME_FINAL}"
    export RAY_TMPDIR="${RAY_TMP_ROOT}"
    unset PPO_CKPT_PATH PPO_RESUME_DIR
    bash "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh" \
        >"${PPO_RUN_DIR}/launcher.log" 2>&1
}

if [[ ! -d "${SFT_CHECKPOINT}" ]]; then
    run_sft
fi
if [[ ! -f "${SFT_CHECKPOINT}/model.safetensors.index.json" ]]; then
    echo "C-SFT did not produce the expected checkpoint: ${SFT_CHECKPOINT}" >&2
    exit 1
fi
sha256sum "${SFT_CHECKPOINT}/model.safetensors.index.json" \
    >"${RUN_ROOT}/c_sft_model_index.sha256"

if [[ ! -s "${RUN_ROOT}/c_sft_k2/eval_trials.jsonl" ]]; then
    run_coupled_eval c_sft_k2
fi
if [[ ! -f "${PPO_CHECKPOINT}" ]]; then
    run_coupled_ppo
fi
if [[ ! -f "${PPO_CHECKPOINT}" ]]; then
    echo "C-PPO did not produce the expected checkpoint: ${PPO_CHECKPOINT}" >&2
    exit 1
fi
sha256sum "${PPO_CHECKPOINT}" >"${RUN_ROOT}/c_ppo_full_weights.sha256"

if [[ ! -s "${RUN_ROOT}/c_ppo_step${PPO_STEPS}_k2/eval_trials.jsonl" ]]; then
    run_coupled_eval "c_ppo_step${PPO_STEPS}_k2" "${PPO_CHECKPOINT}"
fi

printf '%s\n' "${RUN_ROOT}"
