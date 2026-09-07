#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLINF_REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
GR00T_REPO=$(cd "${RLINF_REPO}/.." && pwd)

RUN_ID=${FDVLA_C_TARGET27_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_C_TARGET27_RUN_ROOT:-${RLINF_REPO}/logs/fdvla_c_sft_target27/${RUN_ID}}
GRID_ROOT=${RUN_ROOT}/sft_grid
OUTPUT_ROOT=${GRID_ROOT}/C
SELECTION_ROOT=${RUN_ROOT}/selection
BASE_MODEL_PATH=${FDVLA_C_TARGET27_BASE_MODEL_PATH:-${GR00T_REPO}/checkpoints/GR00T-N1.7-3B}
BACKBONE_PATH=${GR00T_BACKBONE_PATH:-${GR00T_REPO}/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}
DATASET_PATH=${FDVLA_C_TARGET27_DATASET_PATH:-${GR00T_REPO}/examples/LIBERO/nvidia/LIBERO_LeRobot_v3/libero_10}
MAX_STEPS=${FDVLA_C_TARGET27_MAX_STEPS:-1200}
SAVE_INTERVAL=${FDVLA_C_TARGET27_SAVE_INTERVAL:-100}
MIN_SELECTION_STEP=${FDVLA_C_TARGET27_MIN_SELECTION_STEP:-800}
LEARNING_RATE=${FDVLA_C_TARGET27_LR:-1e-4}
SAVE_TOTAL_LIMIT=${FDVLA_C_TARGET27_SAVE_TOTAL_LIMIT:-$((MAX_STEPS / SAVE_INTERVAL))}
EXPERIMENT_NAME=libero10_coupled_current_s4_a4_grid${MAX_STEPS}

mkdir -p "${RUN_ROOT}" "${OUTPUT_ROOT}" "${SELECTION_ROOT}"
{
    printf 'protocol_label=fdvla_c_sft_target27\n'
    printf 'git_sha=%s\n' "$(git -C "${RLINF_REPO}" rev-parse HEAD)"
    printf 'base_model=%s\n' "${BASE_MODEL_PATH}"
    printf 'dataset=%s\n' "${DATASET_PATH}"
    printf 'max_steps=%s\n' "${MAX_STEPS}"
    printf 'save_interval=%s\n' "${SAVE_INTERVAL}"
    printf 'learning_rate=%s\n' "${LEARNING_RATE}"
    printf 'save_total_limit=%s\n' "${SAVE_TOTAL_LIMIT}"
    printf 'selection_steps=%s..%s\n' "${MIN_SELECTION_STEP}" "${MAX_STEPS}"
    printf 'selection_protocol=task0_trials48_k2_noise14026\n'
    printf 'target_successes=13/48\n'
} >"${RUN_ROOT}/protocol.env"

checkpoint=${OUTPUT_ROOT}/${EXPERIMENT_NAME}/checkpoint-${MAX_STEPS}
if [[ ! -f "${checkpoint}/model.safetensors.index.json" ]]; then
    start_seconds=${SECONDS}
    (
        cd "${GR00T_REPO}"
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export NUM_GPUS=4
        export MASTER_PORT=${FDVLA_C_TARGET27_MASTER_PORT:-29661}
        export SAVE_STEPS="${SAVE_INTERVAL}"
        export SAVE_TOTAL_LIMIT
        export MAX_STEPS
        export USE_WANDB=0
        export DATALOADER_NUM_WORKERS=4
        export GLOBAL_BATCH_SIZE=320
        export LEARNING_RATE
        export SHARD_SIZE=1024
        export NUM_SHARDS_PER_EPOCH=100000
        export EPISODE_SAMPLING_RATE=1.0
        bash examples/finetune.sh \
            --base-model-path "${BASE_MODEL_PATH}" \
            --dataset-path "${DATASET_PATH}" \
            --embodiment-tag libero_sim \
            --output-dir "${OUTPUT_ROOT}" \
            --experiment-name "${EXPERIMENT_NAME}" \
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
    printf 'process_elapsed_seconds=%s\n' "$((SECONDS - start_seconds))" \
        >"${RUN_ROOT}/sft_process_time.txt"
fi

cd "${RLINF_REPO}"
export FDVLA_MATCHED_SFT_RUN_ROOT="${GRID_ROOT}"
export FDVLA_MATCHED_SELECTION_ROOT="${SELECTION_ROOT}"
export FDVLA_MATCHED_SFT_MAX_STEPS="${MAX_STEPS}"
export FDVLA_MATCHED_SFT_SAVE_INTERVAL="${SAVE_INTERVAL}"
export FDVLA_MATCHED_SELECTION_MIN_STEP="${MIN_SELECTION_STEP}"
export FDVLA_MATCHED_SELECTION_ARMS=C
export FDVLA_MATCHED_SELECTION_NOISE_SEED=14026
export FDVLA_MATCHED_TARGET_SUCCESSES=13
export FDVLA_MATCHED_RAY_SHORT_PREFIX=${FDVLA_C_TARGET27_RAY_SHORT_PREFIX:-fct27}
export GR00T_BACKBONE_PATH="${BACKBONE_PATH}"
bash "${SCRIPT_DIR}/run_fdvla_matched_start_selection.sh" \
    >"${SELECTION_ROOT}/watcher.log" 2>&1

cat "${SELECTION_ROOT}/selected.env"
