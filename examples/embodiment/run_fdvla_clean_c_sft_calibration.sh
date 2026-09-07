#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLINF_REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
GR00T_REPO=$(cd "${RLINF_REPO}/.." && pwd)

RUN_ID=${FDVLA_CLEAN_C_SFT_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_CLEAN_C_SFT_RUN_ROOT:-${RLINF_REPO}/logs/fdvla_clean_c_sft_calibration/${RUN_ID}}
GRID_ROOT=${RUN_ROOT}/sft_grid
SELECTION_ROOT=${RUN_ROOT}/selection
BASE_MODEL_PATH=${FDVLA_CLEAN_C_BASE_MODEL_PATH:-${GR00T_REPO}/checkpoints/GR00T-N1.7-LIBERO/libero_10}
BACKBONE_PATH=${GR00T_BACKBONE_PATH:-${GR00T_REPO}/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}
DATASET_PATH=${FDVLA_CLEAN_C_DATASET_PATH:-${RLINF_REPO}/logs/fdvla_task0_sft_data/libero_10_task0_v1}

MAX_STEPS=${FDVLA_CLEAN_C_SFT_MAX_STEPS:-5}
SAVE_INTERVAL=${FDVLA_CLEAN_C_SFT_SAVE_INTERVAL:-1}
GLOBAL_BATCH_SIZE=${FDVLA_CLEAN_C_SFT_GLOBAL_BATCH_SIZE:-320}
LEARNING_RATE=${FDVLA_CLEAN_C_SFT_LR:-1e-4}
EXPERIMENT_NAME=${FDVLA_CLEAN_C_EXPERIMENT:-libero10_coupled_clean_s1_a0_grid${MAX_STEPS}}

if (( MAX_STEPS != SAVE_INTERVAL * 5 )); then
    echo "MAX_STEPS must equal 5 * SAVE_INTERVAL because finetune.sh retains five checkpoints." >&2
    exit 2
fi
for required_path in "${BASE_MODEL_PATH}" "${BACKBONE_PATH}" "${DATASET_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required input does not exist: ${required_path}" >&2
        exit 2
    fi
done

mkdir -p "${GRID_ROOT}/C" "${SELECTION_ROOT}"
{
    printf 'protocol_label=fdvla_clean_c_sft_calibration\n'
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'git_sha=%s\n' "$(git -C "${RLINF_REPO}" rev-parse HEAD)"
    printf 'base_model=%s\n' "${BASE_MODEL_PATH}"
    printf 'dataset=%s\n' "${DATASET_PATH}"
    printf 'dataset_scope=task0_only_finite_demonstrations\n'
    printf 'candidate_steps=%s\n' "$(seq -s, "${SAVE_INTERVAL}" "${SAVE_INTERVAL}" "${MAX_STEPS}")"
    printf 'global_batch_size=%s\n' "${GLOBAL_BATCH_SIZE}"
    printf 'learning_rate=%s\n' "${LEARNING_RATE}"
    printf 'trainable_scope=dit_only\n'
    printf 'packet_age_embedding=false\n'
    printf 'state_history_length=1\n'
    printf 'action_history_length=0\n'
    printf 'delay_adapters=false\n'
    printf 'selection_trials=48\n'
    printf 'selection_noise_seed=2026\n'
    printf 'selection_target_successes=14/48\n'
    printf 'heldout_confirmation_noise_seed=26026\n'
} >"${RUN_ROOT}/protocol.env"
git -C "${RLINF_REPO}" status --short >"${RUN_ROOT}/git_status_before.txt"
sha256sum "${BASE_MODEL_PATH}/model.safetensors.index.json" >"${RUN_ROOT}/base_model_index.sha256"

checkpoint_dir=${GRID_ROOT}/C/${EXPERIMENT_NAME}/checkpoint-${MAX_STEPS}
if [[ ! -f "${checkpoint_dir}/model.safetensors.index.json" ]]; then
    start_ns=$(date +%s%N)
    (
        cd "${GR00T_REPO}"
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export NUM_GPUS=4
        export MASTER_PORT=${FDVLA_CLEAN_C_MASTER_PORT:-29661}
        export SAVE_STEPS="${SAVE_INTERVAL}"
        export MAX_STEPS
        export USE_WANDB=0
        export DATALOADER_NUM_WORKERS=${FDVLA_CLEAN_C_SFT_DATA_WORKERS:-4}
        export GLOBAL_BATCH_SIZE
        export LEARNING_RATE
        export SHARD_SIZE=1024
        export NUM_SHARDS_PER_EPOCH=100000
        export EPISODE_SAMPLING_RATE=1.0
        bash examples/finetune.sh \
            --base-model-path "${BASE_MODEL_PATH}" \
            --dataset-path "${DATASET_PATH}" \
            --embodiment-tag libero_sim \
            --output-dir "${GRID_ROOT}/C" \
            --experiment-name "${EXPERIMENT_NAME}" \
            --state-dropout-prob 0.2 \
            --use-percentiles true \
            --save-only-model \
            -- \
            --backbone-model-path "${BACKBONE_PATH}" \
            --no-tune-projector \
            --no-tune-delay-adapter \
            --no-tune-sampler-dt-adapter \
            --no-tune-stale-residual-adapter \
            --no-tune-stale-semantic-token-adapter \
            --tune-diffusion-model \
            --no-tune-vlln \
            --no-delay-augmentation-enabled \
            --no-use-packet-age-embedding \
            --state-history-length 1 \
            --action-history-length 0 \
            --action-history-noise-std 0.0
    ) >"${RUN_ROOT}/C_sft_launcher.log" 2>&1
    end_ns=$(date +%s%N)
    awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
        'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
        >"${RUN_ROOT}/C_sft_process_time.txt"
fi

for step in $(seq "${SAVE_INTERVAL}" "${SAVE_INTERVAL}" "${MAX_STEPS}"); do
    cfg=${GRID_ROOT}/C/${EXPERIMENT_NAME}/checkpoint-${step}/experiment_cfg/final_model_config.json
    test -f "${cfg}"
    python - "${cfg}" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1]))
assert model.get("use_packet_age_embedding", False) is False
assert model.get("action_history_length", 0) == 0
assert model.get("state_history_length", 1) == 1
assert model.get("tune_delay_adapter", False) is False
assert model.get("tune_sampler_dt_adapter", False) is False
assert model.get("tune_stale_residual_adapter", False) is False
assert model.get("tune_stale_semantic_token_adapter", False) is False
assert model.get("tune_diffusion_model", False) is True
PY
done

env \
    FDVLA_MATCHED_SFT_RUN_ROOT="${GRID_ROOT}" \
    FDVLA_MATCHED_SELECTION_ROOT="${SELECTION_ROOT}" \
    FDVLA_MATCHED_SFT_MAX_STEPS="${MAX_STEPS}" \
    FDVLA_MATCHED_SFT_SAVE_INTERVAL="${SAVE_INTERVAL}" \
    FDVLA_MATCHED_SELECTION_MIN_STEP="${SAVE_INTERVAL}" \
    FDVLA_MATCHED_SELECTION_ARMS=C \
    FDVLA_MATCHED_C_EXPERIMENT="${EXPERIMENT_NAME}" \
    FDVLA_MATCHED_SELECTION_NOISE_SEED=2026 \
    FDVLA_MATCHED_TARGET_SUCCESSES=14 \
    FDVLA_MATCHED_SELECTION_EXECUTION_HORIZON=8 \
    FDVLA_MATCHED_SELECTION_TEXT_PADDING_TOKENS=570 \
    FDVLA_MATCHED_SELECTION_REQUIRE_PACKET_AGE_INPUT=false \
    FDVLA_MATCHED_SELECTION_INITIALIZE_PACKET_AGE_ADAPTER=false \
    FDVLA_MATCHED_SELECTION_ACTION_HISTORY_LENGTH=0 \
    FDVLA_MATCHED_SELECTION_ZERO_INIT_DELAY_ADAPTERS=false \
    FDVLA_MATCHED_RAY_SHORT_ROOT=/dev/shm \
    FDVLA_MATCHED_RAY_SHORT_PREFIX=fcc \
    GR00T_BACKBONE_PATH="${BACKBONE_PATH}" \
    bash "${SCRIPT_DIR}/run_fdvla_matched_start_selection.sh"

printf '%s\n' "${RUN_ROOT}"
