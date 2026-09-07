#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLINF_REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
GR00T_REPO=$(cd "${RLINF_REPO}/.." && pwd)

RUN_ID=${FDVLA_MATCHED_SFT_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_MATCHED_SFT_RUN_ROOT:-${RLINF_REPO}/logs/fdvla_matched_start/${RUN_ID}/sft_grid}
BASE_MODEL_PATH=${FDVLA_MATCHED_BASE_MODEL_PATH:-${GR00T_REPO}/checkpoints/GR00T-N1.7-3B}
BACKBONE_PATH=${GR00T_BACKBONE_PATH:-${GR00T_REPO}/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}
DATASET_PATH=${FDVLA_MATCHED_DATASET_PATH:-${GR00T_REPO}/examples/LIBERO/nvidia/LIBERO_LeRobot_v3/libero_10}

MAX_STEPS=${FDVLA_MATCHED_SFT_MAX_STEPS:-1000}
SAVE_INTERVAL=${FDVLA_MATCHED_SFT_SAVE_INTERVAL:-200}
GLOBAL_BATCH_SIZE=${FDVLA_MATCHED_SFT_GLOBAL_BATCH_SIZE:-320}
LEARNING_RATE=${FDVLA_MATCHED_SFT_LR:-1e-4}
SFT_ARMS=${FDVLA_MATCHED_SFT_ARMS:-"C D"}

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
if [[ "${RUN_ROOT}" != /* ]]; then
    echo "RUN_ROOT must be absolute: ${RUN_ROOT}" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}"
{
    printf 'protocol_label=fdvla_matched_start_sft_grid\n'
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'git_sha=%s\n' "$(git -C "${RLINF_REPO}" rev-parse HEAD)"
    printf 'base_model=%s\n' "${BASE_MODEL_PATH}"
    printf 'backbone=%s\n' "${BACKBONE_PATH}"
    printf 'dataset=%s\n' "${DATASET_PATH}"
    printf 'candidate_steps=%s\n' "$(seq -s, "${SAVE_INTERVAL}" "${SAVE_INTERVAL}" "${MAX_STEPS}")"
    printf 'global_batch_size=%s\n' "${GLOBAL_BATCH_SIZE}"
    printf 'learning_rate=%s\n' "${LEARNING_RATE}"
    printf 'sft_arms=%s\n' "${SFT_ARMS}"
    printf 'state_history=4\n'
    printf 'action_history=4\n'
    printf 'c_semantic_training=current_frame_age0\n'
    printf 'd_semantic_training=uniform_delay_0to400ms_probability_age0_0.3\n'
    printf 'selection_target_successes=13/48\n'
} >"${RUN_ROOT}/protocol.env"
git -C "${RLINF_REPO}" status --short >"${RUN_ROOT}/git_status_before.txt"
sha256sum "${BASE_MODEL_PATH}/model.safetensors.index.json" >"${RUN_ROOT}/base_model_index.sha256"

run_sft() {
    local arm=$1
    local experiment_name output_root launcher_log master_port
    local -a temporal_args

    case "${arm}" in
        C)
            experiment_name="libero10_coupled_current_s4_a4_grid${MAX_STEPS}"
            temporal_args=(--no-delay-augmentation-enabled)
            master_port=${FDVLA_MATCHED_C_MASTER_PORT:-29651}
            ;;
        D)
            experiment_name="libero10_decoupled_delay0to8_s4_a4_grid${MAX_STEPS}"
            temporal_args=(
                --delay-augmentation-enabled
                --probability-zero-delay 0.3
                --min-delay-ms 0
                --max-delay-ms 400
                --delay-distribution uniform
                --delay-control-dt-ms 50
            )
            master_port=${FDVLA_MATCHED_D_MASTER_PORT:-29652}
            ;;
        *)
            echo "Unknown arm: ${arm}" >&2
            exit 2
            ;;
    esac

    output_root=${RUN_ROOT}/${arm}
    launcher_log=${RUN_ROOT}/${arm}_sft_launcher.log
    mkdir -p "${output_root}"
    if [[ -f "${output_root}/${experiment_name}/checkpoint-${MAX_STEPS}/model.safetensors.index.json" ]]; then
        printf '%s grid already complete; skipping.\n' "${arm}"
        return
    fi

    local start_ns end_ns
    start_ns=$(date +%s%N)
    (
        cd "${GR00T_REPO}"
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export NUM_GPUS=4
        export MASTER_PORT="${master_port}"
        export SAVE_STEPS="${SAVE_INTERVAL}"
        export MAX_STEPS="${MAX_STEPS}"
        export USE_WANDB=0
        export DATALOADER_NUM_WORKERS=${FDVLA_MATCHED_SFT_DATA_WORKERS:-4}
        export GLOBAL_BATCH_SIZE
        export LEARNING_RATE
        export SHARD_SIZE=1024
        export NUM_SHARDS_PER_EPOCH=100000
        export EPISODE_SAMPLING_RATE=1.0
        bash examples/finetune.sh \
            --base-model-path "${BASE_MODEL_PATH}" \
            --dataset-path "${DATASET_PATH}" \
            --embodiment-tag libero_sim \
            --output-dir "${output_root}" \
            --experiment-name "${experiment_name}" \
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
            "${temporal_args[@]}" \
            --use-packet-age-embedding \
            --packet-age-embedding-type scalar \
            --packet-age-input-features packet_age \
            --packet-age-normalization-ms 400 \
            --state-history-length 4 \
            --action-history-length 4 \
            --action-history-noise-std 0.01
    ) >"${launcher_log}" 2>&1
    end_ns=$(date +%s%N)
    awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
        'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
        >"${RUN_ROOT}/${arm}_sft_process_time.txt"

    for step in $(seq "${SAVE_INTERVAL}" "${SAVE_INTERVAL}" "${MAX_STEPS}"); do
        test -f "${output_root}/${experiment_name}/checkpoint-${step}/model.safetensors.index.json"
    done
}

read -r -a sft_arms <<<"${SFT_ARMS}"
for arm in "${sft_arms[@]}"; do
    if [[ "${arm}" != C && "${arm}" != D ]]; then
        echo "Unknown SFT arm: ${arm}" >&2
        exit 2
    fi
    run_sft "${arm}"
done

printf '%s\n' "${RUN_ROOT}"
