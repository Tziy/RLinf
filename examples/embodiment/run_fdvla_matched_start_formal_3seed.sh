#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

BATCHING_ENV=${FDVLA_FORMAL_BATCHING_ENV:?Set FDVLA_FORMAL_BATCHING_ENV to the frozen timing-only batching manifest}
if [[ ! -f "${BATCHING_ENV}" ]]; then
    echo "Missing frozen batching manifest: ${BATCHING_ENV}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${BATCHING_ENV}"
: "${selected_batching:?Missing selected_batching in batching manifest}"
: "${semantic_batch_max_requests:?Missing semantic_batch_max_requests}"
: "${semantic_batch_target_requests:?Missing semantic_batch_target_requests}"
: "${semantic_batch_wait_ms:?Missing semantic_batch_wait_ms}"

SELECTED_ENV=${FDVLA_FORMAL_SELECTED_ENV:?Set FDVLA_FORMAL_SELECTED_ENV to the frozen matched-start manifest}
if [[ ! -f "${SELECTED_ENV}" ]]; then
    echo "Missing frozen matched-start manifest: ${SELECTED_ENV}" >&2
    exit 2
fi

FORMAL_RUN_ID=${FDVLA_FORMAL_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')_task0_k2_bf16_tail4_formal}
FORMAL_ROOT=${FDVLA_FORMAL_ROOT:-${REPO_DIR}/logs/fdvla_matched_start/${FORMAL_RUN_ID}}
FORMAL_SEEDS=${FDVLA_FORMAL_SEEDS:-"0 1 2"}
FORMAL_EVAL_SEED=${FDVLA_FORMAL_EVAL_SEED:-0}
FORMAL_EVAL_NOISE_SEED=${FDVLA_FORMAL_EVAL_NOISE_SEED:-27026}
SELECTION_NOISE_SEED=${FDVLA_FORMAL_SELECTION_NOISE_SEED:-14026}
PPO_STEPS=${FDVLA_FORMAL_PPO_STEPS:-50}
EXECUTION_HORIZON=${FDVLA_FORMAL_EXECUTION_HORIZON:-2}
PPO_GAMMA_VALUE=${FDVLA_FORMAL_PPO_GAMMA:-0.99}
PPO_GAE_LAMBDA_VALUE=${FDVLA_FORMAL_PPO_GAE_LAMBDA:-0.95}
PPO_ACTOR_LR_VALUE=${FDVLA_FORMAL_PPO_ACTOR_LR:-5.0e-7}
PPO_VALUE_LR_VALUE=${FDVLA_FORMAL_PPO_VALUE_LR:-2.0e-5}
PPO_CRITIC_WARMUP_STEPS_VALUE=${FDVLA_FORMAL_PPO_CRITIC_WARMUP_STEPS:-0}
PPO_VAL_INTERVAL_VALUE=${FDVLA_FORMAL_PPO_VAL_INTERVAL:-5}
PPO_SAVE_INTERVAL_VALUE=${FDVLA_FORMAL_PPO_SAVE_INTERVAL:-5}
PPO_GLOBAL_BATCH_SIZE_VALUE=${FDVLA_FORMAL_PPO_GLOBAL_BATCH_SIZE:-960}
PPO_MICRO_BATCH_SIZE_VALUE=${FDVLA_FORMAL_PPO_MICRO_BATCH_SIZE:-2}
PPO_UPDATE_EPOCHS_VALUE=${FDVLA_FORMAL_PPO_UPDATE_EPOCHS:-1}
PPO_KL_BETA_VALUE=${FDVLA_FORMAL_PPO_KL_BETA:-0.0}
ACTOR_PRECISION_VALUE=${FDVLA_FORMAL_ACTOR_MODEL_PRECISION:-bf16}
ROLLOUT_PRECISION_VALUE=${FDVLA_FORMAL_ROLLOUT_MODEL_PRECISION:-bf16}
DIT_TRAIN_LAST_N_BLOCKS_VALUE=${FDVLA_FORMAL_DIT_TRAIN_LAST_N_BLOCKS:-4}
SEMANTIC_PUBLISH_INTERVAL_VALUE=${FDVLA_FORMAL_SEMANTIC_PUBLISH_INTERVAL_FRAMES:-8}
MAX_START_DEVIATION=${FDVLA_FORMAL_MAX_START_DEVIATION_SUCCESSES:-4}
MAX_START_GAP=${FDVLA_FORMAL_MAX_START_GAP_SUCCESSES:-4}

read -r -a seeds <<<"${FORMAL_SEEDS}"
if ((${#seeds[@]} != 3)); then
    echo "Formal comparison requires exactly three training seeds; got: ${FORMAL_SEEDS}" >&2
    exit 2
fi
declare -A seen=()
for seed in "${seeds[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Training seed must be a non-negative integer: ${seed}" >&2
        exit 2
    fi
    if [[ -n "${seen[${seed}]:-}" ]]; then
        echo "Training seeds must be unique: ${FORMAL_SEEDS}" >&2
        exit 2
    fi
    seen[${seed}]=1
done
if ((PPO_STEPS != 50 || EXECUTION_HORIZON != 2 || PPO_CRITIC_WARMUP_STEPS_VALUE != 0 || PPO_VAL_INTERVAL_VALUE != 5 || PPO_SAVE_INTERVAL_VALUE != 5 || PPO_GLOBAL_BATCH_SIZE_VALUE != 960 || PPO_MICRO_BATCH_SIZE_VALUE != 2 || PPO_UPDATE_EPOCHS_VALUE != 1 || DIT_TRAIN_LAST_N_BLOCKS_VALUE != 4 || SEMANTIC_PUBLISH_INTERVAL_VALUE != 8 || MAX_START_DEVIATION != 4 || MAX_START_GAP != 4)) || [[ "${PPO_GAMMA_VALUE}" != "0.99" || "${PPO_GAE_LAMBDA_VALUE}" != "0.95" || "${PPO_ACTOR_LR_VALUE}" != "5.0e-7" || "${PPO_VALUE_LR_VALUE}" != "2.0e-5" || "${PPO_KL_BETA_VALUE}" != "0.0" || "${ACTOR_PRECISION_VALUE}" != "bf16" || "${ROLLOUT_PRECISION_VALUE}" != "bf16" ]]; then
    echo "Refusing to relabel a non-registered run as formal: steps=${PPO_STEPS}, K=${EXECUTION_HORIZON}, actor_lr=${PPO_ACTOR_LR_VALUE}, value_lr=${PPO_VALUE_LR_VALUE}, gamma=${PPO_GAMMA_VALUE}, gae_lambda=${PPO_GAE_LAMBDA_VALUE}, critic_warmup=${PPO_CRITIC_WARMUP_STEPS_VALUE}, val_interval=${PPO_VAL_INTERVAL_VALUE}, save_interval=${PPO_SAVE_INTERVAL_VALUE}, global_batch=${PPO_GLOBAL_BATCH_SIZE_VALUE}, micro_batch=${PPO_MICRO_BATCH_SIZE_VALUE}, update_epochs=${PPO_UPDATE_EPOCHS_VALUE}, kl_beta=${PPO_KL_BETA_VALUE}, actor_precision=${ACTOR_PRECISION_VALUE}, rollout_precision=${ROLLOUT_PRECISION_VALUE}, tail_blocks=${DIT_TRAIN_LAST_N_BLOCKS_VALUE}, semantic_interval=${SEMANTIC_PUBLISH_INTERVAL_VALUE}, max_start_deviation=${MAX_START_DEVIATION}, max_start_gap=${MAX_START_GAP}" >&2
    exit 2
fi

if [[ -e "${FORMAL_ROOT}/formal_protocol.env" ]]; then
    echo "Formal root already contains a locked protocol: ${FORMAL_ROOT}" >&2
    echo "Use a new FDVLA_FORMAL_RUN_ID; this launcher never overwrites or silently resumes formal runs." >&2
    exit 2
fi
mkdir -p "${FORMAL_ROOT}"
cp "${SELECTED_ENV}" "${FORMAL_ROOT}/selected_start_manifest.env"
cp "${BATCHING_ENV}" "${FORMAL_ROOT}/selected_batching_manifest.env"
git -C "${REPO_DIR}" status --short >"${FORMAL_ROOT}/git_status_before.txt"
git -C "${REPO_DIR}" rev-parse HEAD >"${FORMAL_ROOT}/git_sha.txt"

{
    printf 'formal_run_id=%s\n' "${FORMAL_RUN_ID}"
    printf 'selected_start_manifest=%s\n' "$(realpath "${SELECTED_ENV}")"
    printf 'selected_batching_manifest=%s\n' "$(realpath "${BATCHING_ENV}")"
    printf 'selected_batching=%s\n' "${selected_batching}"
    printf 'semantic_batch_max_requests=%s\n' "${semantic_batch_max_requests}"
    printf 'semantic_batch_target_requests=%s\n' "${semantic_batch_target_requests}"
    printf 'semantic_batch_wait_ms=%s\n' "${semantic_batch_wait_ms}"
    printf 'training_seeds=%s\n' "${FORMAL_SEEDS}"
    printf 'eval_seed=%s\n' "${FORMAL_EVAL_SEED}"
    printf 'selection_noise_seed=%s\n' "${SELECTION_NOISE_SEED}"
    printf 'training_eval_noise_seed=%s\n' "${FORMAL_EVAL_NOISE_SEED}"
    printf 'ppo_steps=%s\n' "${PPO_STEPS}"
    printf 'execution_horizon=%s\n' "${EXECUTION_HORIZON}"
    printf 'gamma=%s\n' "${PPO_GAMMA_VALUE}"
    printf 'gae_lambda=%s\n' "${PPO_GAE_LAMBDA_VALUE}"
    printf 'actor_lr=%s\n' "${PPO_ACTOR_LR_VALUE}"
    printf 'value_lr=%s\n' "${PPO_VALUE_LR_VALUE}"
    printf 'critic_warmup_steps=%s\n' "${PPO_CRITIC_WARMUP_STEPS_VALUE}"
    printf 'val_interval=%s\n' "${PPO_VAL_INTERVAL_VALUE}"
    printf 'save_interval=%s\n' "${PPO_SAVE_INTERVAL_VALUE}"
    printf 'global_batch_size=%s\n' "${PPO_GLOBAL_BATCH_SIZE_VALUE}"
    printf 'micro_batch_size=%s\n' "${PPO_MICRO_BATCH_SIZE_VALUE}"
    printf 'update_epochs=%s\n' "${PPO_UPDATE_EPOCHS_VALUE}"
    printf 'kl_beta=%s\n' "${PPO_KL_BETA_VALUE}"
    printf 'actor_precision=%s\n' "${ACTOR_PRECISION_VALUE}"
    printf 'rollout_precision=%s\n' "${ROLLOUT_PRECISION_VALUE}"
    printf 'dit_train_last_n_blocks=%s\n' "${DIT_TRAIN_LAST_N_BLOCKS_VALUE}"
    printf 'semantic_publish_interval_frames=%s\n' "${SEMANTIC_PUBLISH_INTERVAL_VALUE}"
    printf 'max_start_deviation_successes=%s\n' "${MAX_START_DEVIATION}"
    printf 'max_start_gap_successes=%s\n' "${MAX_START_GAP}"
    printf 'reward_contract=binary_terminal_success_once\n'
    printf 'reward_model_enabled=false\n'
    printf 'formal_final_eval_is_separate=true\n'
    printf 'start_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
} >"${FORMAL_ROOT}/formal_protocol.env"
printf 'seed\tstatus\tstart_utc\tend_utc\trun_root\n' >"${FORMAL_ROOT}/formal_runs.tsv"

for seed in "${seeds[@]}"; do
    seed_root=${FORMAL_ROOT}/seed${seed}/ppo_compare
    seed_start=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    printf '%s\trunning\t%s\t\t%s\n' "${seed}" "${seed_start}" "${seed_root}" \
        >>"${FORMAL_ROOT}/formal_runs.tsv"

    FDVLA_MATCHED_SELECTED_ENV="${SELECTED_ENV}" \
    FDVLA_MATCHED_BATCHING_ENV="${BATCHING_ENV}" \
    FDVLA_MATCHED_PPO_RUN_ID="${FORMAL_RUN_ID}_seed${seed}" \
    FDVLA_MATCHED_PPO_RUN_ROOT="${seed_root}" \
    FDVLA_MATCHED_PPO_RAY_SHORT_PREFIX="ff${seed}" \
    FDVLA_MATCHED_RUN_ARMS="C,D" \
    FDVLA_MATCHED_MAX_START_DEVIATION_SUCCESSES="${MAX_START_DEVIATION}" \
    FDVLA_MATCHED_MAX_START_GAP_SUCCESSES="${MAX_START_GAP}" \
    FDVLA_MATCHED_SELECTION_NOISE_SEED="${SELECTION_NOISE_SEED}" \
    FDVLA_MATCHED_PPO_EVAL_NOISE_SEED="${FORMAL_EVAL_NOISE_SEED}" \
    FDVLA_MATCHED_PPO_TRAIN_SEED="${seed}" \
    FDVLA_MATCHED_PPO_EVAL_SEED="${FORMAL_EVAL_SEED}" \
    FDVLA_MATCHED_PPO_EXECUTION_HORIZON="${EXECUTION_HORIZON}" \
    FDVLA_MATCHED_PPO_GAMMA="${PPO_GAMMA_VALUE}" \
    FDVLA_MATCHED_PPO_GAE_LAMBDA="${PPO_GAE_LAMBDA_VALUE}" \
    FDVLA_MATCHED_PPO_ACTOR_LR="${PPO_ACTOR_LR_VALUE}" \
    FDVLA_MATCHED_PPO_VALUE_LR="${PPO_VALUE_LR_VALUE}" \
    FDVLA_MATCHED_PPO_CRITIC_WARMUP_STEPS="${PPO_CRITIC_WARMUP_STEPS_VALUE}" \
    FDVLA_MATCHED_PPO_VAL_INTERVAL="${PPO_VAL_INTERVAL_VALUE}" \
    FDVLA_MATCHED_PPO_SAVE_INTERVAL="${PPO_SAVE_INTERVAL_VALUE}" \
    FDVLA_MATCHED_PPO_GLOBAL_BATCH_SIZE="${PPO_GLOBAL_BATCH_SIZE_VALUE}" \
    FDVLA_MATCHED_PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE_VALUE}" \
    FDVLA_MATCHED_PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS_VALUE}" \
    FDVLA_MATCHED_PPO_KL_BETA="${PPO_KL_BETA_VALUE}" \
    FDVLA_MATCHED_ACTOR_MODEL_PRECISION="${ACTOR_PRECISION_VALUE}" \
    FDVLA_MATCHED_ROLLOUT_MODEL_PRECISION="${ROLLOUT_PRECISION_VALUE}" \
    FDVLA_MATCHED_DIT_TRAIN_LAST_N_BLOCKS="${DIT_TRAIN_LAST_N_BLOCKS_VALUE}" \
    FDVLA_MATCHED_SEMANTIC_PUBLISH_INTERVAL_FRAMES="${SEMANTIC_PUBLISH_INTERVAL_VALUE}" \
    PPO_MAX_STEPS="${PPO_STEPS}" \
        bash "${SCRIPT_DIR}/run_fdvla_matched_start_ppo_compare.sh"

    seed_end=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    printf '%s\tcomplete\t%s\t%s\t%s\n' "${seed}" "${seed_start}" "${seed_end}" "${seed_root}" \
        >>"${FORMAL_ROOT}/formal_runs.tsv"
done

printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >>"${FORMAL_ROOT}/formal_protocol.env"
printf '%s\n' "${FORMAL_ROOT}"
