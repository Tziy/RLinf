#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
GR00T_REPO=$(cd "${REPO_DIR}/.." && pwd)

BATCHING_ENV=${FDVLA_MATCHED_BATCHING_ENV:-}
if [[ -n "${BATCHING_ENV}" ]]; then
    if [[ ! -f "${BATCHING_ENV}" ]]; then
        echo "Missing selected batching manifest: ${BATCHING_ENV}" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${BATCHING_ENV}"
else
    selected_batching=unregistered_batch1
    semantic_batch_max_requests=1
    semantic_batch_target_requests=1
    semantic_batch_wait_ms=0
fi
: "${selected_batching:?Missing selected_batching}"
: "${semantic_batch_max_requests:?Missing semantic_batch_max_requests}"
: "${semantic_batch_target_requests:?Missing semantic_batch_target_requests}"
: "${semantic_batch_wait_ms:?Missing semantic_batch_wait_ms}"

SELECTED_ENV=${FDVLA_MATCHED_SELECTED_ENV:?Set FDVLA_MATCHED_SELECTED_ENV}
if [[ ! -f "${SELECTED_ENV}" ]]; then
    echo "Missing selected-start manifest: ${SELECTED_ENV}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${SELECTED_ENV}"
: "${C_selected_checkpoint:?Missing C_selected_checkpoint in selected manifest}"
: "${D_selected_checkpoint:?Missing D_selected_checkpoint in selected manifest}"
: "${C_selected_checkpoint_sha256:?Missing C_selected_checkpoint_sha256 in selected manifest}"
: "${D_selected_checkpoint_sha256:?Missing D_selected_checkpoint_sha256 in selected manifest}"
: "${selection_policy:?Missing selection_policy in selected manifest}"
: "${selection_max_deviation:?Missing selection_max_deviation in selected manifest}"
: "${selection_max_gap:?Missing selection_max_gap in selected manifest}"
C_START_TRIALS=${C_selected_trials:-48}
D_START_TRIALS=${D_selected_trials:-48}
if ((C_START_TRIALS != D_START_TRIALS)); then
    echo "Refusing to start PPO: C/D start manifests use different trial counts (${C_START_TRIALS} vs ${D_START_TRIALS})." >&2
    exit 2
fi
START_TRIALS=${C_START_TRIALS}
if [[ -z "${selected_start_gap:-}" ]]; then
    selected_start_gap=$((C_selected_successes - D_selected_successes))
    if ((selected_start_gap < 0)); then
        selected_start_gap=$((-selected_start_gap))
    fi
fi
MAX_START_DEVIATION=${FDVLA_MATCHED_MAX_START_DEVIATION_SUCCESSES:-4}
MAX_START_GAP=${FDVLA_MATCHED_MAX_START_GAP_SUCCESSES:-4}
if [[ "${selection_policy}" != joint_eligible_pair_v1 \
    || "${selection_max_deviation}" != "${MAX_START_DEVIATION}" \
    || "${selection_max_gap}" != "${MAX_START_GAP}" ]]; then
    echo "Selected-start policy or thresholds do not match the formal launcher." >&2
    exit 2
fi
for arm in C D; do
    successes_var=${arm}_selected_successes
    arm_successes=${!successes_var}
    deviation=$((arm_successes - selection_target_successes))
    if ((deviation < 0)); then
        deviation=$((-deviation))
    fi
    if ((deviation > MAX_START_DEVIATION)); then
        echo "Refusing to start PPO: ${arm} start ${arm_successes}/${START_TRIALS} is outside target ${selection_target_successes}/${START_TRIALS} +/- ${MAX_START_DEVIATION}." >&2
        exit 2
    fi
done
if ((selected_start_gap > MAX_START_GAP)); then
    echo "Refusing to start PPO: selected C/D start gap ${selected_start_gap}/${START_TRIALS} exceeds ${MAX_START_GAP}." >&2
    exit 2
fi

BACKBONE_PATH=${GR00T_BACKBONE_PATH:-${GR00T_REPO}/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}
RUN_ID=${FDVLA_MATCHED_PPO_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_MATCHED_PPO_RUN_ROOT:-${REPO_DIR}/logs/fdvla_matched_start/${RUN_ID}/ppo_compare}
FINAL_EVAL_NOISE_SEED=${FDVLA_MATCHED_PPO_EVAL_NOISE_SEED:-14026}
RAY_SHORT_PREFIX=${FDVLA_MATCHED_PPO_RAY_SHORT_PREFIX:-fmppo}
PPO_EXECUTION_HORIZON=${FDVLA_MATCHED_PPO_EXECUTION_HORIZON:-2}
PPO_TRAIN_SEED_VALUE=${FDVLA_MATCHED_PPO_TRAIN_SEED:-0}
PPO_EVAL_SEED_VALUE=${FDVLA_MATCHED_PPO_EVAL_SEED:-0}
PPO_GAMMA_VALUE=${FDVLA_MATCHED_PPO_GAMMA:-0.99}
PPO_GAE_LAMBDA_VALUE=${FDVLA_MATCHED_PPO_GAE_LAMBDA:-0.95}
PPO_ACTOR_LR_VALUE=${FDVLA_MATCHED_PPO_ACTOR_LR:-5.0e-7}
PPO_VALUE_LR_VALUE=${FDVLA_MATCHED_PPO_VALUE_LR:-2.0e-5}
PPO_CRITIC_WARMUP_STEPS_VALUE=${FDVLA_MATCHED_PPO_CRITIC_WARMUP_STEPS:-0}
PPO_VAL_INTERVAL_VALUE=${FDVLA_MATCHED_PPO_VAL_INTERVAL:-5}
PPO_SAVE_INTERVAL_VALUE=${FDVLA_MATCHED_PPO_SAVE_INTERVAL:-5}
PPO_GLOBAL_BATCH_SIZE_VALUE=${FDVLA_MATCHED_PPO_GLOBAL_BATCH_SIZE:-960}
PPO_MICRO_BATCH_SIZE_VALUE=${FDVLA_MATCHED_PPO_MICRO_BATCH_SIZE:-2}
PPO_UPDATE_EPOCHS_VALUE=${FDVLA_MATCHED_PPO_UPDATE_EPOCHS:-1}
PPO_KL_BETA_VALUE=${FDVLA_MATCHED_PPO_KL_BETA:-0.0}
ACTOR_PRECISION_VALUE=${FDVLA_MATCHED_ACTOR_MODEL_PRECISION:-bf16}
ROLLOUT_PRECISION_VALUE=${FDVLA_MATCHED_ROLLOUT_MODEL_PRECISION:-bf16}
DIT_TRAIN_LAST_N_BLOCKS_VALUE=${FDVLA_MATCHED_DIT_TRAIN_LAST_N_BLOCKS:-4}
SEMANTIC_PUBLISH_INTERVAL_VALUE=${FDVLA_MATCHED_SEMANTIC_PUBLISH_INTERVAL_FRAMES:-8}
SELECTION_NOISE_SEED=${FDVLA_MATCHED_SELECTION_NOISE_SEED:-unknown}
MATCHED_RUN_ARMS=${FDVLA_MATCHED_RUN_ARMS:-C,D}
case "${MATCHED_RUN_ARMS}" in
    C | D | C,D | D,C) ;;
    *)
        echo "FDVLA_MATCHED_RUN_ARMS must be one of C, D, C,D, or D,C; got ${MATCHED_RUN_ARMS}." >&2
        exit 2
        ;;
esac

# Freeze one source snapshot across selection provenance, both arms, and all seeds.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/fdvla_binary_common.sh"
PAIR_WORKTREE_SHA256=$(fdvla_worktree_sha256)

assert_pair_worktree_frozen() {
    local phase=$1
    local current_sha
    current_sha=$(fdvla_worktree_sha256)
    if [[ "${current_sha}" != "${PAIR_WORKTREE_SHA256}" ]]; then
        printf "FDVLA matched run worktree changed during %s: expected=%s actual=%s\n" \
            "${phase}" "${PAIR_WORKTREE_SHA256}" "${current_sha}" >&2
        return 2
    fi
}

assert_selection_provenance() {
    local arm=$1
    local jsonl_var=${arm}_selection_jsonl
    local jsonl=${!jsonl_var:-}
    if [[ -z "${jsonl}" || ! -f "${jsonl}" ]]; then
        echo "Missing ${arm} selection JSONL provenance: ${jsonl}" >&2
        return 2
    fi
    local metadata_dir selection_sha selection_noise selection_checkpoint_sha
    local selected_checkpoint_var selected_checkpoint selected_checkpoint_sha_var
    local selected_checkpoint_sha current_checkpoint_sha
    metadata_dir=$(dirname "${jsonl}")/fdvla_metadata
    selection_sha=$(<"${metadata_dir}/worktree_sha256.txt")
    selection_checkpoint_sha=$(<"${metadata_dir}/checkpoint_sha256.txt")
    selected_checkpoint_var=${arm}_selected_checkpoint
    selected_checkpoint=${!selected_checkpoint_var}
    selected_checkpoint_sha_var=${arm}_selected_checkpoint_sha256
    selected_checkpoint_sha=${!selected_checkpoint_sha_var}
    current_checkpoint_sha=$(fdvla_checkpoint_sha256 "${selected_checkpoint}")
    selection_noise=$(sed -n "s/^policy_noise_seed=//p" \
        "${metadata_dir}/realtime_condition.env")
    if [[ "${selection_sha}" != "${PAIR_WORKTREE_SHA256}" ]]; then
        echo "${arm} selection and PPO worktree fingerprints differ." >&2
        return 2
    fi
    if [[ "${selection_checkpoint_sha}" != "${selected_checkpoint_sha}" \
        || "${current_checkpoint_sha}" != "${selected_checkpoint_sha}" ]]; then
        echo "${arm} selected checkpoint SHA256 mismatch." >&2
        return 2
    fi
    if [[ "${selection_noise}" != "${SELECTION_NOISE_SEED}" ]]; then
        echo "${arm} selection noise seed mismatch: ${selection_noise} != ${SELECTION_NOISE_SEED}" >&2
        return 2
    fi
}

if [[ "${SELECTION_NOISE_SEED}" == "${FINAL_EVAL_NOISE_SEED}" ]]; then
    echo "Selection and training-eval policy-noise seeds must be disjoint." >&2
    exit 2
fi
assert_pair_worktree_frozen preflight
assert_selection_provenance C
assert_selection_provenance D

for required_path in "${C_selected_checkpoint}" "${D_selected_checkpoint}" "${BACKBONE_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required input does not exist: ${required_path}" >&2
        exit 2
    fi
done
mkdir -p "${RUN_ROOT}"
cp "${SELECTED_ENV}" "${RUN_ROOT}/selected_start_manifest.env"
git -C "${REPO_DIR}" status --short >"${RUN_ROOT}/git_status_before.txt"
printf "%s\n" "${PAIR_WORKTREE_SHA256}" >"${RUN_ROOT}/pair_worktree_sha256.txt"

export TRAIN_TASK_ID_FILTER='[0]'
export EVAL_TASK_ID_FILTER='[0]'
export BALANCE_TRAIN_TASK_ASSIGNMENT=true
export PRESERVE_TRAIN_TASK_ASSIGNMENT=true
export UNIQUE_TRAIN_TRIAL_ASSIGNMENT=true
export TRAIN_AUTO_RESET=true
export TRAIN_NUM_ENVS=60
export EVAL_NUM_ENVS=48
export TARGET_TRAIN_ENVS=60
export TARGET_EVAL_ENVS=48
export TRAIN_MAX_EPISODE_STEPS=480
export TRAIN_ROLLOUT_STEPS=256
export EVAL_MAX_EPISODE_STEPS=480
export EVAL_ROLLOUT_STEPS=480
export TRAIN_ROLLOUT_EPOCH=4
export EVAL_ROLLOUT_EPOCH=1
export EVAL_AUTO_RESET=false
export EVAL_IGNORE_TERMINATIONS=false
export PPO_MAX_STEPS=${PPO_MAX_STEPS:-50}
export PPO_GLOBAL_BATCH_SIZE="${PPO_GLOBAL_BATCH_SIZE_VALUE}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE_VALUE}"
export PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS_VALUE}"
export PPO_KL_BETA="${PPO_KL_BETA_VALUE}"
export FDVLA_ACTOR_MODEL_PRECISION="${ACTOR_PRECISION_VALUE}"
export FDVLA_ROLLOUT_MODEL_PRECISION="${ROLLOUT_PRECISION_VALUE}"
export DIT_TRAIN_LAST_N_BLOCKS="${DIT_TRAIN_LAST_N_BLOCKS_VALUE}"
export SEMANTIC_ADAPTER_ONLY_TRAIN=false
export TRAIN_VALUE_HEAD_WITH_DIT_ONLY=true
export PPO_ACTOR_LR="${PPO_ACTOR_LR_VALUE}"
export PPO_VALUE_LR="${PPO_VALUE_LR_VALUE}"
export PPO_CRITIC_WARMUP_STEPS="${PPO_CRITIC_WARMUP_STEPS_VALUE}"
export PPO_EVAL_BEFORE_TRAINING=true
export PPO_VAL_INTERVAL="${PPO_VAL_INTERVAL_VALUE}"
export PPO_SAVE_INTERVAL="${PPO_SAVE_INTERVAL_VALUE}"
export PPO_SUCCESS_EARLY_STOP_ENABLED=false
export ACTION_CHUNK_SIZE=16
export TRAIN_EXECUTION_HORIZON="${PPO_EXECUTION_HORIZON}"
export EVAL_EXECUTION_HORIZON="${PPO_EXECUTION_HORIZON}"
export DENOISING_STEPS=4
export PPO_GAMMA="${PPO_GAMMA_VALUE}"
export PPO_GAE_LAMBDA="${PPO_GAE_LAMBDA_VALUE}"
export TRAIN_SEED="${PPO_TRAIN_SEED_VALUE}"
export EVAL_SEED="${PPO_EVAL_SEED_VALUE}"
export EVAL_NOISE_SEED="${FINAL_EVAL_NOISE_SEED}"
export DETERMINISTIC_EVAL_NOISE=true
export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false
# Natural-latest D training: no exact-age selection or blocking in rollout.
# C/D evaluation is exact age 0 so learning curves share the same semantic-age condition.
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=0
export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_FETCH_TARGET_AGE_FRAMES=-1
export SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES=-1
export SEMANTIC_PUBLISH_INTERVAL_FRAMES="${SEMANTIC_PUBLISH_INTERVAL_VALUE}"
export SEMANTIC_MID_CHUNK_PUBLISH=false
export SEMANTIC_MID_CHUNK_FRAME=4
export SEMANTIC_MID_CHUNK_MIN_FRAME=1
export SEMANTIC_TEXT_PADDING_TOKENS=160
export ONLY_EVAL=false

prepare_short_ray_path() {
    local arm=$1
    local run_dir=$2
    local arm_slug=${arm,,}
    arm_slug=${arm_slug//[^a-z0-9]/}
    local short_path="/tmp/${RAY_SHORT_PREFIX}${arm_slug}"
    mkdir -p "${run_dir}/ray_tmp"
    if [[ -L "${short_path}" ]]; then
        local current_target
        current_target=$(readlink "${short_path}")
        if [[ "${current_target}" != "${run_dir}/ray_tmp" ]]; then
            echo "Short Ray path points elsewhere: ${short_path} -> ${current_target}" >&2
            return 2
        fi
    elif [[ -e "${short_path}" ]]; then
        echo "Short Ray path exists and is not a symlink: ${short_path}" >&2
        return 2
    else
        ln -s "${run_dir}/ray_tmp" "${short_path}"
    fi
    export RAY_TMPDIR="${short_path}"
}

run_arm() {
    local arm=$1
    local checkpoint=$2
    local launcher=$3
    local run_dir=${RUN_ROOT}/${arm}
    case "${arm}" in
        C-PPO)
            export REQUIRE_PACKET_AGE_INPUT=false
            export INITIALIZE_PACKET_AGE_ADAPTER=false
            export ACTION_HISTORY_LENGTH=0
            export ZERO_INIT_NEW_DELAY_ADAPTERS=false
            ;;
        D-PPO)
            export REQUIRE_PACKET_AGE_INPUT=true
            export INITIALIZE_PACKET_AGE_ADAPTER=true
            export ACTION_HISTORY_LENGTH=4
            export ZERO_INIT_NEW_DELAY_ADAPTERS=true
            ;;
        *)
            echo "Unknown matched PPO arm: ${arm}" >&2
            return 2
            ;;
    esac
    mkdir -p "${run_dir}"
    prepare_short_ray_path "${arm}" "${run_dir}"

    export GR00T_MODEL_PATH="${checkpoint}"
    export GR00T_BACKBONE_PATH="${BACKBONE_PATH}"
    export FDVLA_RUN_TIMESTAMP="${RUN_ID}_${arm}"
    export FDVLA_RUN_LOG_DIR="${run_dir}"
    export RLINF_LOG_DIR="${run_dir}"
    export FDVLA_METADATA_DIR="${run_dir}/fdvla_metadata"
    export PPO_EXPERIMENT_NAME="fdvla_matched_start_${arm}"
    export FDVLA_METHOD_ID="${arm}"
    unset PPO_CKPT_PATH PPO_RESUME_DIR

    {
        printf 'run_id=%s\n' "${RUN_ID}"
        printf 'arm=%s\n' "${arm}"
        printf 'initial_checkpoint=%s\n' "${checkpoint}"
        printf 'start_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        printf 'selection_noise_seed=%s\n' "${SELECTION_NOISE_SEED}"
        printf 'training_eval_noise_seed=%s\n' "${FINAL_EVAL_NOISE_SEED}"
        printf 'train_seed=%s\n' "${TRAIN_SEED}"
        printf 'eval_seed=%s\n' "${EVAL_SEED}"
        printf 'train_execution_horizon=%s\n' "${TRAIN_EXECUTION_HORIZON}"
        printf 'eval_execution_horizon=%s\n' "${EVAL_EXECUTION_HORIZON}"
        printf 'ppo_gamma=%s\n' "${PPO_GAMMA}"
        printf 'ppo_gae_lambda=%s\n' "${PPO_GAE_LAMBDA}"
        printf 'ppo_actor_lr=%s\n' "${PPO_ACTOR_LR}"
        printf 'ppo_value_lr=%s\n' "${PPO_VALUE_LR}"
        printf 'ppo_critic_warmup_steps=%s\n' "${PPO_CRITIC_WARMUP_STEPS}"
        printf 'ppo_val_interval=%s\n' "${PPO_VAL_INTERVAL}"
        printf 'ppo_save_interval=%s\n' "${PPO_SAVE_INTERVAL}"
        printf 'ppo_global_batch_size=%s\n' "${PPO_GLOBAL_BATCH_SIZE}"
        printf 'ppo_micro_batch_size=%s\n' "${PPO_MICRO_BATCH_SIZE}"
        printf 'ppo_update_epochs=%s\n' "${PPO_UPDATE_EPOCHS}"
        printf 'ppo_kl_beta=%s\n' "${PPO_KL_BETA}"
        printf 'actor_precision=%s\n' "${FDVLA_ACTOR_MODEL_PRECISION}"
        printf 'rollout_precision=%s\n' "${FDVLA_ROLLOUT_MODEL_PRECISION}"
        printf 'dit_train_last_n_blocks=%s\n' "${DIT_TRAIN_LAST_N_BLOCKS}"
        printf 'semantic_publish_interval_frames=%s\n' "${SEMANTIC_PUBLISH_INTERVAL_FRAMES}"
        printf 'require_packet_age_input=%s\n' "${REQUIRE_PACKET_AGE_INPUT}"
        printf 'initialize_packet_age_adapter=%s\n' "${INITIALIZE_PACKET_AGE_ADAPTER}"
        printf 'action_history_length=%s\n' "${ACTION_HISTORY_LENGTH}"
        printf 'zero_init_new_delay_adapters=%s\n' "${ZERO_INIT_NEW_DELAY_ADAPTERS}"
        printf 'run_arms=%s\n' "${MATCHED_RUN_ARMS}"
        printf 'semantic_train_random_age_min_frames=%s\n' "${SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES}"
        printf 'semantic_train_random_age_max_frames=%s\n' "${SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES}"
        printf 'selected_batching=%s\n' "${selected_batching}"
        printf 'semantic_batch_max_requests=%s\n' "${semantic_batch_max_requests}"
        printf 'semantic_batch_target_requests=%s\n' "${semantic_batch_target_requests}"
        printf 'semantic_batch_wait_ms=%s\n' "${semantic_batch_wait_ms}"
        printf 'ppo_steps=%s\n' "${PPO_MAX_STEPS}"
    } >"${run_dir}/run.env"

    local start_ns end_ns
    start_ns=$(date +%s%N)
    assert_pair_worktree_frozen "before_${arm}"
    bash "${launcher}" >"${run_dir}/launcher.log" 2>&1
    assert_pair_worktree_frozen "after_${arm}"
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

# Equal four-GPU budget. Coupled uses all four GPUs for local VLM + DiT.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ACTOR_GPU_PLACEMENT=0-3
export ROLLOUT_GPU_PLACEMENT=0-3
export ENV_GPU_PLACEMENT=0-3
if [[ ",${MATCHED_RUN_ARMS}," == *,C,* ]]; then
    run_arm C-PPO "${C_selected_checkpoint}" "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh"
fi

# Decoupled uses four colocated frozen semantic servers and four DiT/actor ranks.
# The action path never waits for an exact age; it consumes the latest complete packet.
export ALLOWED_GPU_IDS=0,1,2,3
export SEMANTIC_GPU_IDS=0,1,2,3
export DIT_GPU_IDS=0,1,2,3
export ACTOR_GPU_IDS=0,1,2,3
export NUM_SEMANTIC_GPUS=4
export MAX_USED_MIB=1024
export ENV_WORKERS_PER_GPU=1
export SEMANTIC_PREPROCESS_PROXY=false
export SEMANTIC_BATCH_MAX_REQUESTS="${semantic_batch_max_requests}"
export SEMANTIC_BATCH_TARGET_REQUESTS="${semantic_batch_target_requests}"
export SEMANTIC_BATCH_TARGET_ENVS=0
export SEMANTIC_BATCH_WAIT_MS="${semantic_batch_wait_ms}"
export SEMANTIC_RPC_BATCH_WAIT_MS=2
export SEMANTIC_BOOTSTRAP_TARGET_ENVS=0
export SEMANTIC_BOOTSTRAP_WAIT_MS=30000
export SEMANTIC_CACHE_HISTORY_SIZE=32
export COLOCATED_SEMANTIC_FETCH_PAUSE_MS=0
if [[ ",${MATCHED_RUN_ARMS}," == *,D,* ]]; then
    run_arm D-PPO "${D_selected_checkpoint}" "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh"
fi

assert_pair_worktree_frozen complete
printf '%s\n' "${RUN_ROOT}"
