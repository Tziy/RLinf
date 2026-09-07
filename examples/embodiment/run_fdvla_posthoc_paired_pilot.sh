#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
source "${SCRIPT_DIR}/fdvla_binary_common.sh"

fdvla_require_inputs

RUN_ID=${FDVLA_POSTHOC_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_POSTHOC_RUN_ROOT:-"${REPO_DIR}/logs/fdvla_posthoc_pair/${RUN_ID}"}
mkdir -p "${RUN_ROOT}"
PAIR_WORKTREE_SHA256=$(fdvla_worktree_sha256)
printf '%s\n' "${PAIR_WORKTREE_SHA256}" >"${RUN_ROOT}/pair_worktree_sha256.txt"

assert_pair_worktree_frozen() {
    local phase=$1
    local current_sha
    current_sha=$(fdvla_worktree_sha256)
    if [[ "${current_sha}" != "${PAIR_WORKTREE_SHA256}" ]]; then
        printf 'FDVLA paired worktree changed during %s: expected=%s actual=%s\n' \
            "${phase}" "${PAIR_WORKTREE_SHA256}" "${current_sha}" >&2
        return 1
    fi
}

# A short, discriminative Task-0 pilot. Both arms execute the same post-hoc
# packet collection/replay work; only the auxiliary weight differs (0 vs 0.01).
export TRAIN_TASK_ID_FILTER=${TRAIN_TASK_ID_FILTER:-'[0]'}
export EVAL_TASK_ID_FILTER=${EVAL_TASK_ID_FILTER:-'[0]'}
export BALANCE_TRAIN_TASK_ASSIGNMENT=true
export PRESERVE_TRAIN_TASK_ASSIGNMENT=true
export UNIQUE_TRAIN_TRIAL_ASSIGNMENT=true
export TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS:-60}
export EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-48}
export TARGET_TRAIN_ENVS=${TARGET_TRAIN_ENVS:-${TRAIN_NUM_ENVS}}
export TARGET_EVAL_ENVS=${TARGET_EVAL_ENVS:-${EVAL_NUM_ENVS}}
export TRAIN_MAX_EPISODE_STEPS=${TRAIN_MAX_EPISODE_STEPS:-480}
export TRAIN_ROLLOUT_STEPS=${TRAIN_ROLLOUT_STEPS:-256}
export EVAL_MAX_EPISODE_STEPS=${EVAL_MAX_EPISODE_STEPS:-480}
export EVAL_ROLLOUT_STEPS=${EVAL_ROLLOUT_STEPS:-480}
export TRAIN_ROLLOUT_EPOCH=${TRAIN_ROLLOUT_EPOCH:-4}
export PPO_MAX_STEPS=${PPO_MAX_STEPS:-10}
export PPO_GLOBAL_BATCH_SIZE=${PPO_GLOBAL_BATCH_SIZE:-960}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
export PPO_UPDATE_EPOCHS=${PPO_UPDATE_EPOCHS:-1}
export PPO_ACTOR_LR=${PPO_ACTOR_LR:-5.0e-7}
export PPO_VALUE_LR=${PPO_VALUE_LR:-2.0e-5}
export PPO_KL_BETA=${PPO_KL_BETA:-0.0}
export FDVLA_ACTOR_MODEL_PRECISION=${FDVLA_ACTOR_MODEL_PRECISION:-bf16}
export FDVLA_ROLLOUT_MODEL_PRECISION=${FDVLA_ROLLOUT_MODEL_PRECISION:-bf16}
export DIT_TRAIN_LAST_N_BLOCKS=${DIT_TRAIN_LAST_N_BLOCKS:-4}
export TRAIN_VALUE_HEAD_WITH_DIT_ONLY=${TRAIN_VALUE_HEAD_WITH_DIT_ONLY:-true}
export PPO_EVAL_BEFORE_TRAINING=${PPO_EVAL_BEFORE_TRAINING:-true}
export PPO_VAL_INTERVAL=${PPO_VAL_INTERVAL:-5}
export PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:-5}
export ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-16}
export TRAIN_EXECUTION_HORIZON=${TRAIN_EXECUTION_HORIZON:-2}
export EVAL_EXECUTION_HORIZON=${EVAL_EXECUTION_HORIZON:-2}
export SEMANTIC_MID_CHUNK_FRAME=${SEMANTIC_MID_CHUNK_FRAME:-4}
export SEMANTIC_PUBLISH_INTERVAL_FRAMES=${SEMANTIC_PUBLISH_INTERVAL_FRAMES:-8}
export SEMANTIC_MID_CHUNK_PUBLISH=${SEMANTIC_MID_CHUNK_PUBLISH:-false}
export DENOISING_STEPS=${DENOISING_STEPS:-4}
export TRAIN_SEED=${TRAIN_SEED:-0}
export EVAL_SEED=${EVAL_SEED:-0}
export EVAL_NOISE_SEED=${EVAL_NOISE_SEED:-2026}
export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false
# Training consumes the latest completed packet and never waits for exact age.
# Delay diversity is constructed after rollout from buffered semantic tensors.
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
export POSTHOC_REQUIRE_NONBLOCKING_ROLLOUT=true
export POSTHOC_SEMANTIC_DELAY_BANK_ENABLED=false
export SEMANTIC_ENV_BOUNDARY_PUBLISH=false
# Fixed-age evaluation is a separate endpoint and cannot change train retention.
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=4
export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
export POSTHOC_REPLAY_MICROBATCH_INTERVAL=${POSTHOC_REPLAY_MICROBATCH_INTERVAL:-8}
export POSTHOC_REPLAY_MICROBATCH_OFFSET=${POSTHOC_REPLAY_MICROBATCH_OFFSET:-0}

run_arm() {
    local method=$1
    local dry_run=${DRY_RUN:-false}
    local slug=${method,,}
    slug=${slug//-/_}
    local run_dir="${RUN_ROOT}/${method}"
    mkdir -p "${run_dir}"
    assert_pair_worktree_frozen "before_${method}"

    export FDVLA_RUN_TIMESTAMP="${RUN_ID}_${method}"
    export FDVLA_RUN_LOG_DIR="${run_dir}"
    export RLINF_LOG_DIR="${run_dir}"
    export FDVLA_METADATA_DIR="${run_dir}/fdvla_metadata_${method}"
    export FDVLA_METHOD_ID="${method}"
    export PPO_EXPERIMENT_NAME="fdvla_posthoc_pair_${slug}"
    # Ray's Unix-domain socket paths must remain below 108 bytes.
    export RAY_TMPDIR="/tmp/fp_${slug}"

    local start_ns end_ns
    start_ns=$(date +%s%N)
    env -u POSTHOC_SEMANTIC_DELAY_ENABLED \
        -u POSTHOC_SEMANTIC_DELAY_WEIGHT \
        -u POSTHOC_FORCE_REPLAY_FORWARD \
        -u POSTHOC_ALLOW_HINDSIGHT_COMPLETION \
        METHOD="${method}" STAGE=pilot DRY_RUN="${dry_run}" \
        bash "${SCRIPT_DIR}/run_fdvla_sim_matrix.sh" \
        >"${run_dir}/launcher.log" 2>&1
    end_ns=$(date +%s%N)
    assert_pair_worktree_frozen "after_${method}"
    if [[ "${dry_run}" == true ]]; then
        return 0
    fi
    if rg -q \
        'Exiting main process due to a failure|Error executing job with overrides' \
        "${run_dir}/launcher.log"; then
        echo "Detected worker failure in ${method}; refusing to launch the next arm." >&2
        return 1
    fi
    awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
        'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
        >"${run_dir}/process_time.txt"

    python "${REPO_DIR}/examples/analysis/summarize_fdvla_training_run.py" \
        --run-dir "${run_dir}" \
        --run-log "${run_dir}/launcher.log" \
        --output-dir "${run_dir}/summary"
}

run_arm D-PPO-PosthocControl
run_arm D-PPO-PosthocAug

if [[ "${DRY_RUN:-false}" == true ]]; then
    printf '%s\n' "${RUN_ROOT}"
    exit 0
fi

for identity_file in git_sha.txt worktree_sha256.txt checkpoint_sha256.txt \
    backbone_sha256.txt; do
    cmp \
        "${RUN_ROOT}/D-PPO-PosthocControl/fdvla_metadata_D-PPO-PosthocControl/${identity_file}" \
        "${RUN_ROOT}/D-PPO-PosthocAug/fdvla_metadata_D-PPO-PosthocAug/${identity_file}"
done

python "${REPO_DIR}/examples/analysis/summarize_fdvla_posthoc_pair.py" \
    --run-root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/paired_summary.json" \
    --expected-replay-fraction 0.125

printf '%s\n' "${RUN_ROOT}"
