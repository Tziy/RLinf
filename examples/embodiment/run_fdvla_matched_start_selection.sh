#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLINF_REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
GR00T_REPO=$(cd "${RLINF_REPO}/.." && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/fdvla_binary_common.sh"


GRID_ROOT=${FDVLA_MATCHED_SFT_RUN_ROOT:?Set FDVLA_MATCHED_SFT_RUN_ROOT to the completed grid}
C_GRID_ROOT=${FDVLA_MATCHED_C_GRID_ROOT:-${GRID_ROOT}}
D_GRID_ROOT=${FDVLA_MATCHED_D_GRID_ROOT:-${GRID_ROOT}}
SELECTION_ROOT=${FDVLA_MATCHED_SELECTION_ROOT:-$(dirname "${GRID_ROOT}")/selection}
BACKBONE_PATH=${GR00T_BACKBONE_PATH:-${GR00T_REPO}/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}
SELECTION_NOISE_SEED=${FDVLA_MATCHED_SELECTION_NOISE_SEED:-14026}
TARGET_SUCCESSES=${FDVLA_MATCHED_TARGET_SUCCESSES:-13}
MAX_START_DEVIATION=${FDVLA_MATCHED_MAX_START_DEVIATION_SUCCESSES:-4}
MAX_START_GAP=${FDVLA_MATCHED_MAX_START_GAP_SUCCESSES:-4}
MAX_STEPS=${FDVLA_MATCHED_SFT_MAX_STEPS:-1000}
SAVE_INTERVAL=${FDVLA_MATCHED_SFT_SAVE_INTERVAL:-200}
MIN_STEP=${FDVLA_MATCHED_SELECTION_MIN_STEP:-${SAVE_INTERVAL}}
SELECTION_STEPS=${FDVLA_MATCHED_SELECTION_STEPS:-}
SELECTION_ARMS=${FDVLA_MATCHED_SELECTION_ARMS:-"C D"}
BATCHING_ENV=${FDVLA_MATCHED_BATCHING_ENV:-}
if [[ " ${SELECTION_ARMS} " == *" D "* ]]; then
    if [[ -z "${BATCHING_ENV}" || ! -f "${BATCHING_ENV}" ]]; then
        echo "D matched-start selection requires FDVLA_MATCHED_BATCHING_ENV." >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${BATCHING_ENV}"
    : "${selected_batching:?Missing selected_batching in batching manifest}"
    : "${semantic_batch_max_requests:?Missing semantic_batch_max_requests}"
    : "${semantic_batch_target_requests:?Missing semantic_batch_target_requests}"
    : "${semantic_batch_wait_ms:?Missing semantic_batch_wait_ms}"
fi
RAY_SHORT_PREFIX=${FDVLA_MATCHED_RAY_SHORT_PREFIX:-fm}
EVAL_TIMEOUT_SECONDS=${FDVLA_MATCHED_EVAL_TIMEOUT_SECONDS:-600}
EVAL_MAX_ATTEMPTS=${FDVLA_MATCHED_EVAL_MAX_ATTEMPTS:-2}
SELECTION_TASK_FILTER=${FDVLA_MATCHED_SELECTION_TASK_FILTER:-'[0]'}
SELECTION_NUM_TRIALS=${FDVLA_MATCHED_SELECTION_NUM_TRIALS:-48}
SELECTION_TASK_COUNT=${FDVLA_MATCHED_SELECTION_TASK_COUNT:-1}
SELECTION_EXECUTION_HORIZON=${FDVLA_MATCHED_SELECTION_EXECUTION_HORIZON:-2}

if (( SELECTION_NUM_TRIALS % SELECTION_TASK_COUNT != 0 )); then
    echo "SELECTION_NUM_TRIALS must be divisible by SELECTION_TASK_COUNT." >&2
    exit 2
fi

C_EXPERIMENT=${FDVLA_MATCHED_C_EXPERIMENT:-libero10_coupled_current_s4_a4_grid${MAX_STEPS}}
D_EXPERIMENT=${FDVLA_MATCHED_D_EXPERIMENT:-libero10_decoupled_delay0to8_s4_a4_grid${MAX_STEPS}}
RAY_SHORT_ROOT=${FDVLA_MATCHED_RAY_SHORT_ROOT:-/tmp}
C_CHECKPOINT_OVERRIDE=${FDVLA_MATCHED_C_CHECKPOINT_OVERRIDE:-}
D_CHECKPOINT_OVERRIDE=${FDVLA_MATCHED_D_CHECKPOINT_OVERRIDE:-}
RESULTS_TSV=${SELECTION_ROOT}/candidates.tsv
SELECTED_ENV=${SELECTION_ROOT}/selected.env

mkdir -p "${SELECTION_ROOT}"
if [[ -n "${BATCHING_ENV}" ]]; then
    cp "${BATCHING_ENV}" "${SELECTION_ROOT}/selected_batching_manifest.env"
    sha256sum "${SELECTION_ROOT}/selected_batching_manifest.env" \
        >"${SELECTION_ROOT}/selected_batching_manifest.sha256"
fi
SELECTION_WORKTREE_SHA256=$(fdvla_worktree_sha256)
SELECTION_WORKTREE_PATH=${SELECTION_ROOT}/selection_worktree_sha256.txt
if [[ -e "${SELECTION_WORKTREE_PATH}" ]]; then
    RECORDED_SELECTION_WORKTREE_SHA256=$(<"${SELECTION_WORKTREE_PATH}")
    if [[ "${RECORDED_SELECTION_WORKTREE_SHA256}" != "${SELECTION_WORKTREE_SHA256}" ]]; then
        echo "Selection root belongs to a different worktree; use a new FDVLA_MATCHED_SELECTION_ROOT." >&2
        exit 2
    fi
else
    printf '%s\n' "${SELECTION_WORKTREE_SHA256}" >"${SELECTION_WORKTREE_PATH}"
fi

assert_selection_worktree_frozen() {
    local phase=$1
    local current_sha
    current_sha=$(fdvla_worktree_sha256)
    if [[ "${current_sha}" != "${SELECTION_WORKTREE_SHA256}" ]]; then
        printf 'FDVLA selection worktree changed during %s: expected=%s actual=%s\n' \
            "${phase}" "${SELECTION_WORKTREE_SHA256}" "${current_sha}" >&2
        return 2
    fi
}

assert_candidate_provenance() {
    local arm=$1
    local step=$2
    local eval_dir=$3
    local metadata_dir="${eval_dir}/fdvla_metadata"
    local recorded_sha recorded_noise recorded_horizon recorded_fixed_age
    recorded_sha=$(<"${metadata_dir}/worktree_sha256.txt")
    recorded_noise=$(sed -n 's/^policy_noise_seed=//p' \
        "${metadata_dir}/realtime_condition.env")
    recorded_horizon=$(sed -n 's/^eval_execution_horizon=//p' \
        "${metadata_dir}/realtime_condition.env")
    recorded_fixed_age=$(sed -n 's/^semantic_eval_fixed_age_frames=//p' \
        "${metadata_dir}/realtime_condition.env")
    if [[ "${recorded_sha}" != "${SELECTION_WORKTREE_SHA256}" \
        || "${recorded_noise}" != "${SELECTION_NOISE_SEED}" \
        || "${recorded_horizon}" != "${SELECTION_EXECUTION_HORIZON}" \
        || "${recorded_fixed_age}" != "0" ]]; then
        echo "Candidate provenance mismatch for ${arm} step ${step}." >&2
        return 2
    fi
}

assert_selection_worktree_frozen preflight
if [[ ! -e "${RESULTS_TSV}" ]]; then
    printf 'arm\tstep\tsuccesses\ttrials\tsuccess_rate\tcheckpoint\tcheckpoint_sha256\tjsonl\n' >"${RESULTS_TSV}"
fi

checkpoint_for() {
    local arm=$1
    local step=$2
    case "${arm}" in
        C)
            if [[ -n "${C_CHECKPOINT_OVERRIDE}" ]]; then
                printf '%s\n' "${C_CHECKPOINT_OVERRIDE}"
            else
                printf '%s\n' "${C_GRID_ROOT}/C/${C_EXPERIMENT}/checkpoint-${step}"
            fi
            ;;
        D)
            if [[ -n "${D_CHECKPOINT_OVERRIDE}" ]]; then
                printf '%s\n' "${D_CHECKPOINT_OVERRIDE}"
            else
                printf '%s\n' "${D_GRID_ROOT}/D/${D_EXPERIMENT}/checkpoint-${step}"
            fi
            ;;
        *) return 2 ;;
    esac
}

apply_eval_protocol() {
    local arm=$1
    local arm_prefix=FDVLA_MATCHED_${arm}_SELECTION
    local default_require_packet_age default_initialize_packet_age
    local default_history_length default_zero_init_delay_adapters
    if [[ "${arm}" == C ]]; then
        default_require_packet_age=false
        default_initialize_packet_age=false
        default_history_length=0
        default_zero_init_delay_adapters=false
    else
        default_require_packet_age=true
        default_initialize_packet_age=true
        default_history_length=4
        default_zero_init_delay_adapters=true
    fi
    export ACTION_CHUNK_SIZE=16
    export EVAL_EXECUTION_HORIZON="${SELECTION_EXECUTION_HORIZON}"
    export DENOISING_STEPS=4
    if [[ "${SELECTION_TASK_FILTER}" == all ]]; then
        unset EVAL_TASK_ID_FILTER
    else
        export EVAL_TASK_ID_FILTER="${SELECTION_TASK_FILTER}"
    fi
    export EVAL_NUM_ENVS="${SELECTION_NUM_TRIALS}"
    export TARGET_EVAL_ENVS="${SELECTION_NUM_TRIALS}"
    export EVAL_MAX_EPISODE_STEPS=480
    export EVAL_ROLLOUT_STEPS=480
    export EVAL_ROLLOUT_EPOCH=1
    export EVAL_AUTO_RESET=false
    export EVAL_IGNORE_TERMINATIONS=false
    export EVAL_SEED=0
    export EVAL_NOISE_SEED="${SELECTION_NOISE_SEED}"
    export DETERMINISTIC_EVAL_NOISE=true
    local name value
    name=${arm_prefix}_TEXT_PADDING_TOKENS
    value=${!name:-${FDVLA_MATCHED_SELECTION_TEXT_PADDING_TOKENS:-160}}
    export SEMANTIC_TEXT_PADDING_TOKENS=${value}
    name=${arm_prefix}_REQUIRE_PACKET_AGE_INPUT
    value=${!name:-${FDVLA_MATCHED_SELECTION_REQUIRE_PACKET_AGE_INPUT:-${default_require_packet_age}}}
    export REQUIRE_PACKET_AGE_INPUT=${value}
    name=${arm_prefix}_INITIALIZE_PACKET_AGE_ADAPTER
    value=${!name:-${FDVLA_MATCHED_SELECTION_INITIALIZE_PACKET_AGE_ADAPTER:-${default_initialize_packet_age}}}
    export INITIALIZE_PACKET_AGE_ADAPTER=${value}
    name=${arm_prefix}_ACTION_HISTORY_LENGTH
    value=${!name:-${FDVLA_MATCHED_SELECTION_ACTION_HISTORY_LENGTH:-${default_history_length}}}
    export ACTION_HISTORY_LENGTH=${value}
    name=${arm_prefix}_ZERO_INIT_DELAY_ADAPTERS
    value=${!name:-${FDVLA_MATCHED_SELECTION_ZERO_INIT_DELAY_ADAPTERS:-${default_zero_init_delay_adapters}}}
    export ZERO_INIT_NEW_DELAY_ADAPTERS=${value}
    name=${arm_prefix}_FIXED_AGE_FRAMES
    value=${!name:-${FDVLA_MATCHED_SELECTION_FIXED_AGE_FRAMES:-0}}
    export SEMANTIC_EVAL_FIXED_AGE_FRAMES=${value}
    export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
    export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
    export SEMANTIC_FETCH_TARGET_AGE_FRAMES=-1
    export SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES=-1
    export SEMANTIC_MID_CHUNK_PUBLISH=true
    export SEMANTIC_MID_CHUNK_FRAME=${FDVLA_MATCHED_SELECTION_MID_CHUNK_FRAME:-1}
    export SEMANTIC_MID_CHUNK_MIN_FRAME=${FDVLA_MATCHED_SELECTION_MID_CHUNK_MIN_FRAME:-1}
    export ONLY_EVAL=true
}

prepare_short_ray_path() {
    local arm=$1
    local step=$2
    local eval_dir=$3
    local short_path="${RAY_SHORT_ROOT}/${RAY_SHORT_PREFIX}${arm,,}${step}"
    mkdir -p "${eval_dir}/ray_tmp"
    if [[ -L "${short_path}" ]]; then
        local current_target
        current_target=$(readlink "${short_path}")
        if [[ "${current_target}" != "${eval_dir}/ray_tmp" ]]; then
            echo "Short Ray path points elsewhere: ${short_path} -> ${current_target}" >&2
            return 2
        fi
    elif [[ -e "${short_path}" ]]; then
        echo "Short Ray path already exists and is not a symlink: ${short_path}" >&2
        return 2
    else
        ln -s "${eval_dir}/ray_tmp" "${short_path}"
    fi
    export RAY_TMPDIR="${short_path}"
}

run_c_candidate() {
    local step=$1
    local checkpoint=$2
    local eval_dir=$3
    apply_eval_protocol C
    prepare_short_ray_path C "${step}" "${eval_dir}"
    export GR00T_MODEL_PATH="${checkpoint}"
    export GR00T_BACKBONE_PATH="${BACKBONE_PATH}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export ACTOR_GPU_PLACEMENT=0-3
    export ROLLOUT_GPU_PLACEMENT=0-3
    export ENV_GPU_PLACEMENT=0-3
    export FDVLA_METHOD_ID="C-SFT-selection-step${step}"
    export FDVLA_RUN_TIMESTAMP="matched_C_${step}"
    export FDVLA_RUN_LOG_DIR="${eval_dir}"
    export RLINF_LOG_DIR="${eval_dir}"
    export FDVLA_METADATA_DIR="${eval_dir}/fdvla_metadata"
    export PPO_EXPERIMENT_NAME="matched_c_sft_step${step}"
    unset PPO_CKPT_PATH PPO_RESUME_DIR
    timeout --signal=INT --kill-after=30s "${EVAL_TIMEOUT_SECONDS}" \
        bash "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh" \
        >"${eval_dir}/launcher.log" 2>&1
}

run_d_candidate() {
    local step=$1
    local checkpoint=$2
    local eval_dir=$3
    apply_eval_protocol D
    prepare_short_ray_path D "${step}" "${eval_dir}"
    export GR00T_MODEL_PATH="${checkpoint}"
    export GR00T_BACKBONE_PATH="${BACKBONE_PATH}"
    export ALLOWED_GPU_IDS=0,1,2,3
    export SEMANTIC_GPU_IDS=${FDVLA_MATCHED_D_SEMANTIC_GPU_IDS:-0,1,2,3}
    export DIT_GPU_IDS=${FDVLA_MATCHED_D_DIT_GPU_IDS:-0,1,2,3}
    export ACTOR_GPU_IDS=${FDVLA_MATCHED_D_ACTOR_GPU_IDS:-0,1,2,3}
    export NUM_SEMANTIC_GPUS=${FDVLA_MATCHED_D_NUM_SEMANTIC_GPUS:-4}
    export MAX_USED_MIB=1024
    export ENV_WORKERS_PER_GPU=1
    export TARGET_TRAIN_ENVS=60
    export SEMANTIC_PREPROCESS_PROXY=false
    export SEMANTIC_BATCH_MAX_REQUESTS=${semantic_batch_max_requests}
    export SEMANTIC_BATCH_TARGET_REQUESTS=${semantic_batch_target_requests}
    export SEMANTIC_BATCH_TARGET_ENVS=0
    export SEMANTIC_BATCH_WAIT_MS=${semantic_batch_wait_ms}
    export SEMANTIC_RPC_BATCH_WAIT_MS=2
    export SEMANTIC_BOOTSTRAP_TARGET_ENVS=0
    export SEMANTIC_BOOTSTRAP_WAIT_MS=30000
    export SEMANTIC_CACHE_HISTORY_SIZE=32
    export COLOCATED_SEMANTIC_FETCH_PAUSE_MS=0
    export FDVLA_METHOD_ID="D-SFT-selection-step${step}"
    export FDVLA_RUN_TIMESTAMP="matched_D_${step}"
    export FDVLA_RUN_LOG_DIR="${eval_dir}"
    export RLINF_LOG_DIR="${eval_dir}"
    export FDVLA_METADATA_DIR="${eval_dir}/fdvla_metadata"
    export PPO_EXPERIMENT_NAME="matched_d_sft_step${step}"
    unset PPO_CKPT_PATH PPO_RESUME_DIR
    timeout --signal=INT --kill-after=30s "${EVAL_TIMEOUT_SECONDS}" \
        bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh" \
        >"${eval_dir}/launcher.log" 2>&1
}

append_result() {
    local arm=$1
    local step=$2
    local checkpoint=$3
    local jsonl=$4
    local checkpoint_sha256
    checkpoint_sha256=$(<"$(dirname "${jsonl}")/fdvla_metadata/checkpoint_sha256.txt")
    python - "${RESULTS_TSV}" "${arm}" "${step}" "${checkpoint}" "${checkpoint_sha256}" "${jsonl}" \
        "${SELECTION_NUM_TRIALS}" "${SELECTION_TASK_COUNT}" \
        "${SELECTION_EXECUTION_HORIZON}" "0" <<'PY'
from collections import Counter
import json
import pathlib
import sys

(
    result_path,
    arm,
    step,
    checkpoint,
    checkpoint_sha256,
    jsonl,
    expected_trials_text,
    expected_task_count_text,
    expected_horizon_text,
    expected_age_text,
) = sys.argv[1:]
expected_trials = int(expected_trials_text)
expected_task_count = int(expected_task_count_text)
expected_horizon = int(expected_horizon_text)
expected_age = int(expected_age_text)
rows = [json.loads(line) for line in pathlib.Path(jsonl).read_text().splitlines() if line.strip()]
if len(rows) != expected_trials:
    raise SystemExit(f"Expected {expected_trials} trials, got {len(rows)}: {jsonl}")
identities = {(row["task_id"], row["trial_id"], row["policy_noise_seed"]) for row in rows}
if len(identities) != expected_trials:
    raise SystemExit(
        f"Expected {expected_trials} unique trial identities, got {len(identities)}: {jsonl}"
    )
task_counts = Counter(int(row["task_id"]) for row in rows)
if len(task_counts) != expected_task_count:
    raise SystemExit(
        f"Expected {expected_task_count} tasks, got {dict(sorted(task_counts.items()))}: {jsonl}"
    )
expected_per_task = expected_trials // expected_task_count
if set(task_counts.values()) != {expected_per_task}:
    raise SystemExit(
        f"Expected {expected_per_task} trials per task, got "
        f"{dict(sorted(task_counts.items()))}: {jsonl}"
    )
requested_ages = {
    int(row.get("requested_semantic_age", row["semantic_age_frames"])) for row in rows
}
if requested_ages != {expected_age}:
    raise SystemExit(
        f"Expected requested semantic age {expected_age}, got {sorted(requested_ages)}: {jsonl}"
    )
horizons = {int(row["action_execution_horizon"]) for row in rows}
if horizons != {expected_horizon}:
    raise SystemExit(
        f"Expected execution horizon {expected_horizon}, got {sorted(horizons)}: {jsonl}"
    )
nonbootstrap_mismatches = sum(
    max(
        int(row.get("semantic_requested_actual_age_mismatch_boundary_count", row.get("semantic_age_mismatch_count", 0)))
        - int(row.get("semantic_bootstrap_clipped_boundary_count", row.get("semantic_age_bootstrap_mismatch_count", 0))),
        0,
    )
    for row in rows
)
if nonbootstrap_mismatches:
    raise SystemExit(
        f"Expected zero non-bootstrap age mismatches, got {nonbootstrap_mismatches}: {jsonl}"
    )
successes = sum(bool(row["success"]) for row in rows)
with pathlib.Path(result_path).open("a") as stream:
    stream.write(
        f"{arm}\t{step}\t{successes}\t{len(rows)}\t{successes / len(rows):.9f}"
        f"\t{checkpoint}\t{checkpoint_sha256}\t{jsonl}\n"
    )
PY
}

candidate_recorded() {
    local arm=$1
    local step=$2
    awk -F '\t' -v arm="${arm}" -v step="${step}" \
        'NR > 1 && $1 == arm && $2 == step {found = 1} END {exit !found}' "${RESULTS_TSV}"
}

read -r -a selection_arms <<<"${SELECTION_ARMS}"
for arm in "${selection_arms[@]}"; do
    if [[ "${arm}" != C && "${arm}" != D ]]; then
        echo "Unknown selection arm: ${arm}" >&2
        exit 2
    fi
    arm_steps_name="FDVLA_MATCHED_${arm}_SELECTION_STEPS"
    arm_steps="${!arm_steps_name:-${SELECTION_STEPS}}"
    if [[ -n "${arm_steps}" ]]; then
        read -r -a candidate_steps <<<"${arm_steps}"
    else
        mapfile -t candidate_steps < <(
            seq "${MIN_STEP}" "${SAVE_INTERVAL}" "${MAX_STEPS}"
        )
    fi
    if ((${#candidate_steps[@]} == 0)); then
        echo "No selection candidate steps were provided for arm ${arm}." >&2
        exit 2
    fi
    for step in "${candidate_steps[@]}"; do
        if [[ ! "${step}" =~ ^[1-9][0-9]*$ ]]; then
            echo "Invalid selection candidate step for ${arm}: ${step}" >&2
            exit 2
        fi
    done
    for step in "${candidate_steps[@]}"; do
        checkpoint=$(checkpoint_for "${arm}" "${step}")
        test -f "${checkpoint}/model.safetensors.index.json"
        eval_dir="${SELECTION_ROOT}/${arm}/step_${step}"
        jsonl="${eval_dir}/eval_trials.jsonl"
        mkdir -p "${eval_dir}"
        if candidate_recorded "${arm}" "${step}"; then
            assert_candidate_provenance "${arm}" "${step}" "${eval_dir}"
            continue
        fi
        if [[ ! -s "${jsonl}" ]]; then
            completed=false
            for attempt in $(seq 1 "${EVAL_MAX_ATTEMPTS}"); do
                if [[ "${arm}" == C ]]; then
                    if run_c_candidate "${step}" "${checkpoint}" "${eval_dir}"; then
                        completed=true
                        break
                    fi
                elif run_d_candidate "${step}" "${checkpoint}" "${eval_dir}"; then
                    completed=true
                    break
                fi
                cp "${eval_dir}/launcher.log" \
                    "${eval_dir}/launcher_attempt_${attempt}_failed.log"
                sleep 5
            done
            if [[ "${completed}" != true ]]; then
                echo "Candidate failed after ${EVAL_MAX_ATTEMPTS} attempts: ${arm} step ${step}" >&2
                exit 1
            fi
        fi
        assert_selection_worktree_frozen "before_record_${arm}_${step}"
        assert_candidate_provenance "${arm}" "${step}" "${eval_dir}"
        append_result "${arm}" "${step}" "${checkpoint}" "${jsonl}"
    done
done

python "${RLINF_REPO}/examples/analysis/select_fdvla_paired_starts.py" \
    --c-tsv "${RESULTS_TSV}" \
    --d-tsv "${RESULTS_TSV}" \
    --output "${SELECTED_ENV}" \
    --target-successes "${TARGET_SUCCESSES}" \
    --max-deviation "${MAX_START_DEVIATION}" \
    --max-gap "${MAX_START_GAP}" \
    --expected-trials "${SELECTION_NUM_TRIALS}" \
    --expected-age 0 \
    --expected-horizon "${SELECTION_EXECUTION_HORIZON}"

assert_selection_worktree_frozen complete
cat "${RESULTS_TSV}"
cat "${SELECTED_ENV}"
