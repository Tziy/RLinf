#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/fdvla_binary_common.sh"

read -r -a AGES <<<"${SEMANTIC_AGES:-0 1 2 4 6 8 12 16}"
read -r -a ACTION_HORIZONS <<<"${ACTION_HORIZONS:-1 2 4 8 16}"
ACTION_PREDICTION_HORIZON=${ACTION_CHUNK_SIZE:-16}
SURFACE_CONTROL_HZ=${FDVLA_CONTROL_HZ:-20}
DRY_RUN=${DRY_RUN:-true}
RESUME_SURFACE=${RESUME_SURFACE:-false}
SURFACE_SWEEP_MODE=${SURFACE_SWEEP_MODE:-cartesian}
POLICY_METHOD=${POLICY_METHOD:-D-SFT}
SURFACE_MIN_TRIALS=${SURFACE_MIN_TRIALS:-400}
SURFACE_EVAL_NUM_ENVS=${SURFACE_EVAL_NUM_ENVS:-52}
SURFACE_TASK_ID_FILTER=${SURFACE_TASK_ID_FILTER:-${EVAL_TASK_ID_FILTER:-'[0]'}}
read -r -a SURFACE_NOISE_SEEDS <<<"${SURFACE_POLICY_NOISE_SEEDS:-2026 2027 2028 2029 2030 2031 2032 2033}"
if [[ ! "${SURFACE_MIN_TRIALS}" =~ ^[1-9][0-9]*$ \
    || ! "${SURFACE_EVAL_NUM_ENVS}" =~ ^[1-9][0-9]*$ \
    || ! "${ACTION_PREDICTION_HORIZON}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SURFACE_MIN_TRIALS and SURFACE_EVAL_NUM_ENVS must be positive integers." >&2
    exit 2
fi
if ((ACTION_PREDICTION_HORIZON != 16)); then
    echo "Formal surface requires the pre-registered 16-step action prediction horizon." >&2
    exit 2
fi
if [[ "${SURFACE_CONTROL_HZ}" != 20 && "${SURFACE_CONTROL_HZ}" != 20.0 ]]; then
    echo "Formal surface requires the pre-registered 20 Hz control frequency." >&2
    exit 2
fi
if ((${#AGES[@]} == 0 || ${#ACTION_HORIZONS[@]} == 0)); then
    echo "SEMANTIC_AGES and ACTION_HORIZONS must each contain at least one value." >&2
    exit 2
fi
case "${SURFACE_SWEEP_MODE}" in
    cartesian | deadline_axes)
        ;;
    *)
        echo "SURFACE_SWEEP_MODE must be cartesian or deadline_axes." >&2
        exit 2
        ;;
esac
if ((${#SURFACE_NOISE_SEEDS[@]} == 0)); then
    echo "SURFACE_POLICY_NOISE_SEEDS must contain at least one integer seed." >&2
    exit 2
fi
declare -A SEEN_NOISE_SEEDS=()
for noise_seed in "${SURFACE_NOISE_SEEDS[@]}"; do
    if [[ ! "${noise_seed}" =~ ^[0-9]+$ ]]; then
        echo "Invalid non-negative integer policy noise seed: ${noise_seed}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_NOISE_SEEDS[${noise_seed}]:-}" ]]; then
        echo "Duplicate policy noise seed: ${noise_seed}" >&2
        exit 2
    fi
    SEEN_NOISE_SEEDS[${noise_seed}]=1
done
SURFACE_EVAL_ROLLOUT_EPOCH=${EVAL_ROLLOUT_EPOCH:-1}
if [[ ! "${SURFACE_EVAL_ROLLOUT_EPOCH}" =~ ^[0-9]+$ ]] || ((SURFACE_EVAL_ROLLOUT_EPOCH != 1)); then
    echo "Formal surface requires EVAL_ROLLOUT_EPOCH=1; use distinct policy-noise seeds for independent trials." >&2
    exit 2
fi

declare -A SEEN_AGES=()
for semantic_age in "${AGES[@]}"; do
    if [[ ! "${semantic_age}" =~ ^[0-9]+$ ]]; then
        echo "Invalid non-negative integer semantic age: ${semantic_age}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_AGES[${semantic_age}]:-}" ]]; then
        echo "Duplicate semantic age: ${semantic_age}" >&2
        exit 2
    fi
    SEEN_AGES[${semantic_age}]=1
done

declare -A SEEN_ACTION_HORIZONS=()
for action_horizon in "${ACTION_HORIZONS[@]}"; do
    if [[ ! "${action_horizon}" =~ ^[0-9]+$ ]] || ((action_horizon < 1)); then
        echo "Invalid positive integer action horizon: ${action_horizon}" >&2
        exit 2
    fi
    if ((action_horizon > ACTION_PREDICTION_HORIZON)); then
        echo "Action horizon ${action_horizon} exceeds prediction horizon ${ACTION_PREDICTION_HORIZON}." >&2
        exit 2
    fi
    if [[ -n "${SEEN_ACTION_HORIZONS[${action_horizon}]:-}" ]]; then
        echo "Duplicate action horizon: ${action_horizon}" >&2
        exit 2
    fi
    SEEN_ACTION_HORIZONS[${action_horizon}]=1
done

if [[ "${SURFACE_SWEEP_MODE}" == deadline_axes ]]; then
    if [[ -z "${SEEN_AGES[0]:-}" || -z "${SEEN_ACTION_HORIZONS[1]:-}" ]]; then
        echo "deadline_axes requires semantic age 0 and action horizon 1." >&2
        exit 2
    fi
fi

case "${POLICY_METHOD}" in
    C-PPO | D-PPO)
        : "${PPO_CKPT_PATH:?POLICY_METHOD=${POLICY_METHOD} requires PPO_CKPT_PATH}"
        ;;
    D-SFT)
        unset PPO_CKPT_PATH
        ;;
    *)
        echo "POLICY_METHOD must be C-PPO, D-SFT, or D-PPO." >&2
        exit 2
        ;;
esac
EVAL_LAUNCHER=${SCRIPT_DIR}/run_fdvla_binary_ppo.sh
[[ "${POLICY_METHOD}" == C-PPO ]] && EVAL_LAUNCHER=${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh

export FDVLA_METHOD_ID=${POLICY_METHOD}

REPO_PATH=${FDVLA_REPO_PATH}
SURFACE_ROOT=${FDVLA_SURFACE_ROOT:-"${REPO_PATH}/logs/fdvla_realtime_surface/$(date -u +%Y%m%dT%H%M%SZ)_${POLICY_METHOD}"}
if [[ "${SURFACE_ROOT}" != /* ]]; then
    echo "FDVLA_SURFACE_ROOT must be an absolute path: ${SURFACE_ROOT}" >&2
    exit 2
fi
MANIFEST_PATH="${SURFACE_ROOT}/surface_cells.tsv"
MANIFEST_SCHEMA_VERSION=3
MANIFEST_HEADER=$'schema_version\tartifact_hash_algorithm\tmethod\tsemantic_age\taction_horizon\ttask_id_filter\tpolicy_noise_seeds\tpolicy_checkpoint_path\tpolicy_checkpoint_sha256\tinitial_checkpoint_path\tinitial_checkpoint_sha256\tbackbone_path\tbackbone_sha256\tgit_sha\tworktree_sha256\taction_prediction_horizon\tcontrol_hz\tjsonl'

if [[ "${DRY_RUN}" != true ]]; then
    fdvla_require_inputs
    if [[ "${POLICY_METHOD}" == *-PPO && "${PPO_CKPT_PATH}" != /* ]]; then
        echo "PPO_CKPT_PATH must be absolute for a formal PPO surface." >&2
        exit 2
    fi
    SURFACE_INITIAL_CHECKPOINT_PATH=${GR00T_MODEL_PATH}
    SURFACE_INITIAL_CHECKPOINT_SHA256=$(
        fdvla_checkpoint_sha256 "${SURFACE_INITIAL_CHECKPOINT_PATH}"
    )
    SURFACE_BACKBONE_PATH=${GR00T_BACKBONE_PATH}
    SURFACE_BACKBONE_SHA256=$(fdvla_checkpoint_sha256 "${SURFACE_BACKBONE_PATH}")
    SURFACE_POLICY_CHECKPOINT_PATH=${PPO_CKPT_PATH:-${GR00T_MODEL_PATH}}
    if [[ "${SURFACE_POLICY_CHECKPOINT_PATH}" == "${SURFACE_INITIAL_CHECKPOINT_PATH}" ]]; then
        SURFACE_POLICY_CHECKPOINT_SHA256=${SURFACE_INITIAL_CHECKPOINT_SHA256}
    else
        SURFACE_POLICY_CHECKPOINT_SHA256=$(
            fdvla_checkpoint_sha256 "${SURFACE_POLICY_CHECKPOINT_PATH}"
        )
    fi
    SURFACE_GIT_SHA=$(git -C "${REPO_PATH}" rev-parse HEAD)
    SURFACE_WORKTREE_SHA256=$(fdvla_worktree_sha256)

    if [[ -e "${MANIFEST_PATH}" ]]; then
        if [[ "${RESUME_SURFACE}" != true ]]; then
            echo "Refusing to overwrite existing surface manifest: ${MANIFEST_PATH}" >&2
            echo "Set RESUME_SURFACE=true to validate and skip completed cells." >&2
            exit 2
        fi
        if [[ "$(head -n 1 "${MANIFEST_PATH}")" != "${MANIFEST_HEADER}" ]]; then
            echo "Invalid existing surface manifest header: ${MANIFEST_PATH}" >&2
            exit 2
        fi
    else
        mkdir -p "${SURFACE_ROOT}"
        printf '%s\n' "${MANIFEST_HEADER}" > "${MANIFEST_PATH}"
    fi
fi

validate_cell_jsonl() {
    local method=$1
    local semantic_age=$2
    local action_horizon=$3
    local jsonl_path=$4
    local validation_dir=$5
    PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}" \
        python "${REPO_PATH}/examples/analysis/summarize_fdvla_realtime_surface.py" \
        --cell "${method}:${semantic_age}:${action_horizon}=${jsonl_path}" \
        --min-trials-per-cell "${SURFACE_MIN_TRIALS}" \
        --allow-incomplete \
        --output-dir "${validation_dir}" \
        --skip-plots \
        > "${validation_dir}.log"
}

run_cell() {
    local semantic_age=$1
    local action_horizon=$2
    local label="age_${semantic_age}_ka_${action_horizon}"
    local cell_dir="${SURFACE_ROOT}/${label}"
    printf '%s method=%s semantic_age_frames=%s action_execution_horizon=%s checkpoint=%s\n' \
        "${label}" \
        "${POLICY_METHOD}" \
        "${semantic_age}" \
        "${action_horizon}" \
        "${PPO_CKPT_PATH:-${GR00T_MODEL_PATH:-unset}}"
    if [[ "${DRY_RUN}" == true ]]; then
        printf '  task_filter=%s policy_noise_seeds=%s eval_envs=%s eval_rollout_epoch=%s min_trials=%s\n' \
            "${SURFACE_TASK_ID_FILTER}" \
            "${SURFACE_NOISE_SEEDS[*]}" \
            "${SURFACE_EVAL_NUM_ENVS}" \
            "${SURFACE_EVAL_ROLLOUT_EPOCH}" \
            "${SURFACE_MIN_TRIALS}"
        return
    fi

    local jsonl_path="${cell_dir}/eval_trials_combined.jsonl"
    local validation_dir="${cell_dir}/validation"
    if [[ "${RESUME_SURFACE}" == true ]]; then
        local existing_rows=()
        mapfile -t existing_rows < <(
            awk -F '\t' -v method="${POLICY_METHOD}" -v age="${semantic_age}" -v horizon="${action_horizon}" \
                'NR > 1 && $3 == method && $4 == age && $5 == horizon {print $0}' \
                "${MANIFEST_PATH}"
        )
        if ((${#existing_rows[@]} > 1)); then
            echo "Duplicate manifest rows for ${POLICY_METHOD} age=${semantic_age} horizon=${action_horizon}." >&2
            exit 2
        fi
        if ((${#existing_rows[@]} == 1)); then
            local recorded_schema recorded_hash_algorithm recorded_method
            local recorded_age recorded_horizon
            local recorded_task_filter recorded_noise_seeds recorded_policy_path
            local recorded_policy_sha recorded_initial_path recorded_initial_sha
            local recorded_backbone_path recorded_backbone_sha recorded_git_sha
            local recorded_worktree_sha recorded_prediction_horizon recorded_control_hz
            local recorded_jsonl
            IFS=$'\t' read -r recorded_schema recorded_hash_algorithm \
                recorded_method recorded_age \
                recorded_horizon recorded_task_filter recorded_noise_seeds \
                recorded_policy_path recorded_policy_sha recorded_initial_path \
                recorded_initial_sha recorded_backbone_path recorded_backbone_sha \
                recorded_git_sha recorded_worktree_sha recorded_prediction_horizon \
                recorded_control_hz recorded_jsonl \
                <<< "${existing_rows[0]}"
            if [[ "${recorded_schema}" != "${MANIFEST_SCHEMA_VERSION}" \
                || "${recorded_hash_algorithm}" != "${FDVLA_ARTIFACT_HASH_ALGORITHM}" \
                || "${recorded_task_filter}" != "${SURFACE_TASK_ID_FILTER}" \
                || "${recorded_noise_seeds}" != "${SURFACE_NOISE_SEEDS[*]}" \
                || "${recorded_policy_path}" != "${SURFACE_POLICY_CHECKPOINT_PATH}" \
                || "${recorded_policy_sha}" != "${SURFACE_POLICY_CHECKPOINT_SHA256}" \
                || "${recorded_initial_path}" != "${SURFACE_INITIAL_CHECKPOINT_PATH}" \
                || "${recorded_initial_sha}" != "${SURFACE_INITIAL_CHECKPOINT_SHA256}" \
                || "${recorded_backbone_path}" != "${SURFACE_BACKBONE_PATH}" \
                || "${recorded_backbone_sha}" != "${SURFACE_BACKBONE_SHA256}" \
                || "${recorded_git_sha}" != "${SURFACE_GIT_SHA}" \
                || "${recorded_worktree_sha}" != "${SURFACE_WORKTREE_SHA256}" \
                || "${recorded_prediction_horizon}" != "${ACTION_PREDICTION_HORIZON}" \
                || "${recorded_control_hz}" != "${SURFACE_CONTROL_HZ}" \
                || "${recorded_jsonl}" != "${jsonl_path}" ]]; then
                echo "Resume signature mismatch for ${label}: ${existing_rows[0]}" >&2
                exit 2
            fi
            if [[ ! -s "${jsonl_path}" ]]; then
                echo "Resume result is missing or empty: ${jsonl_path}" >&2
                exit 2
            fi
            validate_cell_jsonl \
                "${POLICY_METHOD}" "${semantic_age}" "${action_horizon}" \
                "${jsonl_path}" "${validation_dir}"
            echo "Resume validated completed cell: ${label}"
            return
        fi
    fi

    mkdir -p "${cell_dir}"
    : > "${jsonl_path}"
    local noise_seed
    for noise_seed in "${SURFACE_NOISE_SEEDS[@]}"; do
        local seed_dir="${cell_dir}/noise_${noise_seed}"
        ONLY_EVAL=true \
            SEMANTIC_EVAL_FIXED_AGE_FRAMES="${semantic_age}" \
            SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1 \
            SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1 \
            EVAL_EXECUTION_HORIZON="${action_horizon}" \
            EVAL_NOISE_SEED="${noise_seed}" \
            EVAL_TASK_ID_FILTER="${SURFACE_TASK_ID_FILTER}" \
            TARGET_EVAL_ENVS="${SURFACE_EVAL_NUM_ENVS}" \
            EVAL_NUM_ENVS="${SURFACE_EVAL_NUM_ENVS}" \
            EVAL_ROLLOUT_EPOCH="${SURFACE_EVAL_ROLLOUT_EPOCH}" \
            FDVLA_RUN_LOG_DIR="${seed_dir}" \
            RLINF_LOG_DIR="${seed_dir}" \
            FDVLA_METADATA_DIR="${seed_dir}/fdvla_metadata_${POLICY_METHOD}" \
            FDVLA_RESOLVED_MODEL_SHA256="${SURFACE_INITIAL_CHECKPOINT_SHA256}" \
            FDVLA_RESOLVED_BACKBONE_SHA256="${SURFACE_BACKBONE_SHA256}" \
            FDVLA_RESOLVED_PPO_SHA256="${SURFACE_POLICY_CHECKPOINT_SHA256}" \
            PPO_EXPERIMENT_NAME="fdvla_surface_${POLICY_METHOD}_${label}_noise_${noise_seed}" \
            bash "${EVAL_LAUNCHER}"

        local seed_jsonl="${seed_dir}/eval_trials.jsonl"
        if [[ ! -s "${seed_jsonl}" ]]; then
            echo "Missing or empty surface result: ${seed_jsonl}" >&2
            exit 1
        fi
        cat "${seed_jsonl}" >> "${jsonl_path}"
    done

    local trial_count
    trial_count=$(wc -l < "${jsonl_path}")
    if ((trial_count < SURFACE_MIN_TRIALS)); then
        echo "Surface cell ${label} has ${trial_count} trials; required ${SURFACE_MIN_TRIALS}." >&2
        exit 1
    fi
    validate_cell_jsonl \
        "${POLICY_METHOD}" "${semantic_age}" "${action_horizon}" \
        "${jsonl_path}" "${validation_dir}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${MANIFEST_SCHEMA_VERSION}" \
        "${FDVLA_ARTIFACT_HASH_ALGORITHM}" \
        "${POLICY_METHOD}" \
        "${semantic_age}" \
        "${action_horizon}" \
        "${SURFACE_TASK_ID_FILTER}" \
        "${SURFACE_NOISE_SEEDS[*]}" \
        "${SURFACE_POLICY_CHECKPOINT_PATH}" \
        "${SURFACE_POLICY_CHECKPOINT_SHA256}" \
        "${SURFACE_INITIAL_CHECKPOINT_PATH}" \
        "${SURFACE_INITIAL_CHECKPOINT_SHA256}" \
        "${SURFACE_BACKBONE_PATH}" \
        "${SURFACE_BACKBONE_SHA256}" \
        "${SURFACE_GIT_SHA}" \
        "${SURFACE_WORKTREE_SHA256}" \
        "${ACTION_PREDICTION_HORIZON}" \
        "${SURFACE_CONTROL_HZ}" \
        "${jsonl_path}" \
        >> "${MANIFEST_PATH}"
}

echo "surface_sweep_mode=${SURFACE_SWEEP_MODE} semantic_ages=${AGES[*]} action_horizons=${ACTION_HORIZONS[*]}"
case "${SURFACE_SWEEP_MODE}" in
    cartesian)
        for semantic_age in "${AGES[@]}"; do
            for action_horizon in "${ACTION_HORIZONS[@]}"; do
                run_cell "${semantic_age}" "${action_horizon}"
            done
        done
        ;;
    deadline_axes)
        for semantic_age in "${AGES[@]}"; do
            run_cell "${semantic_age}" 1
        done
        for action_horizon in "${ACTION_HORIZONS[@]}"; do
            if ((action_horizon != 1)); then
                run_cell 0 "${action_horizon}"
            fi
        done
        ;;
esac

if [[ "${DRY_RUN}" != true ]]; then
    echo "Surface manifest: ${MANIFEST_PATH}"
    echo "Summarize with: python examples/analysis/summarize_fdvla_realtime_surface.py --manifest ${MANIFEST_PATH} --output-dir <OUTPUT_DIR>"
fi
