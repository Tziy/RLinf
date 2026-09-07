#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
AGES=(0 1 2 4 6 8 12)
DRY_RUN=${DRY_RUN:-true}
POLICY_METHOD=${POLICY_METHOD:-D-SFT}

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the common policy checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the frozen backbone}"
: "${FDVLA_INFORMATION_OUTPUT_ROOT:?Set an absolute output directory outside Git}"

if [[ "${FDVLA_INFORMATION_OUTPUT_ROOT}" != /* ]]; then
    echo "FDVLA_INFORMATION_OUTPUT_ROOT must be absolute." >&2
    exit 2
fi
case "${FDVLA_INFORMATION_OUTPUT_ROOT}" in
    "${REPO_DIR}"|"${REPO_DIR}"/*)
        echo "Information NPZ output must be outside the Git source tree." >&2
        exit 2
        ;;
esac
if [[ "${POLICY_METHOD}" == D-PPO && -z "${PPO_CKPT_PATH:-}" ]]; then
    echo "POLICY_METHOD=D-PPO requires PPO_CKPT_PATH." >&2
    exit 2
fi
if [[ "${POLICY_METHOD}" != D-SFT && "${POLICY_METHOD}" != D-PPO ]]; then
    echo "POLICY_METHOD must be D-SFT or D-PPO." >&2
    exit 2
fi

for age in "${AGES[@]}"; do
    raw_dir="${FDVLA_INFORMATION_OUTPUT_ROOT}/raw/${POLICY_METHOD}/age_${age}"
    command=(
        env
        ONLY_EVAL=true
        FDVLA_INFORMATION_EXPORT=true
        FDVLA_INFORMATION_RAW_DIR="${raw_dir}"
        SEMANTIC_EVAL_FIXED_AGE_FRAMES="${age}"
        SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
        SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
        PPO_EXPERIMENT_NAME="fdvla_information_${POLICY_METHOD}_age_${age}"
        FDVLA_METHOD_ID="${POLICY_METHOD}"
        bash "${REPO_DIR}/examples/embodiment/run_fdvla_binary_ppo.sh"
    )
    printf 'age=%s command=' "${age}"
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" != true ]]; then
        "${command[@]}"
    fi
done

dataset_path="${FDVLA_INFORMATION_OUTPUT_ROOT}/fdvla_information_${POLICY_METHOD}.npz"
teacher_checkpoint=${TEACHER_CHECKPOINT:-${GR00T_MODEL_PATH}}
label_command=(
    python "${REPO_DIR}/examples/analysis/collect_fdvla_information_dataset.py"
    --raw-input
)
if [[ "${DRY_RUN}" == true ]]; then
    label_command+=("${FDVLA_INFORMATION_OUTPUT_ROOT}/raw/${POLICY_METHOD}/age_*/*.npz")
else
    mapfile -d '' raw_inputs < <(
        find "${FDVLA_INFORMATION_OUTPUT_ROOT}/raw/${POLICY_METHOD}" -type f -name '*.npz' -print0 | sort -z
    )
    if (( ${#raw_inputs[@]} == 0 )); then
        echo "No raw information episodes were exported." >&2
        exit 2
    fi
    label_command+=("${raw_inputs[@]}")
fi
label_command+=(
    --teacher-checkpoint "${teacher_checkpoint}"
    --teacher-backbone-model-path "${GR00T_BACKBONE_PATH}"
    --teacher-device "${TEACHER_DEVICE:-cuda:0}"
    --teacher-batch-size "${TEACHER_BATCH_SIZE:-8}"
    --output "${dataset_path}"
)
printf 'teacher-label command='
printf '%q ' "${label_command[@]}"
printf '\n'
if [[ "${DRY_RUN}" != true ]]; then
    env PYTHONPATH="$(dirname "${REPO_DIR}")":"${REPO_DIR}":"${PYTHONPATH:-}" \
        "${label_command[@]}"
fi
