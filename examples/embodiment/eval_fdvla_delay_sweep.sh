#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AGES=(0 2 4 6 8 12)
DRY_RUN=${DRY_RUN:-true}
POLICY_METHOD=${POLICY_METHOD:-D-PPO}
SWEEP_ID=${FDVLA_DELAY_SWEEP_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
FDVLA_DELAY_SWEEP_ROOT=${FDVLA_DELAY_SWEEP_ROOT:-${SCRIPT_DIR}/../../logs/fdvla_delay_sweep/${SWEEP_ID}}

if [[ "${FDVLA_DELAY_SWEEP_ROOT}" != /* ]]; then
    echo "FDVLA_DELAY_SWEEP_ROOT must be absolute." >&2
    exit 2
fi
mkdir -p "${FDVLA_DELAY_SWEEP_ROOT}"

case "${POLICY_METHOD}" in
    D-PPO)
        : "${PPO_CKPT_PATH:?POLICY_METHOD=D-PPO requires the selected PPO_CKPT_PATH}"
        ;;
    D-SFT)
        unset PPO_CKPT_PATH
        ;;
    *)
        echo "POLICY_METHOD must be D-SFT or D-PPO." >&2
        exit 2
        ;;
esac
export FDVLA_METHOD_ID=${POLICY_METHOD}

run_eval() {
    local label=$1
    local fixed_age=$2
    local random_min=$3
    local random_max=$4
    local run_dir="${FDVLA_DELAY_SWEEP_ROOT}/${POLICY_METHOD}/${label}"
    printf '%s method=%s checkpoint=%s run_dir=%s\n' \
        "${label}" "${POLICY_METHOD}" "${PPO_CKPT_PATH:-${GR00T_MODEL_PATH:-unset}}" "${run_dir}"
    if [[ "${DRY_RUN}" == true ]]; then
        return
    fi
    ONLY_EVAL=true \
        FDVLA_RUN_LOG_DIR="${run_dir}" \
        RLINF_LOG_DIR="${run_dir}" \
        FDVLA_METADATA_DIR="${run_dir}/fdvla_metadata" \
        FDVLA_RUN_TIMESTAMP="${SWEEP_ID}_${POLICY_METHOD}_${label}" \
        SEMANTIC_EVAL_FIXED_AGE_FRAMES="${fixed_age}" \
        SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES="${random_min}" \
        SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES="${random_max}" \
        PPO_EXPERIMENT_NAME="fdvla_delay_${POLICY_METHOD}_${label}" \
        bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh"
}

for age in "${AGES[@]}"; do
    run_eval "age_${age}" "${age}" -1 -1
done
run_eval random_age_0_6 -1 0 6
run_eval natural_async -1 -1 -1
