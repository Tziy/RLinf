#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the preregistered weak1000 checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the frozen backbone}"

BASELINE_ID=${FDVLA_BASELINE_SCALE_ID:-20260822_scale60_after_resume_v3}
SINGLE_ID=${FDVLA_SINGLE_EXACT_SCALE_ID:-20260822_scale60_single_exact_v1}
BATCH2_ID=${FDVLA_BATCH2_SCALE_ID:-20260822_scale60_single_exact_batch2_v1}
TUNING_ID=${FDVLA_SYSTEM_TUNING_ID:-20260822_weak1000_system_tuning_v1}
TUNING_ROOT=${REPO_DIR}/logs/fdvla_d_system_tuning/${TUNING_ID}
COUPLED_SUMMARY=${REPO_DIR}/logs/fdvla_task0_long_compare/20260822_weak1000_task0_long_token160_cfirst_v4/C-PPO/summary/fdvla_training_health.json
REGISTERED_PARENT=${REPO_DIR}/logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/fdvla_binary_d_ppo_weak1000_task0_seed0/checkpoints/global_step_50
mkdir -p "${TUNING_ROOT}"

export PPO_MAX_STEPS=5
export TRAIN_NUM_ENVS=60
export TARGET_TRAIN_ENVS=60
export EVAL_NUM_ENVS=48
export TARGET_EVAL_ENVS=48
export TRAIN_MAX_EPISODE_STEPS=480
export TRAIN_ROLLOUT_STEPS=256
export TRAIN_ROLLOUT_EPOCH=4
export PPO_GLOBAL_BATCH_SIZE=384
export SEMANTIC_SKIP_LATEST_BEFORE_EXACT=true
export SEMANTIC_BATCH_MAX_REQUESTS=1
export SEMANTIC_BATCH_TARGET_REQUESTS=0
export SEMANTIC_BATCH_TARGET_ENVS=1
export SEMANTIC_BATCH_WAIT_MS=0
unset PPO_RESUME_DIR

export FDVLA_PLACEMENT_SMOKE_ID=${SINGLE_ID}
export RAY_TMPDIR=/tmp/fdvla_d_single_a4r3_v1
bash "${SCRIPT_DIR}/run_fdvla_d_placement_smoke.sh" actor4_rollout3
export RAY_TMPDIR=/tmp/fdvla_d_single_col4_v1
bash "${SCRIPT_DIR}/run_fdvla_d_placement_smoke.sh" colocated4

BASELINE_ROOT=${REPO_DIR}/logs/fdvla_d_placement_smoke/${BASELINE_ID}
SINGLE_ROOT=${REPO_DIR}/logs/fdvla_d_placement_smoke/${SINGLE_ID}
PRIMARY_REPORT=${TUNING_ROOT}/primary_selection.json
python "${REPO_DIR}/examples/analysis/select_fdvla_d_placement.py" \
    --candidate-set system_tuning \
    --candidate actor4_rollout3_double_fetch="${BASELINE_ROOT}/actor4_rollout3" \
    --candidate colocated4_double_fetch="${BASELINE_ROOT}/colocated4" \
    --candidate actor4_rollout3_single_exact_fetch="${SINGLE_ROOT}/actor4_rollout3" \
    --candidate colocated4_single_exact_fetch="${SINGLE_ROOT}/colocated4" \
    --coupled-summary "${COUPLED_SUMMARY}" \
    --output "${PRIMARY_REPORT}"

speed_passed=$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["selected_vs_coupled"]["speed_endpoint_passed"]).lower())' "${PRIMARY_REPORT}")
FINAL_REPORT=${PRIMARY_REPORT}
if [[ "${speed_passed}" != true ]]; then
    export FDVLA_PLACEMENT_SMOKE_ID=${BATCH2_ID}
    export SEMANTIC_BATCH_MAX_REQUESTS=2
    export SEMANTIC_BATCH_TARGET_REQUESTS=2
    export SEMANTIC_BATCH_TARGET_ENVS=0
    export SEMANTIC_BATCH_WAIT_MS=5
    export RAY_TMPDIR=/tmp/fdvla_d_batch2_a4r3_v1
    bash "${SCRIPT_DIR}/run_fdvla_d_placement_smoke.sh" actor4_rollout3
    export RAY_TMPDIR=/tmp/fdvla_d_batch2_col4_v1
    bash "${SCRIPT_DIR}/run_fdvla_d_placement_smoke.sh" colocated4

    BATCH2_ROOT=${REPO_DIR}/logs/fdvla_d_placement_smoke/${BATCH2_ID}
    FINAL_REPORT=${TUNING_ROOT}/fallback_selection.json
    python "${REPO_DIR}/examples/analysis/select_fdvla_d_placement.py" \
        --candidate-set system_tuning_with_fallback \
        --candidate actor4_rollout3_double_fetch="${BASELINE_ROOT}/actor4_rollout3" \
        --candidate colocated4_double_fetch="${BASELINE_ROOT}/colocated4" \
        --candidate actor4_rollout3_single_exact_fetch="${SINGLE_ROOT}/actor4_rollout3" \
        --candidate colocated4_single_exact_fetch="${SINGLE_ROOT}/colocated4" \
        --candidate actor4_rollout3_single_exact_fetch_batch2="${BATCH2_ROOT}/actor4_rollout3" \
        --candidate colocated4_single_exact_fetch_batch2="${BATCH2_ROOT}/colocated4" \
        --coupled-summary "${COUPLED_SUMMARY}" \
        --output "${FINAL_REPORT}"
fi

selected=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_placement"])' "${FINAL_REPORT}")
case "${selected}" in
    actor4_rollout3_*) placement=actor4_rollout3 ;;
    colocated4_*) placement=colocated4 ;;
    *) echo "Unknown selected system candidate: ${selected}" >&2; exit 2 ;;
esac
case "${selected}" in
    *_single_exact_fetch*) export SEMANTIC_SKIP_LATEST_BEFORE_EXACT=true ;;
    *) export SEMANTIC_SKIP_LATEST_BEFORE_EXACT=false ;;
esac
case "${selected}" in
    *_batch2)
        export SEMANTIC_BATCH_MAX_REQUESTS=2
        export SEMANTIC_BATCH_TARGET_REQUESTS=2
        export SEMANTIC_BATCH_TARGET_ENVS=0
        export SEMANTIC_BATCH_WAIT_MS=5
        ;;
    *)
        export SEMANTIC_BATCH_MAX_REQUESTS=1
        export SEMANTIC_BATCH_TARGET_REQUESTS=0
        export SEMANTIC_BATCH_TARGET_ENVS=1
        export SEMANTIC_BATCH_WAIT_MS=0
        ;;
esac
printf 'selected_candidate=%s\nplacement=%s\nsingle_exact_fetch=%s\nbatch_max_requests=%s\nselection_report=%s\n' \
    "${selected}" "${placement}" "${SEMANTIC_SKIP_LATEST_BEFORE_EXACT}" \
    "${SEMANTIC_BATCH_MAX_REQUESTS}" "${FINAL_REPORT}" \
    >"${TUNING_ROOT}/selected_system.env"

export PPO_RESUME_DIR=${REGISTERED_PARENT}
export FDVLA_TARGET80_RUN_ID=${TUNING_ID}_target80
export FDVLA_TARGET80_RUN_ROOT=${REPO_DIR}/logs/fdvla_target80_resume/${FDVLA_TARGET80_RUN_ID}
export RAY_TMPDIR=/tmp/fdvla_target80_${placement}_v1
bash "${SCRIPT_DIR}/run_fdvla_target80_resume.sh" "${placement}"

export FDVLA_TARGET80_TRAIN_RUN=${FDVLA_TARGET80_RUN_ROOT}/${placement}
export FDVLA_HELDOUT_ID=${TUNING_ID}_heldout
export FDVLA_HELDOUT_ROOT=${REPO_DIR}/logs/fdvla_target80_heldout/${FDVLA_HELDOUT_ID}_${placement}
export RAY_TMPDIR=/tmp/fdvla_target80_heldout_${placement}_v1
bash "${SCRIPT_DIR}/eval_fdvla_target80_heldout.sh" "${placement}"

printf '%s\n' "${TUNING_ROOT}"
