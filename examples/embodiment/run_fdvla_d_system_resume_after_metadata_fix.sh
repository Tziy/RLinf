#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the preregistered weak1000 checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the frozen backbone}"

SMOKE_ID=${FDVLA_BATCH2_FIXED_ID:-20260822_scale60_single_exact_batch2_fixed_v2}
TUNING_ID=${FDVLA_SYSTEM_TUNING_FIXED_ID:-20260822_weak1000_system_tuning_fixed_v2}
SMOKE_ROOT=${REPO_DIR}/logs/fdvla_d_placement_smoke/${SMOKE_ID}
TUNING_ROOT=${REPO_DIR}/logs/fdvla_d_system_tuning/${TUNING_ID}
COUPLED_SUMMARY=${REPO_DIR}/logs/fdvla_task0_long_compare/20260822_weak1000_task0_long_token160_cfirst_v4/C-PPO/summary/fdvla_training_health.json
REGISTERED_PARENT=${REPO_DIR}/logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/fdvla_binary_d_ppo_weak1000_task0_seed0/checkpoints/global_step_50
RAY_ROOT=${FDVLA_RAY_ROOT:-/tmp/f2}
mkdir -p "${TUNING_ROOT}" "${RAY_ROOT}"

# Six updates yield three stable samples after the two-update warmup and the
# final-evaluation event exclusion used by the preregistered selector.
export PPO_MAX_STEPS=6
export TRAIN_NUM_ENVS=60
export TARGET_TRAIN_ENVS=60
export EVAL_NUM_ENVS=48
export TARGET_EVAL_ENVS=48
export TRAIN_MAX_EPISODE_STEPS=480
export TRAIN_ROLLOUT_STEPS=256
export TRAIN_ROLLOUT_EPOCH=4
export PPO_GLOBAL_BATCH_SIZE=384
export SEMANTIC_SKIP_LATEST_BEFORE_EXACT=true
export SEMANTIC_BATCH_MAX_REQUESTS=2
export SEMANTIC_BATCH_TARGET_REQUESTS=2
export SEMANTIC_BATCH_TARGET_ENVS=0
export SEMANTIC_BATCH_WAIT_MS=5
unset PPO_RESUME_DIR

export FDVLA_PLACEMENT_SMOKE_ID=${SMOKE_ID}
export RAY_TMPDIR=${RAY_ROOT}/actor4_rollout3
bash "${SCRIPT_DIR}/run_fdvla_d_placement_smoke.sh" actor4_rollout3
export RAY_TMPDIR=${RAY_ROOT}/colocated4
bash "${SCRIPT_DIR}/run_fdvla_d_placement_smoke.sh" colocated4

REPORT=${TUNING_ROOT}/placement_selection.json
python "${REPO_DIR}/examples/analysis/select_fdvla_d_placement.py" \
    --candidate-set placement \
    --candidate actor4_rollout3="${SMOKE_ROOT}/actor4_rollout3" \
    --candidate colocated4="${SMOKE_ROOT}/colocated4" \
    --coupled-summary "${COUPLED_SUMMARY}" \
    --output "${REPORT}"

selected=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_placement"])' "${REPORT}")
printf 'selected_candidate=%s\nsingle_exact_fetch=true\nbatch_max_requests=2\nselection_report=%s\n' \
    "${selected}" "${REPORT}" >"${TUNING_ROOT}/selected_system.env"

export PPO_RESUME_DIR=${REGISTERED_PARENT}
export PPO_MAX_STEPS=150
export FDVLA_TARGET80_RUN_ID=${TUNING_ID}_target80
export FDVLA_TARGET80_RUN_ROOT=${REPO_DIR}/logs/fdvla_target80_resume/${FDVLA_TARGET80_RUN_ID}
export RAY_TMPDIR=${RAY_ROOT}/target80_${selected}
bash "${SCRIPT_DIR}/run_fdvla_target80_resume.sh" "${selected}"

export FDVLA_TARGET80_TRAIN_RUN=${FDVLA_TARGET80_RUN_ROOT}/${selected}
export FDVLA_HELDOUT_ID=${TUNING_ID}_heldout
export FDVLA_HELDOUT_ROOT=${REPO_DIR}/logs/fdvla_target80_heldout/${FDVLA_HELDOUT_ID}_${selected}
export RAY_TMPDIR=${RAY_ROOT}/heldout_${selected}
bash "${SCRIPT_DIR}/eval_fdvla_target80_heldout.sh" "${selected}"

printf '%s\n' "${TUNING_ROOT}"
