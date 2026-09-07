#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the preregistered weak1000 checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the frozen backbone}"

SMOKE_ID=${FDVLA_NATURAL_SMOKE_ID:-20260822_scale60_natural_async_col4_v1}
TUNING_ID=${FDVLA_NATURAL_TUNING_ID:-20260822_weak1000_natural_async_target80_v1}
SMOKE_DIR=${REPO_DIR}/logs/fdvla_d_placement_smoke/${SMOKE_ID}/colocated4
TUNING_ROOT=${REPO_DIR}/logs/fdvla_d_system_tuning/${TUNING_ID}
COUPLED_SUMMARY=${REPO_DIR}/logs/fdvla_task0_long_compare/20260822_weak1000_task0_long_token160_cfirst_v4/C-PPO/summary/fdvla_training_health.json
REGISTERED_PARENT=${REPO_DIR}/logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/fdvla_binary_d_ppo_weak1000_task0_seed0/checkpoints/global_step_50
RAY_ROOT=${FDVLA_RAY_ROOT:-/tmp/f3}
mkdir -p "${TUNING_ROOT}" "${RAY_ROOT}"

export PPO_MAX_STEPS=6
export TRAIN_NUM_ENVS=60
export TARGET_TRAIN_ENVS=60
export EVAL_NUM_ENVS=48
export TARGET_EVAL_ENVS=48
export TRAIN_MAX_EPISODE_STEPS=480
export TRAIN_ROLLOUT_STEPS=256
export TRAIN_ROLLOUT_EPOCH=4
export PPO_GLOBAL_BATCH_SIZE=384
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=-1
export SEMANTIC_SKIP_LATEST_BEFORE_EXACT=true
export SEMANTIC_BATCH_MAX_REQUESTS=2
export SEMANTIC_BATCH_TARGET_REQUESTS=2
export SEMANTIC_BATCH_TARGET_ENVS=0
export SEMANTIC_BATCH_WAIT_MS=5
unset PPO_RESUME_DIR

export FDVLA_PLACEMENT_SMOKE_ID=${SMOKE_ID}
export RAY_TMPDIR=${RAY_ROOT}/smoke
bash "${SCRIPT_DIR}/run_fdvla_d_placement_smoke.sh" colocated4

REPORT=${TUNING_ROOT}/natural_async_speed_gate.json
python - "${SMOKE_DIR}" "${COUPLED_SUMMARY}" "${REPORT}" <<'PY'
import json
import sys
from pathlib import Path

run_dir, coupled_path, output_path = map(Path, sys.argv[1:])
summary = json.loads((run_dir / "summary/fdvla_training_health.json").read_text())
coupled = json.loads(coupled_path.read_text())
zero_fields = (
    "nonfinite_scalar_count",
    "reward_model_invocation_count",
    "trainable_vlm_parameter_count",
    "local_vlm_forward_count",
    "cross_episode_packet_mismatch_count",
    "semantic_replay_fingerprint_mismatch_log_count",
)
violations = {key: summary.get(key) for key in zero_fields if summary.get(key) != 0}
stable = summary["stable_timing"]["metrics"]["time/step"]
stable_count = int(stable["count"])
d_mean = float(stable["mean"])
c_mean = float(coupled["stable_timing"]["metrics"]["time/step"]["mean"])
report = {
    "schema_version": 1,
    "selection_inputs_exclude_policy_success": True,
    "method": "D-PPO-natural-async-colocated4",
    "run_dir": str(run_dir.resolve()),
    "health_zero_field_violations": violations,
    "stable_sample_count": stable_count,
    "d_stable_step_mean_s": d_mean,
    "c_stable_step_mean_s": c_mean,
    "d_time_reduction_fraction": (c_mean - d_mean) / c_mean,
    "d_throughput_speedup_fraction": c_mean / d_mean - 1.0,
    "speed_endpoint_passed": d_mean < c_mean,
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if violations or stable_count < 3 or d_mean >= c_mean:
    raise SystemExit(3)
PY

export PPO_RESUME_DIR=${REGISTERED_PARENT}
export PPO_MAX_STEPS=150
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
export FDVLA_TARGET80_RUN_ID=${TUNING_ID}
export FDVLA_TARGET80_RUN_ROOT=${REPO_DIR}/logs/fdvla_target80_resume/${FDVLA_TARGET80_RUN_ID}
export RAY_TMPDIR=${RAY_ROOT}/target80
bash "${SCRIPT_DIR}/run_fdvla_target80_resume.sh" colocated4

export FDVLA_TARGET80_TRAIN_RUN=${FDVLA_TARGET80_RUN_ROOT}/colocated4
export FDVLA_HELDOUT_ID=${TUNING_ID}_heldout
export FDVLA_HELDOUT_ROOT=${REPO_DIR}/logs/fdvla_target80_heldout/${FDVLA_HELDOUT_ID}_colocated4
export RAY_TMPDIR=${RAY_ROOT}/heldout
bash "${SCRIPT_DIR}/eval_fdvla_target80_heldout.sh" colocated4

printf '%s\n' "${TUNING_ROOT}"
