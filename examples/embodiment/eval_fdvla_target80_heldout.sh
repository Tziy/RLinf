#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the preregistered weak1000 checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the frozen backbone}"
: "${FDVLA_TARGET80_TRAIN_RUN:?Set the completed target80 training run directory}"

if [[ "${FDVLA_TARGET80_TRAIN_RUN}" != /* ]]; then
    echo "FDVLA_TARGET80_TRAIN_RUN must be absolute." >&2
    exit 2
fi
REGISTERED_MODEL_PATH=/vepfs-mlp2/c20250301/240403026/async_vla/async_libero_runs/libero10_decoupled_sft_0to8_rebuilt/W1000_libero10_decoupled_0to8_scalar_age_s4_a4/libero10_delay0to8_s4_a4_weak1000/checkpoint-1000
if [[ "${GR00T_MODEL_PATH}" != "${REGISTERED_MODEL_PATH}" ]]; then
    echo "GR00T_MODEL_PATH differs from the preregistered weak1000 checkpoint." >&2
    exit 2
fi

# Deterministic selection: first preregistered crossing, otherwise the unique
# maximum-step checkpoint. An arbitrary PPO_CKPT_PATH is never accepted.
selected_checkpoint=$(python - "${FDVLA_TARGET80_TRAIN_RUN}" <<'PY_SELECT'
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
record_path = run_dir / "development_success_early_stop.json"
if record_path.exists():
    record = json.loads(record_path.read_text())
    assert record["protocol_label"] == "weak1000_seed0_continue_from_step50_target80"
    assert record["selection_role"] == "development_only"
    assert record["rule"] == "first_fixed_evaluation_at_or_above_threshold"
    assert float(record["threshold"]) == 0.75
    assert list(record["desired_band"]) == [0.75, 0.85]
    step = int(record["global_step"])
    assert 50 <= step <= 150 and step % 10 == 0
    checkpoint = Path(record["checkpoint_dir"])
else:
    matches = list(run_dir.glob("**/checkpoints/global_step_150"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one no-hit global_step_150 checkpoint, found {len(matches)}"
        )
    checkpoint = matches[0]
weights = checkpoint / "actor/model_state_dict/full_weights.pt"
if not weights.is_file():
    raise FileNotFoundError(weights)
print(weights.resolve())
PY_SELECT
)
unset PPO_CKPT_PATH
export PPO_CKPT_PATH=${selected_checkpoint}

PLACEMENT=${1:-actor4_rollout3}
case "${PLACEMENT}" in
    actor4_rollout3)
        export SEMANTIC_GPU_IDS=0
        export DIT_GPU_IDS=1,2,3
        export ACTOR_GPU_IDS=0,1,2,3
        ;;
    colocated4)
        export SEMANTIC_GPU_IDS=0
        export DIT_GPU_IDS=0,1,2,3
        export ACTOR_GPU_IDS=0,1,2,3
        export COLOCATED_SEMANTIC_FETCH_PAUSE_MS=${COLOCATED_SEMANTIC_FETCH_PAUSE_MS:-0}
        ;;
    *)
        echo "Unknown placement: ${PLACEMENT}" >&2
        exit 2
        ;;
esac
export ALLOWED_GPU_IDS=0,1,2,3

HELDOUT_ID=${FDVLA_HELDOUT_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
HELDOUT_ROOT=${FDVLA_HELDOUT_ROOT:-${REPO_DIR}/logs/fdvla_target80_heldout/${HELDOUT_ID}_${PLACEMENT}}
if [[ "${HELDOUT_ROOT}" != /* ]]; then
    echo "FDVLA_HELDOUT_ROOT must be absolute." >&2
    exit 2
fi
mkdir -p "${HELDOUT_ROOT}"
printf 'source_training_run=%s\nselected_checkpoint=%s\npolicy_noise_seed=13027\n' \
    "${FDVLA_TARGET80_TRAIN_RUN}" "${selected_checkpoint}" \
    >"${HELDOUT_ROOT}/heldout_selection.env"

export DRY_RUN=false
export RESUME_SURFACE=false
export SURFACE_SWEEP_MODE=cartesian
export SEMANTIC_AGES=6
export ACTION_HORIZONS=8
export SURFACE_MIN_TRIALS=48
export SURFACE_EVAL_NUM_ENVS=48
export SURFACE_TASK_ID_FILTER='[0]'
export SURFACE_POLICY_NOISE_SEEDS=13027
export ACTION_CHUNK_SIZE=16
export EVAL_ROLLOUT_EPOCH=1
export FDVLA_SURFACE_ROOT=${HELDOUT_ROOT}/surface
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/fdvla_target80_heldout_${PLACEMENT}}

POLICY_METHOD=D-PPO bash "${SCRIPT_DIR}/eval_fdvla_realtime_surface.sh" \
    >"${HELDOUT_ROOT}/heldout_launcher.log" 2>&1
printf '%s\n' "${FDVLA_SURFACE_ROOT}/surface_cells.tsv"
