#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/fdvla_binary_common.sh"

fdvla_require_inputs
fdvla_apply_protocol_defaults
# Coupled is the native current-frame baseline. Do not let the decoupled
# protocol defaults route it through the exact-age central-cache path.
export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=-1
FDVLA_METHOD_ID=${FDVLA_METHOD_ID:-C-PPO}
export PPO_EXPERIMENT_NAME=${PPO_EXPERIMENT_NAME:-fdvla_binary_c_ppo}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ACTOR_GPU_PLACEMENT=${ACTOR_GPU_PLACEMENT:-0-3}
export ROLLOUT_GPU_PLACEMENT=${ROLLOUT_GPU_PLACEMENT:-0-3}
export ENV_GPU_PLACEMENT=${ENV_GPU_PLACEMENT:-0-3}
export ROBOT_PLATFORM=LIBERO
export MUJOCO_GL=${MUJOCO_GL:-egl}

fdvla_prepare_run_paths libero_10_coupled_n1d7_binary_ppo "${FDVLA_METHOD_ID}"
metadata_dir=$(fdvla_write_run_metadata libero_10_coupled_n1d7_binary_ppo "${FDVLA_METHOD_ID}")
echo "FDVLA run metadata: ${metadata_dir}"
exec bash "${SCRIPT_DIR}/run_embodiment.sh"     libero_10_coupled_n1d7_binary_ppo LIBERO
