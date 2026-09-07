#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/fdvla_binary_common.sh"

fdvla_require_inputs
fdvla_apply_protocol_defaults
FDVLA_METHOD_ID=${FDVLA_METHOD_ID:-D-PPO}
export PPO_EXPERIMENT_NAME=${PPO_EXPERIMENT_NAME:-fdvla_binary_d_ppo}
export SEMANTIC_GPU_IDS=${SEMANTIC_GPU_IDS:-0}
export DIT_GPU_IDS=${DIT_GPU_IDS:-1,2,3}
export ACTOR_GPU_IDS=${ACTOR_GPU_IDS:-1,2,3}
export ALLOWED_GPU_IDS=${ALLOWED_GPU_IDS:-0,1,2,3}
export TARGET_TRAIN_ENVS=${TARGET_TRAIN_ENVS:-${TRAIN_NUM_ENVS}}
export TARGET_EVAL_ENVS=${TARGET_EVAL_ENVS:-${EVAL_NUM_ENVS}}

fdvla_prepare_run_paths libero_10_fdvla_binary_ppo "${FDVLA_METHOD_ID}"
metadata_dir=$(fdvla_write_run_metadata libero_10_fdvla_binary_ppo "${FDVLA_METHOD_ID}")
echo "FDVLA run metadata: ${metadata_dir}"
exec bash "${SCRIPT_DIR}/run_gr00t_semantic_cache_sync_ppo.sh"     libero_10_fdvla_binary_ppo
