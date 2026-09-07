#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the common initial checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the common frozen backbone}"

VARIANT=${1:-actor4_rollout3}
SMOKE_ID=${FDVLA_PLACEMENT_SMOKE_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_DIR=${FDVLA_PLACEMENT_SMOKE_ROOT:-"${REPO_DIR}/logs/fdvla_d_placement_smoke/${SMOKE_ID}"}/${VARIANT}

case "${VARIANT}" in
    actor4_rollout3)
        # Keep rollout inference off the semantic-server GPU, but use all four
        # GPUs for the sharded PPO actor update.
        export SEMANTIC_GPU_IDS=0
        export DIT_GPU_IDS=1,2,3
        export ACTOR_GPU_IDS=0,1,2,3
        ;;
    server2_dit2)
        # Two independent semantic servers feed two dedicated DiT GPUs. Keep
        # the PPO actor four-way sharded so only rollout placement changes.
        export SEMANTIC_GPU_IDS=0,1
        export DIT_GPU_IDS=2,3
        export ACTOR_GPU_IDS=0,1,2,3
        ;;
    server2_dit1_env1)
        # Keep semantic inference, DiT action generation, and LIBERO
        # simulation on disjoint physical GPUs. In RLinf the RolloutGroup owns
        # the DiT model, so GPU 3 hosts only EnvGroup interaction/rendering;
        # actor FSDP still time-shares all four GPUs during PPO updates.
        export SEMANTIC_GPU_IDS=0,1
        export DIT_GPU_IDS=2
        export ENV_GPU_IDS=3
        export ACTOR_GPU_IDS=0,1,2,3
        ;;
    server3_dit1)
        # Edge-style placement: three semantic GPUs feed one DiT rollout
        # replica. Central-cache rows are sharded by env_id across all servers.
        export SEMANTIC_GPU_IDS=0,1,2
        export DIT_GPU_IDS=3
        export ACTOR_GPU_IDS=0,1,2
        ;;
    server3_dit1_r3)
        # Three lightweight DiT rollout replicas share the sole DiT GPU. This
        # preserves one-GPU action compute while matching the baseline's three
        # rollout/env processes for a scheduler-control comparison.
        export SEMANTIC_GPU_IDS=0,1,2
        export DIT_GPU_IDS=3
        export ACTOR_GPU_IDS=0,1,2
        export DIT_REPLICAS_PER_GPU=3
        ;;
    colocated4)
        # Share GPU 0 between the frozen semantic server and one DiT rollout
        # rank. This preserves the four-GPU total budget and four-way actor.
        export SEMANTIC_GPU_IDS=0
        export DIT_GPU_IDS=0,1,2,3
        export ACTOR_GPU_IDS=0,1,2,3
        export COLOCATED_SEMANTIC_FETCH_PAUSE_MS=${COLOCATED_SEMANTIC_FETCH_PAUSE_MS:-0}
        ;;
    colocated4_2s)
        # Two semantic servers reduce natural packet age while retaining all
        # four GPUs for DiT rollout and the sharded PPO actor update.
        export SEMANTIC_GPU_IDS=0,1
        export DIT_GPU_IDS=0,1,2,3
        export ACTOR_GPU_IDS=0,1,2,3
        export COLOCATED_SEMANTIC_FETCH_PAUSE_MS=${COLOCATED_SEMANTIC_FETCH_PAUSE_MS:-0}
        ;;
    colocated4_4s)
        # One semantic server per rollout rank maximizes semantic throughput;
        # all servers remain colocated inside the same four-GPU total budget.
        export SEMANTIC_GPU_IDS=0,1,2,3
        export DIT_GPU_IDS=0,1,2,3
        export ACTOR_GPU_IDS=0,1,2,3
        export COLOCATED_SEMANTIC_FETCH_PAUSE_MS=${COLOCATED_SEMANTIC_FETCH_PAUSE_MS:-0}
        ;;
    *)
        echo "Unknown placement variant: ${VARIANT}" >&2
        exit 2
        ;;
esac

export ALLOWED_GPU_IDS=0,1,2,3
export TRAIN_TASK_ID_FILTER=${TRAIN_TASK_ID_FILTER:-[0]}
export EVAL_TASK_ID_FILTER=${EVAL_TASK_ID_FILTER:-[0]}
export BALANCE_TRAIN_TASK_ASSIGNMENT=false
export PRESERVE_TRAIN_TASK_ASSIGNMENT=true
export TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS:-12}
export EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-12}
export TARGET_TRAIN_ENVS=${TARGET_TRAIN_ENVS:-12}
export TARGET_EVAL_ENVS=${TARGET_EVAL_ENVS:-12}
export TRAIN_MAX_EPISODE_STEPS=${TRAIN_MAX_EPISODE_STEPS:-64}
export TRAIN_ROLLOUT_STEPS=${TRAIN_ROLLOUT_STEPS:-64}
export EVAL_MAX_EPISODE_STEPS=${EVAL_MAX_EPISODE_STEPS:-64}
export EVAL_ROLLOUT_STEPS=${EVAL_ROLLOUT_STEPS:-64}
export TRAIN_ROLLOUT_EPOCH=${TRAIN_ROLLOUT_EPOCH:-1}
export PPO_MAX_STEPS=${PPO_MAX_STEPS:-5}
export PPO_GLOBAL_BATCH_SIZE=${PPO_GLOBAL_BATCH_SIZE:-24}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
export PPO_UPDATE_EPOCHS=${PPO_UPDATE_EPOCHS:-1}
export PPO_EVAL_BEFORE_TRAINING=${PPO_EVAL_BEFORE_TRAINING:-false}
export PPO_VAL_INTERVAL=${PPO_VAL_INTERVAL:-1000}
export PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:--1}
export PPO_SUCCESS_EARLY_STOP_ENABLED=false
export ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-4}
export TRAIN_SEED=${TRAIN_SEED:-0}
export EVAL_SEED=${EVAL_SEED:-0}
export RANDOMIZE_TRAIN_SIM_SEED_ON_RESET=false
export FDVLA_METHOD_ID="D-PPO-${VARIANT}"
export PPO_EXPERIMENT_NAME="fdvla_d_placement_smoke_${VARIANT}"
export FDVLA_RUN_TIMESTAMP="${SMOKE_ID}_${VARIANT}"
export FDVLA_RUN_LOG_DIR="${RUN_DIR}"
export RLINF_LOG_DIR="${RUN_DIR}"
export FDVLA_METADATA_DIR="${RUN_DIR}/fdvla_metadata"
export RAY_TMPDIR=${RAY_TMPDIR:-"/tmp/fdvla_d_${VARIANT}"}

mkdir -p "${RUN_DIR}"
printf 'smoke_id=%s\nvariant=%s\nstart_utc=%s\n' \
    "${SMOKE_ID}" "${VARIANT}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    >"${RUN_DIR}/run.env"

start_ns=$(date +%s%N)
set +e
bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh" >"${RUN_DIR}/launcher.log" 2>&1
launcher_status=$?
set -e
end_ns=$(date +%s%N)
awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
    'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
    >"${RUN_DIR}/process_time.txt"
printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >>"${RUN_DIR}/run.env"
if (( launcher_status != 0 )) ||
    rg -q \
        'Exiting main process due to a failure|Error executing job with overrides' \
        "${RUN_DIR}/launcher.log"; then
    echo "Placement smoke worker failure: ${VARIANT}" >&2
    exit 1
fi

python "${REPO_DIR}/examples/analysis/summarize_fdvla_training_run.py" \
    --run-dir "${RUN_DIR}" \
    --run-log "${RUN_DIR}/launcher.log" \
    --output-dir "${RUN_DIR}/summary"

printf '%s\n' "${RUN_DIR}"
