#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the frozen D-SFT checkpoint}"
: "${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH to the frozen backbone}"

RUN_ID=${FDVLA_BATCHING_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')_task0_k2_bf16_tail4}
RUN_ROOT=${FDVLA_BATCHING_RUN_ROOT:-${REPO_DIR}/logs/fdvla_batching_selection/${RUN_ID}}
COUPLED_SUMMARY=${FDVLA_BATCHING_COUPLED_SUMMARY:-}

if [[ -e "${RUN_ROOT}/protocol.env" || -e "${RUN_ROOT}/selection.json" ]]; then
    echo "Batching selection root already initialized: ${RUN_ROOT}" >&2
    echo "Use a new FDVLA_BATCHING_RUN_ID; results are never overwritten." >&2
    exit 2
fi
if [[ -n "${COUPLED_SUMMARY}" && ! -f "${COUPLED_SUMMARY}" ]]; then
    echo "Missing optional coupled timing summary: ${COUPLED_SUMMARY}" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}"
cp "${SCRIPT_DIR}/config/fdvla_binary_run_manifest.yaml" \
    "${RUN_ROOT}/preregistered_manifest.yaml"
sha256sum "${RUN_ROOT}/preregistered_manifest.yaml" \
    >"${RUN_ROOT}/preregistered_manifest.sha256"
git -C "${REPO_DIR}" rev-parse HEAD >"${RUN_ROOT}/git_sha.txt"
git -C "${REPO_DIR}" status --short >"${RUN_ROOT}/git_status_before.txt"

{
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'candidate_order=batch1,batch2,batch4\n'
    printf 'batch1=max1,target1,wait0ms\n'
    printf 'batch2=max2,target2,wait1ms\n'
    printf 'batch4=max4,target4,wait2ms\n'
    printf 'selection_primary=mean_training_wallclock_per_update\n'
    printf 'selection_tie_breakers=mean_rollout_wallclock,mean_semantic_queue_p95,candidate_order\n'
    printf 'selection_uses_success=false\n'
    printf 'train_envs=60\nrollout_epochs=4\nrollout_steps=256\nupdates=6\n'
    printf 'warmup_updates_excluded=2\nfinal_eval_enclosing_update_excluded=1\nstable_updates_required=3\n'
    printf 'execution_horizon=2\nsemantic_publish_interval_frames=8\n'
    printf 'semantic_mode=natural_async_latest_nonblocking\nposthoc_augmentation=false\n'
    printf 'start_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
} >"${RUN_ROOT}/protocol.env"

run_candidate() {
    local candidate=$1
    local max_requests=$2
    local target_requests=$3
    local wait_ms=$4
    local ray_tmp
    ray_tmp=$(mktemp -d "/dev/shm/fdvb.${candidate}.XXXXXX")

    env \
        GR00T_MODEL_PATH="${GR00T_MODEL_PATH}" \
        GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH}" \
        TRAIN_NUM_ENVS=60 TARGET_TRAIN_ENVS=60 \
        EVAL_NUM_ENVS=48 TARGET_EVAL_ENVS=48 \
        TRAIN_MAX_EPISODE_STEPS=480 TRAIN_ROLLOUT_STEPS=256 \
        TRAIN_ROLLOUT_EPOCH=4 PPO_MAX_STEPS=6 \
        PPO_GLOBAL_BATCH_SIZE=960 PPO_MICRO_BATCH_SIZE=2 PPO_UPDATE_EPOCHS=1 \
        PPO_EVAL_BEFORE_TRAINING=false PPO_VAL_INTERVAL=1000 PPO_SAVE_INTERVAL=-1 \
        PPO_ACTOR_LR=5.0e-7 PPO_VALUE_LR=2.0e-5 PPO_KL_BETA=0.0 \
        FDVLA_ACTOR_MODEL_PRECISION=bf16 FDVLA_ROLLOUT_MODEL_PRECISION=bf16 \
        DIT_TRAIN_LAST_N_BLOCKS=4 TRAIN_VALUE_HEAD_WITH_DIT_ONLY=true \
        ACTION_CHUNK_SIZE=16 TRAIN_EXECUTION_HORIZON=2 EVAL_EXECUTION_HORIZON=2 \
        DENOISING_STEPS=4 SEMANTIC_PUBLISH_INTERVAL_FRAMES=8 \
        SEMANTIC_MID_CHUNK_PUBLISH=false SEMANTIC_ENV_BOUNDARY_PUBLISH=false \
        SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1 \
        SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1 \
        SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1 \
        SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1 \
        POSTHOC_SEMANTIC_DELAY_ENABLED=false \
        SEMANTIC_BATCH_TARGET_ENVS=0 SEMANTIC_RPC_BATCH_WAIT_MS=2 \
        SEMANTIC_PREPROCESS_PROXY=false ENV_WORKERS_PER_GPU=1 \
        MAX_USED_MIB=1024 COLOCATED_SEMANTIC_FETCH_PAUSE_MS=0 \
        TRAIN_SEED=0 EVAL_SEED=0 \
        FDVLA_PLACEMENT_SMOKE_ID="${RUN_ID}_${candidate}" \
        FDVLA_PLACEMENT_SMOKE_ROOT="${RUN_ROOT}/${candidate}" \
        RAY_TMPDIR="${ray_tmp}" \
        SEMANTIC_BATCH_MAX_REQUESTS="${max_requests}" \
        SEMANTIC_BATCH_TARGET_REQUESTS="${target_requests}" \
        SEMANTIC_BATCH_WAIT_MS="${wait_ms}" \
        bash "${SCRIPT_DIR}/run_fdvla_d_placement_smoke.sh" colocated4_4s
}

run_candidate batch1 1 1 0
run_candidate batch2 2 2 1
run_candidate batch4 4 4 2

selector=(
    python "${REPO_DIR}/examples/analysis/select_fdvla_d_placement.py"
    --candidate-set batching
    --min-stable-samples 3
    --candidate "batch1=${RUN_ROOT}/batch1/colocated4_4s"
    --candidate "batch2=${RUN_ROOT}/batch2/colocated4_4s"
    --candidate "batch4=${RUN_ROOT}/batch4/colocated4_4s"
    --output "${RUN_ROOT}/selection.json"
    --env-output "${RUN_ROOT}/selected_batching.env"
)
if [[ -n "${COUPLED_SUMMARY}" ]]; then
    selector+=(--coupled-summary "${COUPLED_SUMMARY}")
fi
"${selector[@]}" >"${RUN_ROOT}/selection.stdout.json"
printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    >>"${RUN_ROOT}/protocol.env"
printf '%s\n' "${RUN_ROOT}"
