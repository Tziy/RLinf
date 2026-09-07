#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

# This is an explicitly D-favourable high-frequency replanning stress test.
# It must not replace the pre-registered K=8 comparison. Both policies predict
# H=16 actions but execute only K=2 before replanning. D consumes the latest
# completed packet without an exact-age wait; only episode bootstrap may block.
export FDVLA_PROTOCOL_LABEL="exploratory_d_favorable_k2_latest_stress"

export GR00T_MODEL_PATH=${GR00T_MODEL_PATH:-/vepfs-mlp2/c20250301/240403026/async_vla/async_libero_runs/libero10_decoupled_sft_0to8_rebuilt/W1000_libero10_decoupled_0to8_scalar_age_s4_a4/libero10_delay0to8_s4_a4_weak1000/checkpoint-1000}
export GR00T_BACKBONE_PATH=${GR00T_BACKBONE_PATH:-/vepfs-mlp2/c20250301/240403026/async_vla/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}

C_PPO_CKPT=${C_PPO_CKPT:-${REPO_DIR}/logs/fdvla_task0_long_compare/20260822_weak1000_task0_long_token160_cfirst_v4/C-PPO/fdvla_task0_long_C-PPO/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt}
D_PPO_CKPT=${D_PPO_CKPT:-${REPO_DIR}/logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/fdvla_binary_d_ppo_weak1000_task0_seed0/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt}

for required_path in \
    "${GR00T_MODEL_PATH}" \
    "${GR00T_BACKBONE_PATH}" \
    "${C_PPO_CKPT}" \
    "${D_PPO_CKPT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required input does not exist: ${required_path}" >&2
        exit 2
    fi
done

RUN_ID=${FDVLA_K2_RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}
RUN_ROOT=${FDVLA_K2_RUN_ROOT:-${REPO_DIR}/logs/fdvla_k2_d_favorable/${RUN_ID}}
RUN_C=${FDVLA_K2_RUN_C:-true}
RUN_D=${FDVLA_K2_RUN_D:-true}
if [[ "${RUN_C}" != true && "${RUN_D}" != true ]]; then
    echo "At least one of FDVLA_K2_RUN_C or FDVLA_K2_RUN_D must be true." >&2
    exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
    echo "Refusing to overwrite existing K=2 run: ${RUN_ROOT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"

# Paired evaluation contract. Keep prediction horizon H=16 and change only the
# executed prefix to K=2. Forty-eight envs give one fixed Task-0 trial each.
export ONLY_EVAL=true
export ACTION_CHUNK_SIZE=16
export EVAL_EXECUTION_HORIZON=2
export DENOISING_STEPS=4
export EVAL_TASK_ID_FILTER='[0]'
export EVAL_NUM_ENVS=48
export TARGET_EVAL_ENVS=48
export EVAL_MAX_EPISODE_STEPS=480
export EVAL_ROLLOUT_STEPS=480
export EVAL_ROLLOUT_EPOCH=1
export EVAL_AUTO_RESET=false
export EVAL_IGNORE_TERMINATIONS=false
export EVAL_SEED=${EVAL_SEED:-0}
export EVAL_NOISE_SEED=${EVAL_NOISE_SEED:-13026}
export DETERMINISTIC_EVAL_NOISE=true
export SEMANTIC_TEXT_PADDING_TOKENS=160
export SEMANTIC_EVAL_FIXED_AGE_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
export SEMANTIC_FETCH_TARGET_AGE_FRAMES=-1
export SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES=-1
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/fdvla_k2_latest}

cat >"${RUN_ROOT}/protocol.env" <<EOF
protocol_label=${FDVLA_PROTOCOL_LABEL}
primary_result=false
task_id=0
trial_count=48
eval_seed=${EVAL_SEED}
policy_noise_seed=${EVAL_NOISE_SEED}
action_prediction_horizon=16
action_execution_horizon=2
semantic_mode_C=current_frame_local_vlm
semantic_mode_D=nonblocking_latest_completed_packet
semantic_fixed_age_D=-1
semantic_random_age_D=-1
C_checkpoint=${C_PPO_CKPT}
D_checkpoint=${D_PPO_CKPT}
run_C=${RUN_C}
run_D=${RUN_D}
D_semantic_gpu_ids=${D_SEMANTIC_GPU_IDS:-0}
EOF

run_arm() {
    local arm=$1
    local checkpoint=$2
    local launcher=$3
    local run_dir="${RUN_ROOT}/${arm}"
    mkdir -p "${run_dir}"

    export PPO_CKPT_PATH="${checkpoint}"
    export FDVLA_METHOD_ID="${arm}"
    export FDVLA_RUN_TIMESTAMP="${RUN_ID}_${arm}"
    export FDVLA_RUN_LOG_DIR="${run_dir}"
    export RLINF_LOG_DIR="${run_dir}"
    export FDVLA_METADATA_DIR="${run_dir}/fdvla_metadata"
    export PPO_EXPERIMENT_NAME="fdvla_k2_latest_stress_${arm}"

    printf 'arm=%s\nstart_utc=%s\n' \
        "${arm}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >"${run_dir}/run.env"
    local start_ns end_ns
    start_ns=$(date +%s%N)
    bash "${launcher}" >"${run_dir}/launcher.log" 2>&1
    end_ns=$(date +%s%N)
    awk -v start_ns="${start_ns}" -v end_ns="${end_ns}" \
        'BEGIN {printf "process_elapsed_seconds=%.3f\n", (end_ns - start_ns) / 1000000000}' \
        >"${run_dir}/process_time.txt"
    printf 'end_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >>"${run_dir}/run.env"

    if [[ ! -s "${run_dir}/eval_trials.jsonl" ]]; then
        echo "Missing paired-eval artifact: ${run_dir}/eval_trials.jsonl" >&2
        exit 1
    fi
    if [[ "$(wc -l <"${run_dir}/eval_trials.jsonl")" -ne 48 ]]; then
        echo "Expected exactly 48 trials for ${arm}." >&2
        exit 1
    fi
}

# Coupled: native current-frame VLM, all four GPUs available to the policy.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ACTOR_GPU_PLACEMENT=0-3
export ROLLOUT_GPU_PLACEMENT=0-3
export ENV_GPU_PLACEMENT=0-3
if [[ "${RUN_C}" == true ]]; then
    run_arm C-PPO "${C_PPO_CKPT}" "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh"
fi

# Decoupled: semantic server on GPU 0 and DiT colocated across all four GPUs.
# Publish at frame 1 because K=2; the action boundary still never requests an
# exact packet and therefore does not wait for a target semantic age.
export SEMANTIC_GPU_IDS=${D_SEMANTIC_GPU_IDS:-0}
export DIT_GPU_IDS=0,1,2,3
export ACTOR_GPU_IDS=0,1,2,3
export ALLOWED_GPU_IDS=0,1,2,3
export ACTOR_GPU_PLACEMENT=0-3
export ROLLOUT_GPU_PLACEMENT=0-3
export ENV_GPU_PLACEMENT=0-3
export SEMANTIC_MID_CHUNK_PUBLISH=true
export SEMANTIC_MID_CHUNK_FRAME=1
export SEMANTIC_MID_CHUNK_MIN_FRAME=1
export SEMANTIC_MID_CHUNK_STAGGER_BY_RANK=false
export SEMANTIC_ENV_BOUNDARY_PUBLISH=false
export SEMANTIC_BOUNDARY_PUBLISH=false
export SEMANTIC_BATCH_MAX_REQUESTS=${SEMANTIC_BATCH_MAX_REQUESTS:-4}
export SEMANTIC_BATCH_TARGET_REQUESTS=${SEMANTIC_BATCH_TARGET_REQUESTS:-4}
export SEMANTIC_BATCH_TARGET_ENVS=${SEMANTIC_BATCH_TARGET_ENVS:-0}
export SEMANTIC_BATCH_WAIT_MS=${SEMANTIC_BATCH_WAIT_MS:-5}
export SEMANTIC_BOOTSTRAP_TARGET_ENVS=${SEMANTIC_BOOTSTRAP_TARGET_ENVS:-0}
export SEMANTIC_RPC_BATCH_WAIT_MS=${SEMANTIC_RPC_BATCH_WAIT_MS:-2}
export SEMANTIC_PREPROCESS_WORKERS=${SEMANTIC_PREPROCESS_WORKERS:-12}
export SEMANTIC_CACHE_HISTORY_SIZE=${SEMANTIC_CACHE_HISTORY_SIZE:-32}
if [[ "${RUN_D}" == true ]]; then
    run_arm D-PPO "${D_PPO_CKPT}" "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh"
fi

printf '%s\n' "${RUN_ROOT}"
