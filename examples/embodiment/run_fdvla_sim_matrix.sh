#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/fdvla_binary_common.sh"

METHOD=${METHOD:-D-PPO}
STAGE=${STAGE:-smoke}
DRY_RUN=${DRY_RUN:-true}
export FDVLA_METHOD_ID=${FDVLA_METHOD_ID:-${METHOD}}
method_slug=${METHOD//-/_}
method_slug=${method_slug,,}
export PPO_EXPERIMENT_NAME=${PPO_EXPERIMENT_NAME:-fdvla_binary_${method_slug}}

case "${STAGE}" in
    smoke)
        export TRAIN_TASK_ID_FILTER=${TRAIN_TASK_ID_FILTER:-'[0]'}
        export EVAL_TASK_ID_FILTER=${EVAL_TASK_ID_FILTER:-'[0]'}
        export TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS:-2}
        export EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-2}
        export TARGET_TRAIN_ENVS=${TARGET_TRAIN_ENVS:-2}
        export TARGET_EVAL_ENVS=${TARGET_EVAL_ENVS:-2}
        export TRAIN_ROLLOUT_EPOCH=${TRAIN_ROLLOUT_EPOCH:-1}
        export TRAIN_ROLLOUT_STEPS=${TRAIN_ROLLOUT_STEPS:-32}
        export EVAL_ROLLOUT_STEPS=${EVAL_ROLLOUT_STEPS:-32}
        export TRAIN_MAX_EPISODE_STEPS=${TRAIN_MAX_EPISODE_STEPS:-32}
        export EVAL_MAX_EPISODE_STEPS=${EVAL_MAX_EPISODE_STEPS:-32}
        export PPO_GLOBAL_BATCH_SIZE=${PPO_GLOBAL_BATCH_SIZE:-4}
        export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
        export PPO_MAX_STEPS=${PPO_MAX_STEPS:-2}
        export PPO_VAL_INTERVAL=${PPO_VAL_INTERVAL:-2}
        export PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:-2}
        export BALANCE_TRAIN_TASK_ASSIGNMENT=false
        export SEMANTIC_GPU_IDS=${SEMANTIC_GPU_IDS:-0}
        export DIT_GPU_IDS=${DIT_GPU_IDS:-1}
        export ACTOR_GPU_IDS=${ACTOR_GPU_IDS:-1}
        export ALLOWED_GPU_IDS=${ALLOWED_GPU_IDS:-0,1}
        ;;
    pilot)
        export TRAIN_TASK_ID_FILTER=${TRAIN_TASK_ID_FILTER:-'[0]'}
        export EVAL_TASK_ID_FILTER=${EVAL_TASK_ID_FILTER:-'[0]'}
        export PPO_VAL_INTERVAL=${PPO_VAL_INTERVAL:-10}
        export PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:-10}
        ;;
    full)
        echo "Full LIBERO-10 is intentionally not launched by default." >&2
        [[ "${ALLOW_FULL_MATRIX:-false}" == true ]] || exit 2
        # Development evaluations cover all ten tasks with equal weight. 120 is
        # divisible by the 10 tasks and by both legal worker placements used in
        # this study (four coupled workers, three decoupled DiT workers).
        unset TRAIN_TASK_ID_FILTER EVAL_TASK_ID_FILTER
        export TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS:-60}
        export EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-120}
        export TARGET_TRAIN_ENVS=${TARGET_TRAIN_ENVS:-${TRAIN_NUM_ENVS}}
        export TARGET_EVAL_ENVS=${TARGET_EVAL_ENVS:-${EVAL_NUM_ENVS}}
        export PPO_VAL_INTERVAL=${PPO_VAL_INTERVAL:-10}
        export PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:-10}
        ;;
    full_eval)
        echo "Formal full LIBERO-10 evaluation is intentionally not launched by default." >&2
        [[ "${ALLOW_FULL_MATRIX:-false}" == true ]] || exit 2
        # Use 42 fixed trials per task. 420 is at least the pre-registered 400
        # trials and is divisible by 10 tasks, four coupled workers, and three
        # decoupled DiT workers, so neither launcher rounds away an episode.
        unset TRAIN_TASK_ID_FILTER EVAL_TASK_ID_FILTER
        export ONLY_EVAL=true
        export EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-420}
        export TARGET_EVAL_ENVS=${TARGET_EVAL_ENVS:-${EVAL_NUM_ENVS}}
        export EVAL_ROLLOUT_EPOCH=${EVAL_ROLLOUT_EPOCH:-1}
        export EVAL_AUTO_RESET=${EVAL_AUTO_RESET:-false}
        export EVAL_IGNORE_TERMINATIONS=${EVAL_IGNORE_TERMINATIONS:-false}
        export EVAL_MAX_EPISODE_STEPS=${EVAL_MAX_EPISODE_STEPS:-480}
        export EVAL_ROLLOUT_STEPS=${EVAL_ROLLOUT_STEPS:-480}
        ;;
    *)
        echo "Unknown STAGE=${STAGE}; expected smoke, pilot, or full." >&2
        exit 2
        ;;
esac

if [[ "${STAGE}" == full || "${STAGE}" == full_eval ]]; then
    if (( EVAL_NUM_ENVS % 10 != 0 )); then
        echo "Full LIBERO-10 evaluation requires EVAL_NUM_ENVS divisible by 10: ${EVAL_NUM_ENVS}" >&2
        exit 2
    fi
fi

if [[ "${STAGE}" == pilot && "${METHOD}" == D-* && "${METHOD}" != D-PPO-Fresh ]]; then
    # Intermediate learning curves must keep the requested semantic condition
    # identical across checkpoints. Random per-boundary scheduling is retained
    # for training and evaluated separately as a deployment condition.
    export SEMANTIC_EVAL_FIXED_AGE_FRAMES=${SEMANTIC_EVAL_FIXED_AGE_FRAMES:-3}
    export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=${SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES:--1}
    export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=${SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES:--1}
fi

# Resolve the same defaults used by the concrete launchers so DRY_RUN prints an
# auditable protocol rather than only the downstream command name.
fdvla_apply_protocol_defaults

case "${METHOD}" in
    C-SFT)
        export ONLY_EVAL=true
        export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
        export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
        export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
        export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
        export SEMANTIC_EVAL_FIXED_AGE_FRAMES=-1
        command=(bash "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh")
        ;;
    D-SFT)
        export ONLY_EVAL=true
        command=(bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh")
        ;;
    C-PPO)
        export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
        export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
        export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
        export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
        export SEMANTIC_EVAL_FIXED_AGE_FRAMES=-1
        command=(bash "${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh")
        ;;
    D-PPO)
        command=(bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh")
        ;;
    D-PPO-PosthocAug)
        export POSTHOC_SEMANTIC_DELAY_ENABLED=true
        export POSTHOC_SEMANTIC_DELAY_WEIGHT=${POSTHOC_SEMANTIC_DELAY_WEIGHT:-0.01}
        export POSTHOC_ALLOW_HINDSIGHT_COMPLETION=true
        export POSTHOC_REQUIRE_NONBLOCKING_ROLLOUT=true
        export POSTHOC_SEMANTIC_DELAY_BANK_ENABLED=false
        export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
        export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
        export SEMANTIC_ENV_BOUNDARY_PUBLISH=false
        command=(bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh")
        ;;
    D-PPO-PosthocControl)
        # Replay-forward zero-weight control for PosthocAug. Both arms use only
        # rollout-buffer tensors; neither requests semantic packets during PPO.
        export POSTHOC_SEMANTIC_DELAY_ENABLED=true
        export POSTHOC_SEMANTIC_DELAY_WEIGHT=0.0
        export POSTHOC_FORCE_REPLAY_FORWARD=true
        export POSTHOC_ALLOW_HINDSIGHT_COMPLETION=true
        export POSTHOC_REQUIRE_NONBLOCKING_ROLLOUT=true
        export POSTHOC_SEMANTIC_DELAY_BANK_ENABLED=false
        export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1
        export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1
        export SEMANTIC_ENV_BOUNDARY_PUBLISH=false
        command=(bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh")
        ;;
    D-PPO-Fresh)
        export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=0
        export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=0
        export SEMANTIC_EVAL_FIXED_AGE_FRAMES=0
        export SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES=-1
        export SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1
        export SEMANTIC_FETCH_TARGET_AGE_FRAMES=0
        export SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES=0
        command=(bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh")
        ;;
    D-PPO-NoAge)
        export SEMANTIC_ZERO_PACKET_AGE=true
        command=(bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh")
        ;;
    D-PPO-NoHistory)
        export SEMANTIC_ZERO_ACTION_HISTORY=true
        command=(bash "${SCRIPT_DIR}/run_fdvla_binary_ppo.sh")
        ;;
    *)
        echo "Unknown METHOD=${METHOD}" >&2
        exit 2
        ;;
esac

if [[ "${STAGE}" == smoke && "${METHOD}" == C-* ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export ACTOR_GPU_PLACEMENT=${ACTOR_GPU_PLACEMENT:-0}
    export ROLLOUT_GPU_PLACEMENT=${ROLLOUT_GPU_PLACEMENT:-0}
    export ENV_GPU_PLACEMENT=${ENV_GPU_PLACEMENT:-0}
fi

printf 'METHOD=%s STAGE=%s command=' "${METHOD}" "${STAGE}"
printf '%q ' "${command[@]}"
printf '\n'
printf 'PROTOCOL only_eval=%s train_random_age=%s:%s eval_fixed_age=%s eval_random_age=%s:%s fetch_target_age=%s fetch_hard_max_age=%s zero_age_input=%s zero_history_input=%s posthoc_enabled=%s posthoc_weight=%s posthoc_force_replay=%s posthoc_hindsight=%s posthoc_bank=%s posthoc_nonblocking=%s boundary_publish=%s\n' \
    "${ONLY_EVAL:-false}" \
    "${SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES}" \
    "${SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES}" \
    "${SEMANTIC_EVAL_FIXED_AGE_FRAMES:--1}" \
    "${SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES}" \
    "${SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES}" \
    "${SEMANTIC_FETCH_TARGET_AGE_FRAMES}" \
    "${SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES}" \
    "${SEMANTIC_ZERO_PACKET_AGE:-false}" \
    "${SEMANTIC_ZERO_ACTION_HISTORY:-false}" \
    "${POSTHOC_SEMANTIC_DELAY_ENABLED:-false}" \
    "${POSTHOC_SEMANTIC_DELAY_WEIGHT:-0.0}" \
    "${POSTHOC_FORCE_REPLAY_FORWARD:-false}" \
    "${POSTHOC_ALLOW_HINDSIGHT_COMPLETION:-false}" \
    "${POSTHOC_SEMANTIC_DELAY_BANK_ENABLED:-false}" \
    "${POSTHOC_REQUIRE_NONBLOCKING_ROLLOUT:-true}" \
    "${SEMANTIC_ENV_BOUNDARY_PUBLISH:-false}"
if [[ "${DRY_RUN}" == true ]]; then
    exit 0
fi
exec "${command[@]}"
