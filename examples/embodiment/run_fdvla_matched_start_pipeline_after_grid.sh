#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GRID_SESSION=${FDVLA_MATCHED_GRID_SESSION:?Set FDVLA_MATCHED_GRID_SESSION}
GRID_ROOT=${FDVLA_MATCHED_SFT_RUN_ROOT:?Set FDVLA_MATCHED_SFT_RUN_ROOT}
SELECTION_ROOT=${FDVLA_MATCHED_SELECTION_ROOT:-$(dirname "${GRID_ROOT}")/selection}
PPO_ROOT=${FDVLA_MATCHED_PPO_RUN_ROOT:-$(dirname "${GRID_ROOT}")/ppo_compare}

while tmux has-session -t "${GRID_SESSION}" 2>/dev/null; do
    sleep 30
done

for arm_and_experiment in \
    'C/libero10_coupled_current_s4_a4_grid500' \
    'D/libero10_decoupled_delay0to8_s4_a4_grid500'; do
    for step in 100 200 300 400 500; do
        checkpoint=${GRID_ROOT}/${arm_and_experiment}/checkpoint-${step}
        if [[ ! -f "${checkpoint}/model.safetensors.index.json" ]]; then
            echo "Grid training ended without candidate checkpoint: ${checkpoint}" >&2
            exit 1
        fi
    done
done

FDVLA_MATCHED_SFT_RUN_ROOT="${GRID_ROOT}" \
FDVLA_MATCHED_SELECTION_ROOT="${SELECTION_ROOT}" \
    bash "${SCRIPT_DIR}/run_fdvla_matched_start_selection.sh"

FDVLA_MATCHED_SELECTED_ENV="${SELECTION_ROOT}/selected.env" \
FDVLA_MATCHED_PPO_RUN_ROOT="${PPO_ROOT}" \
    bash "${SCRIPT_DIR}/run_fdvla_matched_start_ppo_compare.sh"

printf '%s\n' "${PPO_ROOT}"
