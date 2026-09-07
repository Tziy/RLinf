import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from examples.embodiment.train_embodied_agent import _should_launch_reward_worker
from rlinf.algorithms.advantages import select_gae_values_for_critic_warmup
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    override_optimizer_lrs_after_resume,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples" / "embodiment" / "config"


def _load(name: str):
    return OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / name), resolve=False)


def _pop_path(tree: dict, path: str):
    keys = path.split(".")
    node = tree
    for key in keys[:-1]:
        node = node[key]
    return node.pop(keys[-1])


def test_resume_lr_override_updates_optimizer_and_scheduler():
    actor_param = torch.nn.Parameter(torch.ones(1))
    value_param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [actor_param], "lr": 1.0e-6},
            {"params": [value_param], "lr": 1.0e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda _: 1.0, lambda _: 1.0]
    )

    override_optimizer_lrs_after_resume(optimizer, scheduler, [5.0e-7, 2.0e-5])

    assert [group["lr"] for group in optimizer.param_groups] == [5.0e-7, 2.0e-5]
    assert scheduler.base_lrs == [5.0e-7, 2.0e-5]
    assert scheduler.get_last_lr() == [5.0e-7, 2.0e-5]

    actor_param.grad = torch.zeros_like(actor_param)
    value_param.grad = torch.zeros_like(value_param)
    optimizer.step()
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == [5.0e-7, 2.0e-5]


def test_resume_lr_override_rejects_group_count_mismatch():
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{"params": [param], "lr": 1.0e-6}])

    with pytest.raises(ValueError, match="parameter-group count"):
        override_optimizer_lrs_after_resume(optimizer, None, [5.0e-7, 1.0e-4])


def test_critic_warmup_does_not_bootstrap_gae_from_random_value_head():
    values = torch.randn(5, 2)

    assert select_gae_values_for_critic_warmup(values, critic_warmup=True) is None
    assert select_gae_values_for_critic_warmup(values, critic_warmup=False) is values


def test_binary_environment_contract():
    cfg = _load("env/libero_10_binary.yaml")

    assert cfg["use_rel_reward"] is False
    assert cfg["reward_coef"] == 1.0
    assert cfg["use_step_penalty"] is False
    assert cfg["ignore_terminations"] is False


def test_fdvla_configs_keep_native_ppo_and_disable_reward_worker():
    for name in (
        "libero_10_fdvla_binary_ppo.yaml",
        "libero_10_coupled_n1d7_binary_ppo.yaml",
    ):
        cfg = _load(name)
        assert cfg["algorithm"]["loss_type"] == "actor_critic"
        assert cfg["algorithm"]["kl_beta"] == ("${oc.decode:${oc.env:PPO_KL_BETA,0.0}}")
        assert cfg["actor"]["model"]["rl_head_config"][
            "posthoc_semantic_delay_bank_enabled"
        ].endswith("POSTHOC_SEMANTIC_DELAY_BANK_ENABLED,false}}")
        assert cfg["algorithm"]["adv_type"] == "gae"
        assert cfg["algorithm"]["group_size"] == 1
        assert cfg["reward"]["use_reward_model"] is False
        assert _should_launch_reward_worker(cfg["reward"]) is False
        assert cfg["defaults"][0] == "env/libero_10_binary@env.train"
        assert cfg["defaults"][1] == "env/libero_10_binary@env.eval"
        head = cfg["actor"]["model"]["rl_head_config"]
        assert head["dit_train_last_n_blocks"] == (
            "${oc.decode:${oc.env:DIT_TRAIN_LAST_N_BLOCKS,0}}"
        )
        assert "TRAIN_EXECUTION_HORIZON" in head["train_execution_horizon"]
        assert "EVAL_EXECUTION_HORIZON" in head["eval_execution_horizon"]
        assert "EVAL_NOISE_SEED" in head["eval_noise_seed"]
        assert head["semantic_feature_tokens"].endswith("SEMANTIC_FEATURE_TOKENS,160}}")
        assert head["padding_value"].endswith("SEMANTIC_TEXT_PADDING_TOKENS,160}}")
        assert cfg["actor"]["model"]["num_action_chunks"].endswith(",16}}")


def test_actor_master_precision_is_independent_from_rollout_precision():
    for name in (
        "libero_10_fdvla_binary_ppo.yaml",
        "libero_10_coupled_n1d7_binary_ppo.yaml",
    ):
        cfg = _load(name)
        assert cfg["actor"]["model"]["precision"] == (
            "${oc.env:FDVLA_ACTOR_MODEL_PRECISION,bf16}"
        )
        assert cfg["rollout"]["model"]["precision"] == (
            "${oc.env:FDVLA_ROLLOUT_MODEL_PRECISION,bf16}"
        )


def test_coupled_and_decoupled_configs_only_have_registered_differences():
    decoupled = deepcopy(_load("libero_10_fdvla_binary_ppo.yaml"))
    coupled = deepcopy(_load("libero_10_coupled_n1d7_binary_ppo.yaml"))

    different_paths = (
        "cluster.component_placement.actor",
        "cluster.component_placement.rollout",
        "cluster.component_placement.env",
        "runner.logger.experiment_name",
        "actor.model.rl_head_config.execution_mode",
        "actor.model.rl_head_config.semantic_server_enabled",
        "actor.model.rl_head_config.semantic_server_central_cache",
        "actor.model.rl_head_config.semantic_server_boundary_publish",
        "actor.model.rl_head_config.semantic_env_bootstrap_publish",
        "actor.model.rl_head_config.semantic_env_boundary_publish",
        "actor.model.rl_head_config.semantic_mid_chunk_publish",
        "actor.model.rl_head_config.semantic_train_random_age_min_frames",
        "actor.model.rl_head_config.semantic_train_random_age_max_frames",
        "actor.model.rl_head_config.semantic_eval_random_age_min_frames",
        "actor.model.rl_head_config.semantic_eval_random_age_max_frames",
        "actor.model.rl_head_config.verify_semantic_replay",
        "actor.model.rl_head_config.require_packet_age_input",
        "actor.model.rl_head_config.initialize_packet_age_adapter",
        "actor.model.rl_head_config.initialize_action_history_length",
        "actor.model.rl_head_config.zero_init_new_delay_adapters",
        "actor.model.rl_head_config.trainable_modules",
        "actor.model.rl_head_config.drop_local_backbone",
    )
    differences = {
        path: (_pop_path(decoupled, path), _pop_path(coupled, path))
        for path in different_paths
    }

    assert decoupled == coupled
    assert differences["actor.model.rl_head_config.execution_mode"] == (
        "decoupled",
        "coupled",
    )
    assert differences["actor.model.rl_head_config.semantic_server_enabled"] == (
        True,
        False,
    )
    for path in (
        "semantic_server_central_cache",
        "semantic_server_boundary_publish",
        "semantic_env_bootstrap_publish",
        "semantic_env_boundary_publish",
        "semantic_mid_chunk_publish",
    ):
        assert differences[f"actor.model.rl_head_config.{path}"][1] is False

    assert differences["actor.model.rl_head_config.drop_local_backbone"] == (
        True,
        False,
    )
    assert differences["actor.model.rl_head_config.verify_semantic_replay"] == (
        True,
        False,
    )
    for path in (
        "semantic_train_random_age_min_frames",
        "semantic_train_random_age_max_frames",
        "semantic_eval_random_age_min_frames",
        "semantic_eval_random_age_max_frames",
    ):
        decoupled_value, coupled_value = differences[
            f"actor.model.rl_head_config.{path}"
        ]
        assert "SEMANTIC_" in decoupled_value
        assert coupled_value.endswith(",-1}}")


def test_coupled_keeps_vlm_forward_but_uses_dit_only_train_allowlist():
    coupled = _load("libero_10_coupled_n1d7_binary_ppo.yaml")
    head = coupled["actor"]["model"]["rl_head_config"]

    assert head["execution_mode"] == "coupled"
    assert head["drop_local_backbone"] is False
    assert head["dit_only_train"] is True
    assert head["require_frozen_vlm"] is True
    assert head["semantic_server_enabled"] is False
    assert head["semantic_server_central_cache"] is False
    assert head["semantic_env_bootstrap_publish"] is False
    assert head["semantic_env_boundary_publish"] is False
    assert head["semantic_mid_chunk_publish"] is False
    assert head["semantic_train_random_age_min_frames"].endswith(",-1}}")
    assert head["semantic_train_random_age_max_frames"].endswith(",-1}}")
    assert head["semantic_eval_random_age_min_frames"].endswith(",-1}}")
    assert head["semantic_eval_random_age_max_frames"].endswith(",-1}}")
    assert head["require_packet_age_input"].endswith(",false}}")
    assert head["initialize_packet_age_adapter"].endswith(",false}}")
    assert head["initialize_action_history_length"].endswith(",0}}")
    assert head["zero_init_new_delay_adapters"].endswith(",false}}")
    assert "action_head.packet_age_adapter" not in head["trainable_modules"]
    assert "action_head.action_history_adapter" not in head["trainable_modules"]
    assert head["trainable_modules"] == [
        "action_head.model",
        "action_head.value_head",
    ]
    assert all(
        not prefix.startswith("backbone") for prefix in head["trainable_modules"]
    )


def test_fdvla_launchers_share_one_auditable_run_directory():
    common = (ROOT / "examples/embodiment/fdvla_binary_common.sh").read_text()
    decoupled = (ROOT / "examples/embodiment/run_fdvla_binary_ppo.sh").read_text()
    coupled = (ROOT / "examples/embodiment/run_coupled_n1d7_binary_ppo.sh").read_text()
    semantic = (
        ROOT / "examples/embodiment/run_gr00t_semantic_cache_sync_ppo.sh"
    ).read_text()
    trainer = (ROOT / "examples/embodiment/run_embodiment.sh").read_text()

    assert "fdvla_prepare_run_paths()" in common
    assert "FDVLA_RUN_LOG_DIR" in common
    assert "FDVLA_METADATA_DIR" in common
    assert "fdvla_worktree_sha256()" in common
    assert 'worktree_sha256.txt"' in common
    assert "fdvla_prepare_run_paths" in decoupled
    assert "fdvla_prepare_run_paths" in coupled
    assert "FDVLA_RUN_LOG_DIR" in semantic
    assert "export RLINF_LOG_DIR" in semantic
    assert "RLINF_LOG_DIR" in trainer


def test_formal_pilot_uses_fixed_eval_age_but_keeps_random_training_age():
    matrix = (ROOT / "examples/embodiment/run_fdvla_sim_matrix.sh").read_text()
    common = (ROOT / "examples/embodiment/fdvla_binary_common.sh").read_text()

    assert "SEMANTIC_EVAL_FIXED_AGE_FRAMES:-3" in matrix
    assert "SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES:--1" in matrix
    assert "SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES:-6" in common
    assert "PPO_VAL_INTERVAL:-10" in matrix
    assert "PPO_SAVE_INTERVAL:-10" in matrix


def test_realtime_surface_requires_formal_trial_floor():
    surface = (ROOT / "examples/embodiment/eval_fdvla_realtime_surface.sh").read_text()

    assert "SURFACE_MIN_TRIALS:-400" in surface
    assert "EVAL_ROLLOUT_EPOCH" in surface
    assert "trial_count < SURFACE_MIN_TRIALS" in surface


def test_fresh_manifest_disables_random_eval_age_for_exact_age_zero():
    manifest = _load("fdvla_binary_run_manifest.yaml")
    overrides = manifest["methods"]["D-PPO-Fresh"]["overrides"]

    assert overrides["SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES"] == 0
    assert overrides["SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES"] == 0
    assert overrides["SEMANTIC_EVAL_FIXED_AGE_FRAMES"] == 0
    assert overrides["SEMANTIC_EVAL_RANDOM_AGE_MIN_FRAMES"] == -1
    assert overrides["SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES"] == -1
    assert overrides["SEMANTIC_FETCH_TARGET_AGE_FRAMES"] == 0
    assert overrides["SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES"] == 0


def test_initial_checkpoint_screen_is_preregistered_and_disjoint_from_formal_noise():
    manifest = _load("fdvla_binary_run_manifest.yaml")
    screen = manifest["development_initial_checkpoint_screen"]

    assert screen["formal_result"] is False
    assert list(screen["candidates"]) == [
        "weak1000",
        "weak1500",
        "weak1000plus500",
    ]
    assert "weak2000" in screen["excluded_candidates"]
    assert screen["paired_condition"] == {
        "task_id": 0,
        "semantic_age_frames": 6,
        "action_execution_horizon": 8,
        "action_prediction_horizon": 16,
        "policy_noise_seed": 12026,
        "unique_trials": 50,
    }
    assert screen["selection_rule"]["target_success_rate"] == 0.45
    assert screen["selection_rule"]["preferred_interval"] == [0.30, 0.60]
    assert screen["selection_rule"]["tie_break_order"] == [
        "weak1500",
        "weak1000plus500",
        "weak1000",
    ]
    assert screen["isolation"] == {
        "selection_policy_noise_seed_must_not_enter_formal_eval": True,
        "formal_eval_policy_noise_seeds": list(range(2026, 2034)),
        "selection_results_must_be_labeled_development": True,
        "selected_checkpoint_must_be_locked_for_all_methods": True,
    }


def test_target80_development_early_stop_is_preregistered_and_held_out():
    manifest = _load("fdvla_binary_run_manifest.yaml")
    protocol = manifest["development_success_early_stop_preregistered"]

    assert protocol["formal_result"] is False
    assert protocol["parent_checkpoint"]["global_step"] == 50
    assert protocol["selection_condition"] == {
        "task_id": 0,
        "semantic_age_frames": 6,
        "action_execution_horizon": 8,
        "action_prediction_horizon": 16,
        "policy_noise_seed": 13026,
        "unique_trials": 48,
    }
    assert protocol["stopping_rule"] == {
        "metric": "eval/success_once",
        "first_evaluation_at_or_above": 0.75,
        "desired_reporting_band": [0.75, 0.85],
        "evaluation_interval_updates": 10,
        "maximum_global_step": 150,
        "overshoot_policy": (
            "stop_at_first_crossing_and_report_overshoot_without_rewinding"
        ),
        "no_hit_policy": ("stop_at_maximum_global_step_and_report_target_not_reached"),
    }
    assert protocol["held_out_final_condition"]["policy_noise_seed"] == 13027
    assert protocol["held_out_final_condition"]["checkpoint_selection_forbidden"]
    assert protocol["isolation"]["response_surface_noise_seed"] == 14026


def test_target80_runner_matches_preregistered_protocol():
    manifest = _load("fdvla_binary_run_manifest.yaml")
    protocol = manifest["development_success_early_stop_preregistered"]
    script = (ROOT / "examples/embodiment/run_fdvla_target80_resume.sh").read_text()

    assert protocol["parent_checkpoint"]["actor_full_weights_sha256"] in script
    assert 'sha256sum "${parent_actor}"' in script
    expected_exports = {
        "PPO_MAX_STEPS": protocol["stopping_rule"]["maximum_global_step"],
        "PPO_VAL_INTERVAL": protocol["stopping_rule"]["evaluation_interval_updates"],
        "PPO_SUCCESS_EARLY_STOP_THRESHOLD": protocol["stopping_rule"][
            "first_evaluation_at_or_above"
        ],
        "PPO_SUCCESS_EARLY_STOP_BAND_LOW": protocol["stopping_rule"][
            "desired_reporting_band"
        ][0],
        "PPO_SUCCESS_EARLY_STOP_BAND_HIGH": protocol["stopping_rule"][
            "desired_reporting_band"
        ][1],
        "EVAL_NOISE_SEED": protocol["selection_condition"]["policy_noise_seed"],
        "SEMANTIC_EVAL_FIXED_AGE_FRAMES": protocol["selection_condition"][
            "semantic_age_frames"
        ],
        "EVAL_EXECUTION_HORIZON": protocol["selection_condition"][
            "action_execution_horizon"
        ],
        "EVAL_NUM_ENVS": protocol["selection_condition"]["unique_trials"],
    }
    for name, value in expected_exports.items():
        assert f"export {name}={value}" in script
    assert "actor4_rollout3)" in script
    assert "colocated4)" in script
    assert "export ALLOWED_GPU_IDS=0,1,2,3" in script


def test_target80_heldout_runner_forbids_arbitrary_checkpoint():
    manifest = _load("fdvla_binary_run_manifest.yaml")
    condition = manifest["development_success_early_stop_preregistered"][
        "held_out_final_condition"
    ]
    script = (ROOT / "examples/embodiment/eval_fdvla_target80_heldout.sh").read_text()

    assert condition["fresh_process"] is True
    assert condition["checkpoint_selection_forbidden"] is True
    assert "unset PPO_CKPT_PATH" in script
    assert "development_success_early_stop.json" in script
    assert 'glob("**/checkpoints/global_step_150")' in script
    assert (
        f"export SURFACE_POLICY_NOISE_SEEDS={condition['policy_noise_seed']}" in script
    )
    assert f"export SEMANTIC_AGES={condition['semantic_age_frames']}" in script
    assert f"export ACTION_HORIZONS={condition['action_execution_horizon']}" in script
    assert f"export SURFACE_MIN_TRIALS={condition['unique_trials']}" in script
    assert "export RESUME_SURFACE=false" in script
    assert (
        'POLICY_METHOD=D-PPO bash "${SCRIPT_DIR}/eval_fdvla_realtime_surface.sh"'
        in script
    )


def test_binary_configs_expose_disabled_by_default_success_early_stop():
    for name in (
        "libero_10_fdvla_binary_ppo.yaml",
        "libero_10_coupled_n1d7_binary_ppo.yaml",
    ):
        cfg = _load(name)
        early_stop = cfg["runner"]["success_early_stop"]
        assert early_stop["enabled"].endswith(",false}}")
        assert early_stop["metric"].endswith(",eval/success_once}")
        assert early_stop["threshold"].endswith(",0.75}}")
        assert early_stop["desired_band_low"].endswith(",0.75}}")
        assert early_stop["desired_band_high"].endswith(",0.85}}")


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (
            "C-SFT",
            "only_eval=true train_random_age=-1:-1 eval_fixed_age=-1 "
            "eval_random_age=-1:-1",
        ),
        (
            "D-SFT",
            "only_eval=true train_random_age=0:6 eval_fixed_age=-1 eval_random_age=0:6",
        ),
        (
            "C-PPO",
            "only_eval=false train_random_age=-1:-1 eval_fixed_age=-1 "
            "eval_random_age=-1:-1",
        ),
        (
            "D-PPO",
            "only_eval=false train_random_age=0:6 eval_fixed_age=-1 "
            "eval_random_age=0:6",
        ),
        (
            "D-PPO-PosthocAug",
            "train_random_age=-1:-1 eval_fixed_age=-1 eval_random_age=0:6 "
            "fetch_target_age=-1 fetch_hard_max_age=-1 zero_age_input=false "
            "zero_history_input=false posthoc_enabled=true posthoc_weight=0.01 "
            "posthoc_force_replay=false posthoc_hindsight=true "
            "posthoc_bank=false posthoc_nonblocking=true boundary_publish=false",
        ),
        (
            "D-PPO-PosthocControl",
            "train_random_age=-1:-1 eval_fixed_age=-1 eval_random_age=0:6 "
            "fetch_target_age=-1 fetch_hard_max_age=-1 zero_age_input=false "
            "zero_history_input=false posthoc_enabled=true posthoc_weight=0.0 "
            "posthoc_force_replay=true posthoc_hindsight=true "
            "posthoc_bank=false posthoc_nonblocking=true boundary_publish=false",
        ),
        (
            "D-PPO-Fresh",
            "train_random_age=0:0 eval_fixed_age=0 eval_random_age=-1:-1 "
            "fetch_target_age=0 fetch_hard_max_age=0",
        ),
        ("D-PPO-NoAge", "zero_age_input=true zero_history_input=false"),
        ("D-PPO-NoHistory", "zero_age_input=false zero_history_input=true"),
    ],
)
def test_eight_method_dry_run_resolves_auditable_protocol(method, expected):
    script = ROOT / "examples/embodiment/run_fdvla_sim_matrix.sh"
    cleared = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SEMANTIC_", "POSTHOC_"))
        and key not in {"ONLY_EVAL", "METHOD", "STAGE", "DRY_RUN"}
    }
    result = subprocess.run(
        ["bash", str(script)],
        env={
            **cleared,
            "DRY_RUN": "true",
            "METHOD": method,
            "STAGE": "smoke",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_posthoc_augmentation_config_and_manifest_keep_ppo_rows_isolated():
    for name in (
        "libero_10_fdvla_binary_ppo.yaml",
        "libero_10_coupled_n1d7_binary_ppo.yaml",
    ):
        cfg = _load(name)
        posthoc = cfg["algorithm"]["posthoc_semantic_delay_augmentation"]
        assert posthoc["enabled"].endswith(",false}}")
        assert posthoc["delta_frames"] == [-4, -2, 2, 4, 8]
        assert posthoc["consistency_loss_weight"].endswith(",0.0}}")
        assert posthoc["allow_hindsight_completion"].endswith(",false}}")
        assert posthoc["force_replay_forward"].endswith(",false}}")
        assert posthoc["require_nonblocking_rollout"].endswith(",true}}")
        assert posthoc["replay_microbatch_interval"].endswith(",1}}")
        assert posthoc["replay_microbatch_offset"].endswith(",0}}")
        assert cfg["algorithm"]["loss_type"] == "actor_critic"
        assert cfg["actor"]["model"]["rl_head_config"][
            "posthoc_semantic_delay_bank_enabled"
        ].endswith("POSTHOC_SEMANTIC_DELAY_BANK_ENABLED,false}}")

    manifest = _load("fdvla_binary_run_manifest.yaml")
    method = manifest["methods"]["D-PPO-PosthocAug"]
    control = manifest["methods"]["D-PPO-PosthocControl"]
    contract = method["augmentation_contract"]
    assert method["overrides"]["POSTHOC_SEMANTIC_DELAY_ENABLED"] is True
    assert method["overrides"]["POSTHOC_ALLOW_HINDSIGHT_COMPLETION"] is True
    assert method["semantic_mode"] == "natural_async_latest"
    assert method["overrides"]["SEMANTIC_ENV_BOUNDARY_PUBLISH"] is False
    assert method["overrides"]["SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES"] == -1
    assert method["overrides"]["SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES"] == -1
    assert method["overrides"]["POSTHOC_SEMANTIC_DELAY_BANK_ENABLED"] is False
    assert method["overrides"]["POSTHOC_REQUIRE_NONBLOCKING_ROLLOUT"] is True
    assert control["overrides"] == {
        "POSTHOC_SEMANTIC_DELAY_ENABLED": True,
        "POSTHOC_SEMANTIC_DELAY_WEIGHT": 0.0,
        "POSTHOC_FORCE_REPLAY_FORWARD": True,
        "POSTHOC_ALLOW_HINDSIGHT_COMPLETION": True,
        "POSTHOC_REQUIRE_NONBLOCKING_ROLLOUT": True,
        "POSTHOC_SEMANTIC_DELAY_BANK_ENABLED": False,
        "SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES": -1,
        "SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES": -1,
        "SEMANTIC_ENV_BOUNDARY_PUBLISH": False,
    }
    assert contract["hindsight_completion_explicitly_labeled"] is True
    assert contract["hindsight_rows_excluded_from_main_ppo"] is True
    assert contract["ppo_eligible"] is False
    assert contract["old_logprob_reuse"] == "forbidden"
    assert contract["semantic_server_or_vlm_call_during_update"] == "forbidden"
    assert contract["missing_candidate_behavior"] == "skip"
    assert contract["rollout_semantic_source"] == "latest_completed_packet"
    assert contract["augmentation_source"] == "rollout_buffer_only"
    assert contract["actual_age_recomputed_from_frame_ids"] is True


def test_formal_task0_bf16_tail4_protocol_is_locked_and_disjoint():
    protocol = _load("fdvla_binary_run_manifest.yaml")[
        "formal_task0_bf16_tail4_protocol"
    ]

    assert protocol["status"] == "registered_before_formal_runs"
    assert protocol["matched_start_selection"]["semantic_age_frames"] == 0
    assert protocol["matched_start_selection"]["action_execution_horizon"] == 2
    assert protocol["matched_start_selection"]["selection_policy_noise_seed"] == 14026
    assert protocol["training"]["training_seeds"] == [0, 1, 2]
    assert protocol["training"]["global_batch_size"] == 960
    assert protocol["training"]["actor_precision"] == "bf16"
    assert protocol["training"]["dit_train_last_n_blocks"] == 4
    assert protocol["training"]["c_trainable_scope"] == [
        "dit_last_4_blocks",
        "value_head",
    ]
    assert "packet_age_adapter" in protocol["training"]["d_trainable_scope"]
    assert protocol["matched_start_selection"]["candidate_choice_rule"].startswith(
        "eligible_joint_pair"
    )
    assert protocol["timing_selection"]["status"] == "registered_before_batching_runs"
    assert [row["id"] for row in protocol["timing_selection"]["candidates"]] == [
        "batch1",
        "batch2",
        "batch4",
    ]
    assert protocol["learning_curve_evaluation"]["policy_noise_seed"] == 27026
    assert protocol["final_evaluation"]["trials_per_training_seed_and_condition"] == 400
    assert 14026 not in protocol["final_evaluation"]["policy_noise_seeds"]
    assert 27026 not in protocol["final_evaluation"]["policy_noise_seeds"]


def test_matched_start_selection_accepts_arm_specific_checkpoint_steps():
    script = (
        ROOT / "examples/embodiment/run_fdvla_matched_start_selection.sh"
    ).read_text()

    assert "FDVLA_MATCHED_${arm}_SELECTION_STEPS" in script
    assert 'arm_steps="${!arm_steps_name:-${SELECTION_STEPS}}"' in script
    assert 'for step in "${candidate_steps[@]}"' in script
    assert "assert_selection_worktree_frozen complete" in script
    assert "assert_candidate_provenance" in script
    assert "selection_worktree_sha256.txt" in script
    assert "FDVLA_MATCHED_BATCHING_ENV" in script
    assert "selected_batching_manifest.env" in script
    assert r"checkpoint\tcheckpoint_sha256\tjsonl" in script

    assert "FDVLA_MATCHED_C_GRID_ROOT" in script
    assert "FDVLA_MATCHED_D_GRID_ROOT" in script
    assert "requested_ages != {expected_age}" in script
    assert "horizons != {expected_horizon}" in script
    assert "nonbootstrap_mismatches" in script
    assert "select_fdvla_paired_starts.py" in script
    assert "--max-deviation" in script
    assert "checkpoint_sha256" in script
    assert "semantic_requested_actual_age_mismatch_boundary_count" in script
    assert "default_require_packet_age=false" in script
    assert "default_require_packet_age=true" in script


def test_matched_formal_protocol_locks_bf16_tail4_k2_and_provenance():
    compare = (
        ROOT / "examples/embodiment/run_fdvla_matched_start_ppo_compare.sh"
    ).read_text()
    formal = (
        ROOT / "examples/embodiment/run_fdvla_matched_start_formal_3seed.sh"
    ).read_text()

    for contract in (
        "FDVLA_MATCHED_PPO_EXECUTION_HORIZON:-2",
        "FDVLA_MATCHED_PPO_GLOBAL_BATCH_SIZE:-960",
        "FDVLA_MATCHED_PPO_KL_BETA:-0.0",
        "FDVLA_MATCHED_BATCHING_ENV",
        "semantic_batch_max_requests",
        "current_checkpoint_sha=$(fdvla_checkpoint_sha256",
        "REQUIRE_PACKET_AGE_INPUT=false",
        "REQUIRE_PACKET_AGE_INPUT=true",
        "ACTION_HISTORY_LENGTH=0",
        "ACTION_HISTORY_LENGTH=4",
        "FDVLA_MATCHED_ACTOR_MODEL_PRECISION:-bf16",
        "FDVLA_MATCHED_ROLLOUT_MODEL_PRECISION:-bf16",
        "FDVLA_MATCHED_DIT_TRAIN_LAST_N_BLOCKS:-4",
        "FDVLA_MATCHED_SEMANTIC_PUBLISH_INTERVAL_FRAMES:-8",
        "assert_selection_provenance C",
        "assert_selection_provenance D",
        "assert_pair_worktree_frozen complete",
    ):
        assert contract in compare

    for contract in (
        "FDVLA_FORMAL_EXECUTION_HORIZON:-2",
        "FDVLA_FORMAL_PPO_GLOBAL_BATCH_SIZE:-960",
        "FDVLA_FORMAL_PPO_KL_BETA:-0.0",
        "FDVLA_FORMAL_BATCHING_ENV",
        "selected_batching_manifest.env",
        "FDVLA_FORMAL_ACTOR_MODEL_PRECISION:-bf16",
        "FDVLA_FORMAL_ROLLOUT_MODEL_PRECISION:-bf16",
        "FDVLA_FORMAL_DIT_TRAIN_LAST_N_BLOCKS:-4",
        "FDVLA_FORMAL_SEMANTIC_PUBLISH_INTERVAL_FRAMES:-8",
    ):
        assert contract in formal


def test_delay_sweep_isolates_each_condition_run_directory():
    script = (ROOT / "examples/embodiment/eval_fdvla_delay_sweep.sh").read_text()

    assert (
        'local run_dir="${FDVLA_DELAY_SWEEP_ROOT}/${POLICY_METHOD}/${label}"' in script
    )
    assert 'FDVLA_RUN_LOG_DIR="${run_dir}"' in script
    assert 'RLINF_LOG_DIR="${run_dir}"' in script
    assert 'FDVLA_METADATA_DIR="${run_dir}/fdvla_metadata"' in script
    assert 'FDVLA_RUN_TIMESTAMP="${SWEEP_ID}_${POLICY_METHOD}_${label}"' in script


def test_posthoc_paired_pilot_forces_latest_only_training():
    script = (
        ROOT / "examples/embodiment/run_fdvla_posthoc_paired_pilot.sh"
    ).read_text()

    assert "export SEMANTIC_TRAIN_RANDOM_AGE_MIN_FRAMES=-1" in script
    assert "export SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1" in script
    assert "export POSTHOC_REQUIRE_NONBLOCKING_ROLLOUT=true" in script
    assert "export POSTHOC_SEMANTIC_DELAY_BANK_ENABLED=false" in script
    assert "export SEMANTIC_ENV_BOUNDARY_PUBLISH=false" in script
    assert "PPO_GLOBAL_BATCH_SIZE:-960" in script
    assert "PPO_ACTOR_LR:-5.0e-7" in script
    assert "PPO_KL_BETA:-0.0" in script
    assert "FDVLA_ACTOR_MODEL_PRECISION:-bf16" in script
    assert "FDVLA_ROLLOUT_MODEL_PRECISION:-bf16" in script
    assert "DIT_TRAIN_LAST_N_BLOCKS:-4" in script
    assert "TRAIN_EXECUTION_HORIZON:-2" in script
    assert "EVAL_EXECUTION_HORIZON:-2" in script
    assert "SEMANTIC_PUBLISH_INTERVAL_FRAMES:-8" in script


def test_batching_selection_launcher_locks_natural_latest_protocol():
    script = (ROOT / "examples/embodiment/run_fdvla_batching_selection.sh").read_text()

    for contract in (
        "run_candidate batch1 1 1 0",
        "run_candidate batch2 2 2 1",
        "run_candidate batch4 4 4 2",
        "SEMANTIC_TRAIN_RANDOM_AGE_MAX_FRAMES=-1",
        "SEMANTIC_EVAL_RANDOM_AGE_MAX_FRAMES=-1",
        "POSTHOC_SEMANTIC_DELAY_ENABLED=false",
        "SEMANTIC_ENV_BOUNDARY_PUBLISH=false",
        "PPO_MAX_STEPS=6",
        "final_eval_enclosing_update_excluded=1",
        "stable_updates_required=3",
        "TRAIN_EXECUTION_HORIZON=2",
        "--candidate-set batching",
        "--min-stable-samples 3",
        "selected_batching.env",
    ):
        assert contract in script
