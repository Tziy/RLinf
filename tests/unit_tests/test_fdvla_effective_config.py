"""Regression tests for the recorded runtime configuration and exit status."""

import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.utils.fdvla_run_metadata import save_effective_config


def test_runtime_overrides_replace_preflight_and_preserve_original(tmp_path):
    before = "env:\n  train:\n    use_rel_reward: false\n"
    (tmp_path / "resolved_config.yaml").write_text(before)
    cfg = OmegaConf.create(
        {
            "env": {"train": {"use_rel_reward": True}},
            "server_port": 6900,
            "copy": "${server_port}",
        }
    )
    save_effective_config(cfg, tmp_path)
    actual = OmegaConf.load(tmp_path / "resolved_config.yaml")
    assert actual.env.train.use_rel_reward is True
    assert actual["copy"] == 6900
    assert (tmp_path / "preflight_config.yaml").read_text() == before
    save_effective_config(cfg, tmp_path)
    assert (tmp_path / "preflight_config.yaml").read_text() == before


def test_training_error_survives_tee(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\nexit 17\n")
    fake_python.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        RLINF_LOG_DIR=str(tmp_path / "logs"),
    )
    script = (
        Path(__file__).resolve().parents[2] / "examples/embodiment/run_embodiment.sh"
    )
    result = subprocess.run(
        ["bash", str(script), "unused", "LIBERO"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 17
