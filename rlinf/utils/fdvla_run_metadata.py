"""Persist the configuration actually consumed by an FDVLA process."""

import hashlib
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def save_effective_config(cfg: DictConfig, directory: Path) -> None:
    """Save post-validation settings, including CLI overrides and allocated ports.

    Args:
        cfg: The configuration passed to workers after Hydra and validation.
        directory: Run-specific metadata directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "resolved_config.yaml"
    preflight = directory / "preflight_config.yaml"
    if target.exists() and not preflight.exists():
        preflight.write_bytes(target.read_bytes())
    serialized = OmegaConf.to_yaml(cfg, resolve=True)
    temporary = directory / "effective_config.yaml.tmp"
    temporary.write_text(serialized)
    temporary.replace(target)
    (directory / "effective_config_provenance.json").write_text(
        json.dumps(
            {
                "source": "post_hydra_post_validate_before_worker_launch",
                "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
