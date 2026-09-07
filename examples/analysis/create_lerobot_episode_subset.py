#!/usr/bin/env python3
"""Create a deterministic, symlink-based LeRobot v2 episode subset.

The source dataset is never modified. Selected episodes are renumbered locally so
the standard GR00T loader can consume a non-contiguous source selection without a
training-path code change. Dataset statistics are intentionally not copied: the
normal GR00T startup recomputes them from the selected supervision episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(pattern: str, episode_index: int, video_key: str | None = None) -> Path:
    values = {
        "episode_chunk": episode_index // 1000,
        "episode_index": episode_index,
        "video_key": video_key,
    }
    return Path(pattern.format(**values))


def _link(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def create_subset(
    source: Path,
    output: Path,
    task: str,
    num_episodes: int,
    seed: int,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {output}")

    meta = source / "meta"
    info = json.loads((meta / "info.json").read_text())
    if str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError("this utility currently supports per-episode LeRobot v2 paths only")
    episodes = _read_jsonl(meta / "episodes.jsonl")
    matching = [row for row in episodes if task in row.get("tasks", [])]
    if num_episodes < 1 or num_episodes > len(matching):
        raise ValueError(f"requested {num_episodes}; task has {len(matching)} episodes")

    def selection_key(row: dict[str, Any]) -> tuple[str, int]:
        source_index = int(row["episode_index"])
        token = f"{seed}:{source_index}".encode()
        return hashlib.sha256(token).hexdigest(), source_index

    selected = sorted(matching, key=selection_key)[:num_episodes]
    selected = sorted(selected, key=lambda row: int(row["episode_index"]))
    episode_stats_path = meta / "episodes_stats.jsonl"
    stats_by_index = (
        {int(row["episode_index"]): row for row in _read_jsonl(episode_stats_path)}
        if episode_stats_path.exists()
        else {}
    )

    video_keys = sorted(
        key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"
    )
    data_pattern = info["data_path"]
    video_pattern = info.get("video_path")
    output.mkdir(parents=True, exist_ok=True)

    local_episodes = []
    local_stats = []
    manifest_rows = []
    for local_index, row in enumerate(selected):
        source_index = int(row["episode_index"])
        source_data = source / _render(data_pattern, source_index)
        target_data = output / _render(data_pattern, local_index)
        _link(source_data, target_data)

        linked_files = [{"kind": "data", "sha256": _sha256(source_data), "source": str(source_data)}]
        if video_pattern:
            for video_key in video_keys:
                source_video = source / _render(video_pattern, source_index, video_key)
                target_video = output / _render(video_pattern, local_index, video_key)
                _link(source_video, target_video)
                linked_files.append(
                    {"kind": video_key, "sha256": _sha256(source_video), "source": str(source_video)}
                )

        local_row = dict(row)
        local_row["episode_index"] = local_index
        local_episodes.append(local_row)
        if source_index in stats_by_index:
            stat_row = dict(stats_by_index[source_index])
            stat_row["episode_index"] = local_index
            local_stats.append(stat_row)
        manifest_rows.append(
            {
                "files": linked_files,
                "length": int(row["length"]),
                "local_episode_index": local_index,
                "source_episode_index": source_index,
                "task": task,
            }
        )

    local_info = dict(info)
    local_info["total_episodes"] = num_episodes
    local_info["total_frames"] = sum(int(row["length"]) for row in selected)
    local_info["splits"] = {"train": f"0:{num_episodes}"}
    _write_json(output / "meta/info.json", local_info)
    for name in ("modality.json", "tasks.jsonl"):
        source_path = meta / name
        target_path = output / "meta" / name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.symlink_to(source_path.resolve())
    _write_jsonl(output / "meta/episodes.jsonl", local_episodes)
    if local_stats:
        _write_jsonl(output / "meta/episodes_stats.jsonl", local_stats)
    manifest = {
        "normalization": "recomputed_from_subset_by_GR00T_startup",
        "num_episodes": num_episodes,
        "seed": seed,
        "source": str(source.resolve()),
        "task": task,
        "episodes": manifest_rows,
    }
    _write_json(output / "subset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    manifest = create_subset(**vars(args))
    print(json.dumps({k: manifest[k] for k in ("source", "task", "num_episodes", "seed")}, indent=2))


if __name__ == "__main__":
    main()
