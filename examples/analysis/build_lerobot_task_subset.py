#!/usr/bin/env python3
"""Build a reproducible single-task subset of a LeRobot v2 dataset.

Parquet metadata is renumbered to a contiguous episode/index space. Videos are
symlinked read-only by default, while the source dataset's normalization stats
are intentionally retained so that checkpoints trained on the full dataset use
the same action/state scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

META_FILES_WITH_SOURCE_STATS = ("modality.json", "stats.json", "relative_stats.json")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_stat(stat: dict, value: int) -> None:
    stat["min"] = [value]
    stat["max"] = [value]
    stat["mean"] = [float(value)]
    stat["std"] = [0.0]


def build_subset(source: Path, target: Path, task_index: int, copy_videos: bool) -> None:
    source = source.resolve()
    target = target.absolute()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {target}")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")

    info = json.loads((source / "meta" / "info.json").read_text())
    if str(info.get("codebase_version")) != "v2.1":
        raise ValueError("Only per-episode LeRobot v2.1 datasets are supported")
    tasks = read_jsonl(source / "meta" / "tasks.jsonl")
    task_by_index = {int(row["task_index"]): row["task"] for row in tasks}
    if task_index not in task_by_index:
        raise ValueError(f"Unknown task_index {task_index}; available={sorted(task_by_index)}")
    task_text = task_by_index[task_index]

    episodes = read_jsonl(source / "meta" / "episodes.jsonl")
    stats_rows = read_jsonl(source / "meta" / "episodes_stats.jsonl")
    stats_by_episode = {int(row["episode_index"]): row for row in stats_rows}
    selected = [row for row in episodes if task_text in row.get("tasks", [])]
    if not selected:
        raise ValueError(f"No episodes found for task_index={task_index}: {task_text!r}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{target.name}.tmp-", dir=target.parent))
    try:
        (temporary / "meta").mkdir()
        (temporary / "data" / "chunk-000").mkdir(parents=True)
        video_keys = sorted(info.get("features", {}))
        video_keys = [key for key in video_keys if info["features"][key].get("dtype") == "video"]
        for key in video_keys:
            (temporary / "videos" / "chunk-000" / key).mkdir(parents=True)

        output_episodes: list[dict] = []
        output_stats: list[dict] = []
        source_episode_indices: list[int] = []
        global_index = 0
        for new_episode, episode in enumerate(selected):
            old_episode = int(episode["episode_index"])
            source_episode_indices.append(old_episode)
            parquet_source = source / info["data_path"].format(
                episode_chunk=old_episode // int(info["chunks_size"]),
                episode_index=old_episode,
            )
            frame = pd.read_parquet(parquet_source)
            if set(frame["task_index"].unique()) != {task_index}:
                raise ValueError(f"Episode {old_episode} contains unexpected task indices")
            frame["episode_index"] = new_episode
            frame["task_index"] = 0
            frame["index"] = range(global_index, global_index + len(frame))
            parquet_target = temporary / "data" / "chunk-000" / f"episode_{new_episode:06d}.parquet"
            frame.to_parquet(parquet_target, index=False)

            output_episodes.append(
                {"episode_index": new_episode, "tasks": [task_text], "length": len(frame)}
            )
            stat_row = json.loads(json.dumps(stats_by_episode[old_episode]))
            stat_row["episode_index"] = new_episode
            remap_stat(stat_row["stats"]["episode_index"], new_episode)
            remap_stat(stat_row["stats"]["task_index"], 0)
            index_stat = stat_row["stats"]["index"]
            index_stat["min"] = [global_index]
            index_stat["max"] = [global_index + len(frame) - 1]
            index_stat["mean"] = [global_index + (len(frame) - 1) / 2]
            output_stats.append(stat_row)

            for video_key in video_keys:
                video_source = source / info["video_path"].format(
                    episode_chunk=old_episode // int(info["chunks_size"]),
                    episode_index=old_episode,
                    video_key=video_key,
                )
                video_target = (
                    temporary
                    / "videos"
                    / "chunk-000"
                    / video_key
                    / f"episode_{new_episode:06d}.mp4"
                )
                if copy_videos:
                    shutil.copy2(video_source, video_target)
                else:
                    video_target.symlink_to(video_source.resolve())
            global_index += len(frame)

        output_info = json.loads(json.dumps(info))
        output_info.update(
            total_episodes=len(output_episodes),
            total_frames=global_index,
            total_tasks=1,
            total_chunks=1,
            total_videos=len(output_episodes) * len(video_keys),
            splits={"train": f"0:{len(output_episodes)}"},
            data_files=[
                f"data/chunk-000/episode_{index:06d}.parquet"
                for index in range(len(output_episodes))
            ],
        )
        (temporary / "meta" / "info.json").write_text(
            json.dumps(output_info, indent=2, sort_keys=True) + "\n"
        )
        write_jsonl(temporary / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_text}])
        write_jsonl(temporary / "meta" / "episodes.jsonl", output_episodes)
        write_jsonl(temporary / "meta" / "episodes_stats.jsonl", output_stats)
        for name in META_FILES_WITH_SOURCE_STATS:
            shutil.copy2(source / "meta" / name, temporary / "meta" / name)

        manifest = {
            "schema_version": 1,
            "source_dataset": str(source),
            "source_info_sha256": sha256_file(source / "meta" / "info.json"),
            "source_episodes_sha256": sha256_file(source / "meta" / "episodes.jsonl"),
            "task_index_in_source": task_index,
            "task_index_in_subset": 0,
            "task": task_text,
            "source_episode_indices": source_episode_indices,
            "total_episodes": len(output_episodes),
            "total_frames": global_index,
            "normalization_stats": "preserved_from_source_full_dataset",
            "video_storage": "copied" if copy_videos else "absolute_symlink_to_source",
        }
        (temporary / "subset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--copy-videos", action="store_true")
    args = parser.parse_args()
    build_subset(args.source, args.target, args.task_index, args.copy_videos)


if __name__ == "__main__":
    main()
