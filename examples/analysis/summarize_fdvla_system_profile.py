#!/usr/bin/env python3
"""Summarize FDVLA semantic-server JSON/text logs and rollout profile metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

_BATCH_PATTERN = re.compile(
    r"Semantic batch requests=(?P<requests>\d+) envs=(?P<envs>\d+) "
    r"raw_prep_ms=(?P<raw_prep>[0-9.eE+-]+) "
    r"raw_prep_wait_ms=(?P<raw_prep_wait>[0-9.eE+-]+) "
    r"merge_h2d_ms=(?P<merge_h2d>[0-9.eE+-]+) "
    r"forward_ms=(?P<forward>[0-9.eE+-]+) "
    r"queue_age_ms=(?P<queue>[0-9.eE+-]+)"
)


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot summarize an empty profiling stream")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def parse_semantic_server_logs(paths: list[Path]) -> dict:
    records = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            match = _BATCH_PATTERN.search(line)
            if match:
                records.append(
                    {
                        key: int(value) if key in {"requests", "envs"} else float(value)
                        for key, value in match.groupdict().items()
                    }
                )
    if not records:
        raise ValueError("No 'Semantic batch' profiling records were found")
    return {
        "vlm_forward_count": len(records),
        "semantic_rows": sum(record["envs"] for record in records),
        "vlm_forward_latency_ms": _distribution(
            [record["forward"] for record in records]
        ),
        "semantic_queue_latency_ms": _distribution(
            [record["queue"] for record in records]
        ),
        "raw_preprocess_latency_ms": _distribution(
            [record["raw_prep"] for record in records]
        ),
        "raw_preprocess_wait_ms": _distribution(
            [record["raw_prep_wait"] for record in records]
        ),
        "merge_h2d_latency_ms": _distribution(
            [record["merge_h2d"] for record in records]
        ),
        "semantic_batch_envs": _distribution([record["envs"] for record in records]),
    }


def add_control_frame_normalization(summary: dict, control_frames: int) -> dict:
    """Report both physical VLM calls and logical per-environment rows."""

    if control_frames <= 0:
        raise ValueError("control_frames must be positive")
    normalized = dict(summary)
    physical_forwards = int(summary["vlm_forward_count"])
    semantic_rows = int(summary["semantic_rows"])
    normalized["control_frames"] = control_frames
    normalized["physical_vlm_forwards_per_1000_env_control_frames"] = (
        1000.0 * physical_forwards / control_frames
    )
    normalized["logical_semantic_rows_per_1000_env_control_frames"] = (
        1000.0 * semantic_rows / control_frames
    )
    normalized["mean_env_control_frames_per_semantic_row"] = (
        control_frames / semantic_rows
    )
    normalized["mean_semantic_rows_per_physical_vlm_forward"] = (
        semantic_rows / physical_forwards
    )
    # Backward-compatible alias; the explicit physical name is authoritative.
    normalized["vlm_forwards_per_1000_control_frames"] = normalized[
        "physical_vlm_forwards_per_1000_env_control_frames"
    ]
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semantic-server-log", type=Path, action="append", required=True
    )
    parser.add_argument("--control-frames", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = parse_semantic_server_logs(args.semantic_server_log)
    if args.control_frames is not None:
        try:
            summary = add_control_frame_normalization(summary, args.control_frames)
        except ValueError as error:
            parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
