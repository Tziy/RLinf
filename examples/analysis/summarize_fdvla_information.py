#!/usr/bin/env python3
"""Combine probe retention and D-SFT/D-PPO delay-sweep success curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_success(path: Path):
    rows = json.loads(path.read_text())
    return {str(row["semantic_age"]): float(row["success_rate"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-metrics", type=Path, required=True)
    parser.add_argument("--d-sft-delay", type=Path, required=True)
    parser.add_argument("--d-ppo-delay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads(args.probe_metrics.read_text())
    retention = probe["retention_R_by_age"]
    d_sft = _load_success(args.d_sft_delay)
    d_ppo = _load_success(args.d_ppo_delay)
    ages = sorted(set(retention) & set(d_sft) & set(d_ppo), key=int)
    if not ages:
        raise ValueError(
            "Probe, D-SFT, and D-PPO inputs have no overlapping semantic ages"
        )
    rows = [
        {
            "semantic_age": int(age),
            "variational_predictive_information_retention_R": retention[age],
            "D-SFT_success": d_sft[age],
            "D-PPO_success": d_ppo[age],
        }
        for age in ages
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "information_delay_summary.csv").open(
        "w", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "information_delay_summary.json").write_text(
        json.dumps(
            {
                "estimate_name": "variational predictive-information estimate",
                "rows": rows,
                "interpretation_guardrail": (
                    "The frozen VLM does not gain information during PPO; improved "
                    "success can only show better use of residual stale-semantic information."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
