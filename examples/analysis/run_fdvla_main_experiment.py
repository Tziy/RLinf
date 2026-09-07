"""Run a bounded, serial C/D experiment from an explicit JSON plan.

Every launch uses a fresh process and the same physical GPU pool. This controller
only cleans up descendants it observed for the explicitly identified launchers.
It stops on failed jobs or unmatched starts instead of silently changing a plan.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import psutil
import yaml


def read_trials(path: Path) -> dict:
    """Read unique trials using task, initial state, and policy noise identity."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities = [(r["task_id"], r["trial_id"], r["policy_noise_seed"]) for r in rows]
    if not rows or len(set(identities)) != len(rows):
        raise ValueError(f"Missing or duplicate trials: {path}")
    return {
        "trials": len(rows),
        "successes": sum(bool(r["success"]) for r in rows),
        "identities": sorted(identities),
        "path": str(path),
    }


def select_pair(candidates: dict, max_gap: int, min_successes: int) -> dict:
    """Select the strongest eligible pair on calibration outcomes only."""
    eligible = []
    for c_name, c in candidates["C"].items():
        for d_name, d in candidates["D"].items():
            if c["identities"] != d["identities"]:
                raise ValueError("Calibration trial identities do not match")
            gap = abs(c["successes"] - d["successes"])
            strength = min(c["successes"], d["successes"])
            if gap <= max_gap and strength >= min_successes:
                eligible.append((-strength, gap, c_name, d_name))
    if not eligible:
        raise ValueError("No eligible matched C/D start; calibration required")
    _, gap, c_name, d_name = min(eligible)
    return {
        "C": c_name,
        "D": d_name,
        "gap_successes": gap,
        "selection_rule": "max_min_success_then_min_gap_then_lexical",
    }


def validate_pair_configs(c: dict, d: dict) -> None:
    """Reject mismatched learning, reward, task, and physical training settings."""
    paths = [
        "algorithm",
        "actor.optim",
        "actor.fsdp_config",
        "actor.micro_batch_size",
        "actor.global_batch_size",
        "actor.seed",
        "actor.model.precision",
        "actor.model.num_action_chunks",
        "actor.model.denoising_steps",
        "rollout.model.precision",
        "reward",
        "critic",
        "cluster.num_nodes",
    ]
    for mode in ["train", "eval"]:
        for key in [
            "task_suite_name",
            "task_id_filter",
            "total_num_envs",
            "auto_reset",
            "ignore_terminations",
            "max_steps_per_rollout_epoch",
            "max_episode_steps",
            "seed",
            "use_rel_reward",
            "reward_coef",
            "use_step_penalty",
        ]:
            paths.append(f"env.{mode}.{key}")
    for key in [
        "train_execution_horizon",
        "eval_execution_horizon",
        "dit_train_last_n_blocks",
        "train_value_head_with_dit_only",
        "use_vlm_value",
        "value_head_init_seed",
    ]:
        paths.append("actor.model.rl_head_config." + key)
    for path in paths:
        values = []
        for cfg in [c, d]:
            value = cfg
            for key in path.split("."):
                value = value[key]
            values.append(value)
        if values[0] != values[1]:
            raise ValueError(f"C/D mismatch at {path}: {values}")


def cleanup(processes: dict) -> None:
    """Terminate only observed process identities, preserving PID reuse safety."""
    live = []
    for pid, started in processes.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == started and process.is_running():
                process.terminate()
                live.append(process)
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(live, timeout=20)
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def observe(process: psutil.Process, seen: dict) -> None:
    """Remember descendants before a launcher exits and reparents its children."""
    try:
        for child in [process, *process.children(recursive=True)]:
            seen[child.pid] = child.create_time()
    except psutil.NoSuchProcess:
        pass


def write_json(path: Path, value: dict) -> None:
    """Atomically publish controller state."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_digest(repo: Path) -> str:
    """Fingerprint tracked and untracked source without logs or bytecode."""
    names = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "rlinf",
            "examples",
            "tests",
            "docs",
        ],
        cwd=repo,
    ).split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(set(names) - {b""}):
        path = repo / os.fsdecode(name)
        if path.is_file():
            digest.update(name + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    # GR00T is imported from the adjacent checkout, outside RLinf's Git tree.
    for path in sorted((repo.parent / "gr00t").rglob("*.py")):
        digest.update(str(path.relative_to(repo.parent)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class Experiment:
    """Execute the registered development pilot and, if healthy, three seeds."""

    def __init__(self, plan: dict) -> None:
        self.plan = plan
        self.repo = Path(plan["repo"])
        self.root = Path(plan["output"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.digest = source_digest(self.repo)
        self.job_number = 0

    def status(self, phase: str, **details) -> None:
        """Expose a compact progress file for infrequent monitoring."""
        write_json(
            self.root / "status.json", dict(phase=phase, utc=time.time(), **details)
        )

    def wait_existing(self) -> None:
        """Release the registered old pilot only after its saved step is logged."""
        previous = self.plan.get("existing")
        if not previous:
            return
        seen = {}
        try:
            process = psutil.Process(previous["pid"])
        except psutil.NoSuchProcess:
            return
        if process.create_time() != previous["create_time"]:
            raise RuntimeError("Existing process identity changed")
        self.status("waiting_for_existing_checkpoint", **previous)
        while process.is_running():
            observe(process, seen)
            metrics = Path(previous["run_dir"]) / "metrics.log"
            steps = re.findall(
                r"Global Step:\s+(\d+)/",
                metrics.read_text() if metrics.exists() else "",
            )
            step = int(steps[-1]) if steps else -1
            target = previous["release_after_step"]
            checkpoints = list(
                Path(previous["run_dir"]).glob(
                    f"*/checkpoints/global_step_{target}/actor/model_state_dict/full_weights.pt"
                )
            )
            # Metrics are written after synchronous evaluation and checkpoint save.
            if step >= target and len(checkpoints) == 1:
                write_json(
                    self.root / "existing_pilot_handoff.json",
                    {
                        "saved_step": target,
                        "checkpoint": str(checkpoints[0]),
                        "reason": "bounded development pilot; switch to matched precision probe",
                        "original_max_steps": 80,
                        "scientific_result_formal": False,
                    },
                )
                cleanup(seen)
                return
            time.sleep(30)
        cleanup(seen)

    def launch(
        self,
        name: str,
        arm: str,
        checkpoint: str,
        *,
        seed: int = 0,
        eval_only: bool = False,
        noise_seed: int | None = None,
        resume: str | None = None,
        steps: int = 10,
        eval_envs: int = 48,
        weights: str | None = None,
    ) -> Path:
        """Run one fresh process, retain config and logs, and verify termination."""
        if source_digest(self.repo) != self.digest:
            raise RuntimeError("Source changed during registered experiment")
        run = self.root / name
        run.mkdir(parents=True, exist_ok=False)
        self.job_number += 1
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(("PPO_", "FDVLA_", "SEMANTIC_", "TRAIN_", "EVAL_")):
                env.pop(key)
        env.update(self.plan["environment"])
        env.update(
            {
                "GR00T_MODEL_PATH": checkpoint,
                "TRAIN_SEED": str(seed),
                "EVAL_NOISE_SEED": str(noise_seed or self.plan["eval_noise_seed"]),
                "EVAL_NUM_ENVS": str(eval_envs),
                "TARGET_EVAL_ENVS": str(eval_envs),
                "PPO_MAX_STEPS": str(steps),
                "PPO_EXPERIMENT_NAME": name.replace("/", "_"),
                "FDVLA_METHOD_ID": arm
                + ("-SFT" if eval_only and weights is None else "-PPO"),
                "FDVLA_RUN_LOG_DIR": str(run),
                "RLINF_LOG_DIR": str(run),
                "FDVLA_METADATA_DIR": str(run / "fdvla_metadata"),
                "FDVLA_RUN_TIMESTAMP": name.replace("/", "_"),
                "RAY_TMPDIR": f"/dev/shm/fm95_{os.getpid()}_{self.job_number}",
                "ONLY_EVAL": "false",  # Use the training entry's post-Hydra metadata.
                "HYDRA_EXTRA_OVERRIDES": "env.train.use_rel_reward=true env.eval.use_rel_reward=true "
                f"actor.seed={1234 + seed} runner.only_eval={str(eval_only).lower()}",
            }
        )
        if resume:
            env["PPO_RESUME_DIR"] = resume
        if weights:
            env["PPO_CKPT_PATH"] = weights
        command = [
            "bash",
            str(
                self.repo
                / "examples/embodiment"
                / (
                    "run_coupled_n1d7_binary_ppo.sh"
                    if arm == "C"
                    else "run_fdvla_binary_ppo.sh"
                )
            ),
        ]
        # The allowlisted launch settings contain paths and experiment settings only.
        write_json(
            run / "launch.json",
            {
                "command": command,
                "environment": {k: env[k] for k in self.plan["environment"]},
                "checkpoint": checkpoint,
                "seed": seed,
                "steps": steps,
                "eval_only": eval_only,
                "resume": resume,
                "weights": weights,
                "hydra_overrides": env["HYDRA_EXTRA_OVERRIDES"],
                "source_sha256": self.digest,
            },
        )
        # CUDA context release may lag process termination; never kill a
        # GPU process that was not part of this controller's observed tree.
        deadline = time.monotonic() + 120
        while True:
            gpu_rows = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            if not gpu_rows.strip():
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"GPU processes remain before {name}: {gpu_rows}")
            time.sleep(5)
        seen = {}
        self.status("running", job=name, run_dir=str(run), source_sha256=self.digest)
        start = time.time()
        with (run / "launcher.log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=self.repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process = psutil.Process(child.pid)
            while child.poll() is None:
                observe(process, seen)
                if time.time() - start > self.plan["job_timeout_hours"] * 3600:
                    cleanup(seen)
                    raise RuntimeError(f"Job time budget exceeded: {name}")
                time.sleep(10)
        cleanup(seen)
        write_json(
            run / "completion.json",
            {"returncode": child.returncode, "elapsed_seconds": time.time() - start},
        )
        if child.returncode:
            raise RuntimeError(f"Job failed ({child.returncode}): {name}")
        output = run / (
            "eval_trials.jsonl" if eval_only else f"eval_trials_step_{steps}.jsonl"
        )
        trials = read_trials(output)
        if trials["trials"] != eval_envs:
            raise RuntimeError(f"Incomplete evaluation: {output}")
        if not eval_only:
            with (run / "summary.log").open("w") as report:
                subprocess.run(
                    [
                        env["PATH"].split(":")[0] + "/python",
                        str(
                            self.repo
                            / "examples/analysis/summarize_fdvla_training_run.py"
                        ),
                        "--run-dir",
                        str(run),
                        "--output-dir",
                        str(run / "summary"),
                    ],
                    cwd=self.repo,
                    env=env,
                    stdout=report,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            health = json.loads(
                (run / "summary/fdvla_training_health.json").read_text()
            )
            for field in [
                "nonfinite_scalar_count",
                "reward_model_invocation_count",
                "trainable_vlm_parameter_count",
                "semantic_replay_fingerprint_mismatch_log_count",
            ]:
                if health.get(field) != 0:
                    raise RuntimeError(
                        f"Health gate {field}: {health.get(field)} in {name}"
                    )
            if arm == "D":
                for field in [
                    "cross_episode_packet_mismatch_count",
                    "local_vlm_forward_count",
                ]:
                    if health.get(field) != 0:
                        raise RuntimeError(
                            f"D contract {field}: {health.get(field)} in {name}"
                        )
        return run

    def checkpoint(self, run: Path, step: int) -> Path:
        """Require exactly one completed checkpoint for the registered step."""
        matches = list(run.glob(f"*/checkpoints/global_step_{step}"))
        if (
            len(matches) != 1
            or not (matches[0] / "actor/model_state_dict/full_weights.pt").is_file()
        ):
            raise RuntimeError(f"Missing checkpoint {step} in {run}")
        return matches[0]

    def run(self) -> None:
        """Calibrate starts, run both pilots, then continue with fixed seeds."""
        self.wait_existing()
        candidates = {"C": {}, "D": {}}
        for arm in ("C", "D"):
            for label, checkpoint in self.plan["candidates"][arm].items():
                run = self.launch(
                    f"calibration/{arm}_{label}",
                    arm,
                    checkpoint,
                    eval_only=True,
                    noise_seed=self.plan["selection_noise_seed"],
                )
                candidates[arm][label] = read_trials(run / "eval_trials.jsonl")
        selected = select_pair(
            candidates,
            self.plan["max_start_gap_successes"],
            self.plan["min_start_successes"],
        )
        write_json(
            self.root / "selected.json",
            {"selected": selected, "candidates": candidates},
        )
        starts = {a: self.plan["candidates"][a][selected[a]] for a in ("C", "D")}
        pilots = {}
        for arm in ("C", "D"):
            pilots[arm] = self.launch(f"seed0/{arm}_pilot", arm, starts[arm], steps=10)
        configs = [
            yaml.safe_load(
                (pilots[a] / "fdvla_metadata/resolved_config.yaml").read_text()
            )
            for a in ("C", "D")
        ]
        validate_pair_configs(*configs)
        for arm, run in pilots.items():
            before = read_trials(run / "eval_trials_step_0.jsonl")
            after = read_trials(run / "eval_trials_step_10.jsonl")
            if (
                after["successes"]
                < before["successes"] - self.plan["max_pilot_drop_successes"]
            ):
                raise RuntimeError(
                    f"Pilot collapsed: {arm}; retain both results for diagnosis"
                )
        c0 = read_trials(pilots["C"] / "eval_trials_step_0.jsonl")
        d0 = read_trials(pilots["D"] / "eval_trials_step_0.jsonl")
        if (
            c0["identities"] != d0["identities"]
            or abs(c0["successes"] - d0["successes"])
            > self.plan["max_start_gap_successes"]
        ):
            raise RuntimeError(
                "Fresh-process PPO initial evaluations fail matching gate"
            )
        final_runs = {}
        for seed in self.plan["training_seeds"]:
            for arm in ("C", "D") if seed % 2 == 0 else ("D", "C"):
                resume = str(self.checkpoint(pilots[arm], 10)) if seed == 0 else None
                run = self.launch(
                    f"seed{seed}/{arm}_main",
                    arm,
                    starts[arm],
                    seed=seed,
                    resume=resume,
                    steps=self.plan["main_steps"],
                )
                final_runs[f"{seed}_{arm}"] = str(run)
        write_json(self.root / "training_complete.json", final_runs)
        for seed in self.plan["training_seeds"]:
            for arm in ("C", "D"):
                checkpoint = self.checkpoint(
                    Path(final_runs[f"{seed}_{arm}"]), self.plan["main_steps"]
                )
                for noise in self.plan["final_noise_seeds"]:
                    self.launch(
                        f"final/seed{seed}_{arm}_initial_noise{noise}",
                        arm,
                        starts[arm],
                        seed=seed,
                        eval_only=True,
                        noise_seed=noise,
                        eval_envs=100,
                    )
                    self.launch(
                        f"final/seed{seed}_{arm}_noise{noise}",
                        arm,
                        starts[arm],
                        seed=seed,
                        eval_only=True,
                        noise_seed=noise,
                        eval_envs=100,
                        weights=str(
                            checkpoint / "actor/model_state_dict/full_weights.pt"
                        ),
                    )
        self.status(
            "training_and_final_evaluation_complete",
            final_runs=final_runs,
            scientific_result_formal=False,
            remaining="paired statistical and resource audit",
        )


def main() -> None:
    """Execute an explicit plan once; never restart an existing output tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    experiment = Experiment(json.loads(args.plan.read_text()))
    if (experiment.root / "status.json").exists():
        raise RuntimeError("Existing controller status; inspect before resuming")
    try:
        experiment.run()
    except Exception as error:
        experiment.status("needs_attention", error=str(error))
        raise


if __name__ == "__main__":
    main()
