# Task 0 C-PPO / D-PPO experiment record

This experiment tests whether frozen semantic inference can run asynchronously
while a DiT-only policy learns with the measured delay, using the same physical
allocation of four A100 80 GB GPUs. The working branch is `feat/async-ppo`, whose
remote head was checked at `a6d8747a`. Existing unrelated changes are retained.

D development training and its reserved-noise verification have completed.
Matched multi-training-seed C/D results are not complete. The historical plan in
`logs/fdvla_main_20260905/plan.json` is superseded: its candidate-selection,
resume, and 100-state evaluation schedule must not be used as the current plan.

## Effective training configuration

The completed D run is under
`logs/fdvla_main_20260905/run_v15_d_priority/seed0/D_main`.
Its `fdvla_metadata/resolved_config.yaml` is the post-Hydra source of truth.
The earlier preflight metadata is retained separately because CLI overrides can
change reward settings and semantic ports. Training shell failures propagate
through `tee`.

Both development arms use K=2, prediction chunk16, four denoising steps,
train120/eval48, 480 rollout/episode frames, no auto-reset, global batch900,
micro batch15, two PPO epochs, actor LR2e-6, value LR1e-4, and KL beta0.
Actor parameters are FP32; FSDP computation and rollout are BF16.
Gamma is0.9974905699322298 and GAE lambda is0.9872585449025169, preserving
the inherited K=1 frame discount scale. Evaluation/checkpoint intervals are5.

The shared trainable scope includes the last four DiT transformer blocks,
timestep encoder, output normalization/projections, and value head.
D additionally trains packet-age and action-history adapters. The VLM is frozen
in both arms; C retains local coupled inference and D drops the local backbone.
D uses the latest available semantic packets, published every eight frames,
without imposed delay or exact-age scheduling. Semantic servers count within
the same four physical GPUs.

Both environment relative-reward switches are true. Here this is the difference
of the wrapper's first-success reward, not dense simulator progress and not a
strict binary-terminal-reward experiment. Reward Model, GRPO, DAgger, expert
loss, privileged actor/critic observations, and post-hoc augmentation are disabled.
Actor inputs use visual semantics and robot proprioception; D also conditions
on packet age and recent action history. The critic concatenates pooled semantic
features with encoded robot state. `use_vlm_value=false` means state is included;
it does not remove semantics from the critic.

Native `critic_warmup_steps=3` counts optimizer steps, not rollout updates.
There are64 optimizer steps per full update in this setup. A proposed five-update
critic warmup remains unlaunched because the user chose to continue the learning
D run. Do not attribute its improvement to that proposed warmup.

## Completed development evidence

C was stopped at its complete step45 checkpoint when the user prioritized D.
D subsequently completed all80 updates and exited successfully. Its fixed
noise66026 evaluation rose from26/48 initially to39/48 at the final checkpoint;
the best intermediate point was47/48. Use the final checkpoint, not the best
single evaluation, for the reserved-noise verification.

Reserved policy-noise seeds56026..56029 gave initial counts24,26,22,19 and final
counts38,42,40,41, each out of48. The pooled change is91/192 (47.4%) to161/192
(83.9%), or+36.5 percentage points. A paired bootstrap averaging the four noises
within each initial state and resampling48 states gives a95% interval of
+24.0 to+48.4 percentage points. This does not estimate training-seed uncertainty.
The predeclared development gate (final>=40%, gain>=10 percentage points,
bootstrap lower bound>0) passed. No wall-clock superiority over C is established
by this gate.

LIBERO-10 Task0 has50 initial states. Evaluations use48 unique state identities
per noise;192 trajectories per policy are not192 independent states. Training
and evaluation share this state pool, so do not claim unseen-state or task
suite-wide generalization.

Starting performance still needs a common-panel check before the main comparison.
The earlier three-noise calibration averaged C60/144 (41.7%) and D70/144 (48.6%),
but the fresh training-run initial evaluations were C16/48 and D26/48.
Configuration equality alone does not make those single initial evaluations
matched. Do not repeatedly restart D until a favorable asynchronous realization
happens to match C.

## System metric coverage

The completed development source has a profiling defect: the replay fingerprint
is a float tensor[B,64], while packet-reuse profiling flattened it and expectedB
scalars. Training packet-consumption/reuse counters were consequently missing, despite
being emitted as zeros. Treat those historical training counters as unavailable.
Standalone evaluation omits the optional replay fingerprint, so its packet
identity/reuse counters remain valid and are retained in the evaluation events.

The corrected profiler preserves each full fingerprint row, including fractional
values, and marks malformed/nonfinite rows incomplete. This affects observational
counters only. PPO semantic-tensor replay checks and central-cache episode
validation are separate and were active throughout development.
Regression coverage is in `tests/unit_tests/test_fdvla_semantic_vector_profile.py`
and `tests/unit_tests/test_fdvla_system_profile.py`.

Report physical VLM batch calls separately from logical semantic rows: staggered
per-environment publication can retain frequent small batches while reducing the
number of encoded observations. Server logs retain both quantities, summarized by
`examples/analysis/summarize_fdvla_system_profile.py`. The profiler's
`dit_forwards_per_unique_semantic_packet` is a ratio of logical action-generation
rows to unique packets, not a count of all internal denoising-network calls.
Frame age and source-to-completion wall-clock latency are distinct measures.

Distinguish budgeted environment frames from action-generation calls, including
value-bootstrap calls. The training summary records the frame denominator's
source. Report local worker memory separately from total physical GPU memory,
which also includes semantic servers. Preserve startup, scheduled evaluation,
checkpointing, and final verification costs in the resource ledger.

## Artifacts and inspection

Experiment artifacts live under `logs/fdvla_main_20260905`:

- `run_v15_d_priority/seed0/D_main/summary/`: all80 scalar updates and health audit.
- `run_v17_d_repeat_eval/D_repeated_noise_report.json`: reserved-noise paired results.
- `figures_development/D_seed0_learning_development.pdf`: full development curve.
- `figures_development/D_seed0_fixed_eval.csv`: source counts and loop time.
- `CD_EFFECTIVE_CONFIG_AUDIT.json`: actual configuration equality and limitations.
- `D_SYSTEM_METRIC_LIMITATIONS.json`: historical missing packet-reuse statistics.
- `physical_gpu_samples.csv`: physical GPU sampling across sequential jobs.
- `audit/source_before_launch.tar.gz`: preserved development source.

Inspect current state without launching another job:

```bash
/opt/venv/openvla/bin/python logs/fdvla_main_20260905/snapshot_active.py
```

Use a fresh output directory and explicit source-pinned plan for new runs.
Controllers retain launch settings, effective configuration, checkpoint identity,
process duration, and unique trial JSONL; they refuse overwrites and source changes.
Cleanup is restricted to observed process identities. The original generic
controller's calibration/resume schedule is historical and is not an instruction
to resume C or launch new arms automatically.

Further completion requires common-panel start matching, independent paired
training seeds, corrected system profiling, and full wall-clock/resource analysis.
Analyze each policy's change from its own start and paired C/D differences;
report unreached success thresholds as unreached. Keep development data distinct
from the subsequently registered main replication protocol.
