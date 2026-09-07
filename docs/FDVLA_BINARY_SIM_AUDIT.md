# FDVLA binary-reward simulator audit

Audit date: 2026-08-20  
Repository: `/vepfs-mlp2/c20250301/240403026/async_vla/RLinf`

## Git baseline

The required read-only snapshot was taken before edits:

```text
git status --short       # empty
git branch --show-current
codex/fdvla-binary-sim
git rev-parse HEAD
a6d8747a0661fe7ca7ff1a9f7a284d2d75594afa
git diff --stat          # empty
```

`a6d8747a` is also the current `feat/async-ppo` and `origin/feat/async-ppo`
tip. No rebase, merge, pull, reset, force checkout, commit, or push was used.

## 1. LIBERO reward before this experiment

`LiberoEnv.step()` ignores LIBERO's numerical `_reward` return and derives reward
from the simulator `terminations` vector. `_calc_step_reward()` computes

```text
step_penalty + reward_coef * termination
```

and, when `use_rel_reward` is true, returns its temporal difference against
`prev_step_reward`. The stock `env/libero_10.yaml` therefore does **not** implement
the required contract: it uses `reward_coef: 5.0` and `use_rel_reward: true`.

There is a second issue at chunk boundaries. `chunk_step()` continues through the
whole action prefix after a simulator termination. If LIBERO continues returning
`termination=True`, the old `_calc_step_reward()` can expose more than one terminal
bonus even though `_record_metrics()` only counts return before the first success.
The binary experiment must gate the terminal bonus with `~success_once`.

Truncation is computed independently from `elapsed_steps >= max_episode_steps` and
is not passed into `_calc_step_reward()`, so a pure truncation is not success.

The binary path required by this experiment is therefore:

```text
LIBERO termination
  -> first_success = termination & ~success_once
  -> float(first_success) * 1.0
  -> PPO rollout rewards
  -> native GAE
  -> native actor_critic PPO loss
```

It uses no numerical LIBERO dense reward, reward model, progress signal, stage
reward, or step penalty.

## 2. Predicted and executed actions

The GR00T action head predicts `action_horizon` actions (normally 16). The RL
wrapper applies `_execution_action_prefix(predicted, output_action_chunks)` before
returning actions. `output_action_chunks` comes from `num_action_chunks`, and the
environment passes that same value to `prepare_actions()` and then to
`LiberoEnv.chunk_step()`. The paired experiment sets all three relevant horizons
to 16, so 16 actions are predicted and exactly the first 16 are executed.

Action history is updated with that same prefix, not with the unexecuted tail.

## 3. Log-probability/action-prefix alignment

Rollout log-probabilities are sliced in `get_rl_action()` to
`[:, :, :action_chunk, :env_action_dim]`; `action_chunk` is the execution horizon.
The actor replay path applies the same action-chunk and valid/environment action
dimension slices before PPO loss computation. The denoising chain and sampled
`denoise_inds` used during rollout are stored in `forward_inputs` and reused.

Thus PPO log-probability covers the same action prefix sent to the environment.
The contract assumes no external intervention. DAgger is disabled by
`loss_type: actor_critic`; expert labels and intervention replacement never enter
this experiment.

## 4. Semantic tensors in the PPO buffer

During rollout, `_get_rl_action()` detaches every item from `backbone_outputs` and
stores it under `semantic_*` keys, together with `packet_age_s`, action history,
state, denoising chains, and packet identity metadata. `RolloutResult` moves that
dictionary to CPU, `EmbodiedRolloutResult` stacks it by time, and the actor batch
retains it under `forward_inputs`.

The actor `default_forward()` detects the `semantic_*` tensors and constructs a
`BatchFeature` directly from them. In decoupled mode, missing semantic tensors are
a hard error. Consequently the tensor consumed by PPO update is the rollout
tensor, not a newly fetched packet.

The pre-existing code did not count/hash rollout-versus-train semantic mismatches
as an explicit metric. This task adds a deterministic per-row fingerprint to the
rollout buffer and a hard actor-side replay assertion; any mismatch increments the
counter and aborts the update.

## 5. Does actor update invoke the VLM?

No in the decoupled path. `drop_local_backbone: true` substitutes
`_InputOnlyBackbone` before the upstream GR00T constructor. Its `forward()` always
raises, and construction logs:

```text
GR00T execution mode: decoupled
Skipping local VLM construction for DiT-only worker
DiT-only worker contains no local VLM parameters
```

Actor replay uses cached `semantic_*` tensors. The semantic server is frozen and
independent; it is not part of the actor optimizer.

In coupled mode, actor replay has historically fallen back to the local backbone
when no cached semantic tensor is present. That is a local VLM forward during PPO
replay and is scientifically valid for a native coupled baseline, but all backbone
parameters must be frozen. The old coupled example used `dit_only_train: false`,
so it did not establish this boundary and is not a valid paired baseline.

## 6. Parameter loading and freezing

Both rollout and actor start from `GR00T_MODEL_PATH`; the local coupled VLM also
loads its backbone from `GR00T_BACKBONE_PATH`. The decoupled worker replaces the
local backbone and obtains frozen VLM outputs only from the independently launched
semantic server, which loads those same paths.

When `dit_only_train: true`, parameter `requires_grad` is set from an explicit
prefix allowlist. This mechanism is safe for coupled execution as well: it keeps
the local VLM present for forward inference while freezing every `backbone.*`
parameter. The paired configs must use an identical allowlist for the common DiT,
delay adapters, and value head. Runtime reporting/assertions must show:

- VLM total parameters (zero locally for decoupled);
- VLM trainable parameters (always zero);
- trainable DiT/delay-adapter parameters;
- trainable value-head parameters;
- a common-parameter initialization fingerprint.

## 7. Fixed-trial pairing

LIBERO reset state IDs map deterministically to `(task_id, trial_id)`. Evaluation
uses ordered/fixed reset pools, deduplicates each pair, resets the deterministic
semantic-age stream at each evaluation, and derives action noise from
`(task_id, trial_id, frame_id, eval_noise_seed)`. The runner writes one
`eval_trials_step_<step>.jsonl` file per evaluation.

Before this experiment those JSONL rows contain only task ID, trial ID, and
success. They do not persist requested age, actual packet age, bootstrap clipping,
or policy-noise seed. Therefore the existing aggregate is repeatable under one
resolved config, but the file alone cannot prove pairing across arbitrary methods
or distinguish causal episode-start age clipping. The FDVLA manifest and pairing
analysis must make `(task_id, trial_id, semantic_age, policy_noise_seed)` explicit
and treat requested/actual age mismatch caused by `source_frame=max(0,t-d)` as a
separate bootstrap count.

## 8. Present capabilities and missing pieces

Already present on `feat/async-ppo`:

- independent semantic server and per-episode packet histories;
- decoupled worker with no local VLM;
- simulator-frame semantic ages and exact-age cache fetch;
- action-history and packet-age inputs plus zero-input ablation switches;
- rollout-time semantic tensors carried into PPO replay;
- action-prefix/log-probability alignment;
- deterministic trial assignment and action-noise generation;
- native `actor_critic` + GAE + `group_size: 1` PPO;
- evaluation JSONL output and substantial timing instrumentation;
- shared-semantic reward modules, which remain untouched and disabled here.

Missing before the changes in this task:

- a strict first-terminal-success binary reward contract and its tests;
- a binary LIBERO config group (`1.0`, no relative reward, no step penalty);
- scientifically paired N1.7 coupled/decoupled configs from one template;
- frozen-local-VLM assertions and component parameter/fingerprint logs;
- a run manifest for C-SFT, D-SFT, C-PPO, D-PPO, Fresh, NoAge, NoHistory;
- explicit reward-worker invocation and semantic-replay audit counters;
- paired statistical and delay-sweep result tooling;
- the information-probe collection/training/summarization entry points;
- a reproducible two-update Task 0 smoke wrapper and recorded fresh-process check;
- actual pilot/full-suite results under the new binary protocol.

After this audit, the two-update Task 0 smoke, one seed-0 50-update binary D-PPO
pilot, fresh-process checkpoint load, and selected paired development surface cells
were completed and are recorded in `docs/FDVLA_BINARY_SIM_EXPERIMENT.md`. The other
training seeds, paired method pilots, 400-trial formal surface, and full LIBERO-10
suite still require significant external GPU wall-clock time. They are deliberately
not treated as code-only acceptance results and must never be fabricated from the
older learned-reward runs or from synthetic tooling tests.
