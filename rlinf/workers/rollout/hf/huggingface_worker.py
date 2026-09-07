# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import gc
import time
from collections import defaultdict
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.rlt import (
    build_rlt_route,
    predict_rlt_actions,
)
from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, Worker, split_channel_message
from rlinf.utils.checkpoint import (
    gr00t_delay_adapter_missing_prefixes,
    load_state_dict_with_allowed_missing,
)
from rlinf.utils.fdvla_realtime import (
    resolve_eval_execution_horizon,
    resolve_train_execution_horizon,
)
from rlinf.utils.nested_dict_process import split_dict
from rlinf.utils.placement import HybridComponentPlacement


def _fdvla_sample_summary(values: list[float]) -> dict[str, float]:
    """Summarize one rollout-worker profiling stream in its native units."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {}
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "count": float(len(array)),
    }


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.only_eval = cfg.runner.get("only_eval", False)
        self.algorithm_cfg = cfg.get("algorithm", {})
        self.model_cfg = cfg.rollout.model if self.only_eval else cfg.actor.model
        self.actor_group_name = (
            cfg.actor.get("group_name", None)
            if cfg.get("actor", None) is not None
            else None
        )
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        rollout_world_size = self.placement.get_world_size("rollout")
        self.actor_weight_src_rank = 0
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        self._weight_sync_is_sender = self._rank == 0
        train_env_cfg = cfg.env.get("train", None)
        eval_env_cfg = cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.enable_dagger = self.algorithm_cfg.get("loss_type") == "embodied_dagger"
        self.enable_opd = self.algorithm_cfg.get("adv_type") == "opd"
        self.expert_model = None
        self.rlt_feature_model = None
        self.rlt_route = None

        self.total_num_train_envs = (
            cfg.env.train.total_num_envs if self.enable_train else 0
        )
        self.total_num_eval_envs = (
            cfg.env.eval.total_num_envs if self.enable_eval else 0
        )
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = self.total_num_train_envs // self.num_pipeline_stages
        self.eval_batch_size = self.total_num_eval_envs // self.num_pipeline_stages

        self.per_node_train_batch_size = (
            self.train_batch_size // self._world_size if self.enable_train else 0
        )
        self.per_node_eval_batch_size = (
            self.eval_batch_size // self._world_size if self.enable_eval else 0
        )

        self.inference_micro_batch_size = int(
            self.cfg.rollout.get("inference_micro_batch_size", 0)
        )
        if self.inference_micro_batch_size < 0:
            raise ValueError("rollout.inference_micro_batch_size must be non-negative")

        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)

        self.n_train_chunk_steps = 0
        if self.enable_train:
            train_execution_horizon = resolve_train_execution_horizon(
                self.model_cfg.num_action_chunks,
                self.model_cfg.get("rl_head_config", {}).get(
                    "train_execution_horizon", -1
                ),
            )
            train_control_frames = int(cfg.env.train.max_steps_per_rollout_epoch)
            if train_control_frames % train_execution_horizon:
                raise ValueError(
                    "train max_steps_per_rollout_epoch must be divisible by "
                    "train_execution_horizon: "
                    f"control_frames={train_control_frames}, "
                    f"execution_horizon={train_execution_horizon}"
                )
            self.n_train_chunk_steps = train_control_frames // train_execution_horizon
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            eval_execution_horizon = resolve_eval_execution_horizon(
                self.model_cfg.num_action_chunks,
                self.model_cfg.get("rl_head_config", {}).get(
                    "eval_execution_horizon", -1
                ),
            )
            eval_control_frames = int(cfg.env.eval.max_steps_per_rollout_epoch)
            if eval_control_frames % eval_execution_horizon:
                raise ValueError(
                    "eval max_steps_per_rollout_epoch must be divisible by "
                    "eval_execution_horizon: "
                    f"control_frames={eval_control_frames}, "
                    f"execution_horizon={eval_execution_horizon}"
                )
            self.n_eval_chunk_steps = eval_control_frames // eval_execution_horizon
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.finished_episodes = None

        self.weight_syncer = None
        self._sync_weight_comm_options = None
        if not self.only_eval:
            weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
            assert weight_syncer_cfg is not None, (
                "rollout.weight_syncer config must be provided"
            )
            self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
            self._sync_weight_comm_options = self.weight_syncer.comm_options

        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)

        if self.env_decoupled_mode:
            # save the run-time imformation in communicate channel for decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {
                "rollout_results": [],
            }
        self.rollout_queue_size = self.cfg.rollout.get("rollout_queue_size", 0)
        self._fdvla_profile_samples: dict[str, list[float]] = defaultdict(list)
        self._fdvla_action_boundaries = 0
        self._fdvla_control_frames = 0
        self._fdvla_local_vlm_forward_count = 0
        self._fdvla_semantic_packet_consumptions = 0
        self._fdvla_semantic_unique_packets: set[tuple[int, ...]] = set()
        self._fdvla_semantic_consecutive_reuses = 0
        self._fdvla_semantic_identity_incomplete_count = 0
        self._fdvla_last_semantic_packet_by_env: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}

    def pop_execution_times(self) -> dict[str, float]:
        """Return native timers plus windowed FDVLA latency distributions."""
        metrics = super().pop_execution_times()
        for name, values in self._fdvla_profile_samples.items():
            for statistic, value in _fdvla_sample_summary(values).items():
                metrics[f"profile/{name}_{statistic}"] = value
        metrics["profile/action_boundary_count"] = float(self._fdvla_action_boundaries)
        metrics["profile/control_frame_count"] = float(self._fdvla_control_frames)
        metrics["profile/local_vlm_forward_count"] = float(
            self._fdvla_local_vlm_forward_count
        )
        packet_consumptions = int(
            getattr(self, "_fdvla_semantic_packet_consumptions", 0)
        )
        unique_packets = len(getattr(self, "_fdvla_semantic_unique_packets", set()))
        consecutive_reuses = int(getattr(self, "_fdvla_semantic_consecutive_reuses", 0))
        metrics["profile/semantic_packet_consumption_count"] = float(
            packet_consumptions
        )
        metrics["profile/semantic_unique_packet_count"] = float(unique_packets)
        metrics["profile/semantic_consecutive_reuse_count"] = float(consecutive_reuses)
        metrics["profile/dit_forwards_per_unique_semantic_packet"] = float(
            packet_consumptions / unique_packets if unique_packets else 0.0
        )
        metrics["profile/semantic_consecutive_reuse_fraction"] = float(
            consecutive_reuses / packet_consumptions if packet_consumptions else 0.0
        )
        metrics["profile/semantic_identity_incomplete_count"] = float(
            getattr(self, "_fdvla_semantic_identity_incomplete_count", 0)
        )
        metrics["profile/cross_episode_packet_mismatch_count"] = float(
            getattr(
                getattr(self, "hf_model", None),
                "_semantic_cross_episode_packet_mismatch_count",
                0,
            )
        )
        metrics["profile/semantic_latest_fetch_skipped_count"] = float(
            getattr(
                getattr(self, "hf_model", None),
                "_semantic_latest_fetch_skipped_count",
                0,
            )
        )
        if torch.cuda.is_available():
            metrics["profile/peak_gpu_memory_mib"] = float(
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        self._fdvla_profile_samples.clear()
        self._fdvla_action_boundaries = 0
        self._fdvla_control_frames = 0
        self._fdvla_local_vlm_forward_count = 0
        self._fdvla_semantic_packet_consumptions = 0
        self._fdvla_semantic_unique_packets = set()
        self._fdvla_semantic_consecutive_reuses = 0
        self._fdvla_semantic_identity_incomplete_count = 0
        self._fdvla_last_semantic_packet_by_env = {}
        return metrics

    def _record_fdvla_semantic_reuse(self, forward_inputs: dict[str, Any]) -> None:
        """Count policy-side reuse of the exact rollout semantic packet."""
        identity_fields = (
            "rollout_semantic_env_ids",
            "rollout_semantic_episode_generations",
            "rollout_semantic_source_frame_ids",
            "rollout_semantic_versions",
        )
        missing_fields = [
            name for name in identity_fields if forward_inputs.get(name) is None
        ]
        if missing_fields:
            if not getattr(self, "_fdvla_semantic_identity_warning_emitted", False):
                worker_logger = getattr(self, "_logger", None)
                if worker_logger is not None:
                    worker_logger.warning(
                        "FDVLA semantic reuse identity unavailable: missing=%s available=%s",
                        missing_fields,
                        sorted(forward_inputs),
                    )
                self._fdvla_semantic_identity_warning_emitted = True
            self._fdvla_semantic_identity_incomplete_count = (
                int(getattr(self, "_fdvla_semantic_identity_incomplete_count", 0)) + 1
            )
            return
        rows = [
            torch.as_tensor(forward_inputs[name]).detach().cpu().reshape(-1).tolist()
            for name in identity_fields
        ]
        row_count = len(rows[0])
        if any(len(values) != row_count for values in rows[1:]):
            self._fdvla_semantic_identity_incomplete_count = (
                int(getattr(self, "_fdvla_semantic_identity_incomplete_count", 0)) + 1
            )
            return
        fingerprint = forward_inputs.get("rollout_semantic_fingerprint")
        fingerprint_values = None
        if fingerprint is not None:
            fingerprint_rows = torch.as_tensor(fingerprint).detach().cpu()
            if (
                fingerprint_rows.ndim == 0
                or fingerprint_rows.shape[0] != row_count
                or fingerprint_rows.numel() == 0
                or not torch.isfinite(fingerprint_rows).all()
            ):
                self._fdvla_semantic_identity_incomplete_count = (
                    int(getattr(self, "_fdvla_semantic_identity_incomplete_count", 0))
                    + 1
                )
                return
            # Replay auditing stores a vector of float samples for each env.
            # Preserve the batch axis and fractional values in packet identity.
            fingerprint_values = fingerprint_rows.reshape(row_count, -1).tolist()
        unique_packets = getattr(self, "_fdvla_semantic_unique_packets", None)
        if unique_packets is None:
            unique_packets = set()
            self._fdvla_semantic_unique_packets = unique_packets
        last_by_env = getattr(self, "_fdvla_last_semantic_packet_by_env", None)
        if last_by_env is None:
            last_by_env = {}
            self._fdvla_last_semantic_packet_by_env = last_by_env
        for row_index, (env_id, generation, source_frame, version) in enumerate(
            zip(*rows, strict=True)
        ):
            identity = (
                int(env_id),
                int(generation),
                int(source_frame),
                int(version),
            )
            if fingerprint_values is not None:
                identity = (*identity, tuple(fingerprint_values[row_index]))
            stream = (int(env_id), int(generation))
            if last_by_env.get(stream) == identity:
                self._fdvla_semantic_consecutive_reuses = (
                    int(getattr(self, "_fdvla_semantic_consecutive_reuses", 0)) + 1
                )
            last_by_env[stream] = identity
            unique_packets.add(identity)
            self._fdvla_semantic_packet_consumptions = (
                int(getattr(self, "_fdvla_semantic_packet_consumptions", 0)) + 1
            )

    def _record_fdvla_profile(
        self,
        *,
        actions: torch.Tensor | np.ndarray,
        result: dict[str, Any],
        prediction_s: float,
        semantic_fetch_s: float,
    ) -> None:
        """Record observational timing only; never feed it into policy state."""
        if not hasattr(self.hf_model, "_last_semantic_fetch_s"):
            return
        total_ms = max(0.0, float(prediction_s) * 1000.0)
        semantic_ms = max(0.0, float(semantic_fetch_s) * 1000.0)
        self._fdvla_profile_samples["action_generation_total_ms"].append(total_ms)
        self._fdvla_profile_samples["semantic_fetch_ms"].append(semantic_ms)
        self._fdvla_profile_samples["dit_generation_excluding_semantic_ms"].append(
            max(0.0, total_ms - semantic_ms)
        )

        head_cfg = self.model_cfg.get("rl_head_config", {})
        # This is the complete action-critical semantic path, including exact-age
        # waits, payload transfer and local coupled VLM compute. A server's
        # narrower foreground-wait metric omits exact fetches and is not suitable
        # as the action-boundary blocking definition.
        blocking_ms = semantic_ms
        self._fdvla_profile_samples["action_boundary_blocking_ms"].append(
            max(0.0, blocking_ms)
        )

        forward_inputs = result.get("forward_inputs", {})
        self._record_fdvla_semantic_reuse(forward_inputs)
        source = forward_inputs.get("rollout_semantic_source_wallclock_s")
        completed = forward_inputs.get("rollout_semantic_completed_wallclock_s")
        if source is not None and completed is not None:
            queue_ms = (
                (torch.as_tensor(completed) - torch.as_tensor(source))
                .clamp_min(0)
                .double()
                .reshape(-1)
                .mul(1000.0)
                .cpu()
                .tolist()
            )
            self._fdvla_profile_samples["semantic_queue_latency_ms"].extend(queue_ms)

        actual_age = forward_inputs.get("rollout_semantic_actual_age_frames")
        if actual_age is None and forward_inputs.get("packet_age_s") is not None:
            actual_age = torch.as_tensor(forward_inputs["packet_age_s"]) * float(
                head_cfg.get("semantic_control_hz", 20.0)
            )
        if actual_age is not None:
            self._fdvla_profile_samples["semantic_age_frames"].extend(
                torch.as_tensor(actual_age).float().reshape(-1).cpu().tolist()
            )

        action_shape = tuple(actions.shape)
        batch_size = int(action_shape[0]) if action_shape else 0
        chunk_size = int(action_shape[1]) if len(action_shape) >= 2 else 1
        self._fdvla_action_boundaries += 1
        self._fdvla_control_frames += batch_size * chunk_size
        if str(head_cfg.get("execution_mode", "coupled")) == "coupled":
            self._fdvla_local_vlm_forward_count += 1

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.model_cfg)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            self._logger.info(
                f"Loading rollout override checkpoint from {self.cfg.runner.ckpt_path}"
            )
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            dit_only = bool(
                self.model_cfg.get("rl_head_config", {}).get(
                    "drop_local_backbone", False
                )
            )
            if dit_only:
                target_keys = set(self.hf_model.state_dict())
                unexpected_keys = set(model_dict) - target_keys
                invalid_keys = [
                    key for key in unexpected_keys if not key.startswith("backbone.")
                ]
                if invalid_keys:
                    raise RuntimeError(
                        "DiT-only rollout checkpoint has unexpected non-backbone keys: "
                        f"{sorted(invalid_keys)}"
                    )
                model_dict = {
                    key: value
                    for key, value in model_dict.items()
                    if key in target_keys
                }
                self._logger.info(
                    "Discarded %d frozen backbone tensors from the DiT-only "
                    "rollout checkpoint",
                    len(unexpected_keys),
                )
            if dit_only:
                missing_keys = load_state_dict_with_allowed_missing(
                    self.hf_model,
                    model_dict,
                    allowed_missing_prefixes=gr00t_delay_adapter_missing_prefixes(
                        self.model_cfg.get("rl_head_config", {})
                    ),
                )
                if missing_keys:
                    self._logger.info(
                        "Initialized %d new value-head tensors outside the rollout "
                        "override checkpoint",
                        len(missing_keys),
                    )
            else:
                self.hf_model.load_state_dict(model_dict)
            self._logger.info("Loaded rollout override checkpoint successfully")

        rlt_feature_model_config = OmegaConf.select(
            self.cfg, "rollout.rlt_feature_model", default=None
        )
        if rlt_feature_model_config is not None:
            self.rlt_feature_model = get_model(copy.deepcopy(rlt_feature_model_config))
            self.rlt_feature_model.eval()
            self.rlt_feature_model.requires_grad_(False)
            self.rlt_route = build_rlt_route(self.cfg)

        if self.cfg.rollout.get("expert_model", None) and not self.enable_opd:
            expert_model_config = build_expert_model_config(
                self.cfg,
                self.model_cfg,
                rlt_feature_model_config=rlt_feature_model_config,
            )
            self.expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                self.expert_model.load_state_dict(expert_model_dict)

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # sampling parameters for rollout
        sampling_params = self.cfg.rollout.get("sampling_params", None)
        if sampling_params is not None:
            sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            self._train_sampling_params = {
                "do_sample": sampling_params["do_sample"],
                "temperature": sampling_params["temperature_train"]
                if sampling_params["do_sample"]
                else 1.0,
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
            self._eval_sampling_params = {
                "do_sample": True
                if sampling_params.get("temperature_eval", -1) > 0
                else False,
                "temperature": sampling_params["temperature_eval"],
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
        else:
            self._train_sampling_params = {}
            self._eval_sampling_params = {}

        if self.expert_model is not None and self.enable_dagger:
            self._dagger_sampling_params = {
                "beta": self.algorithm_cfg.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.algorithm_cfg.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    async def recv_from_and_record_batch_routes_with_timeout(
        self,
        group_name: str,
        channel: Any | None,
        *,
        route_key: Any = None,
        tag: str | None = None,
        batch_size: int | None = None,
        merge_fn: Optional[Callable[[list[Any]], Any]] = None,
        infer_batch_size_fn: Optional[Callable[[Any], int]] = None,
        timeout_time: float = 0.02,
        recv_queue_size: int = 0,
    ):
        """Receive routed batch shards and record their return routes.

        This method is used in env-decoupled mode. It builds a receive plan for the
        source worker group, receives shard messages from ``channel`` one by one, and
        stops when all planned items are received or ``timeout_time`` is reached.

        Each received channel item must be a dict with ``batch_index`` (route
        metadata) and ``batch`` (payload shard).

        The ``batch_index`` values are stored in ``self.batch_router[tag]`` so a
        later send call can split the response and send each shard back to the original
        source rank. The received payload shards are validated, then merged with
        ``merge_fn`` or the default ``merge_batches``.

        Args:
            group_name: Source worker group name.
            channel: Channel used to receive routed batch shards.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag used to build receive keys and index recorded routes.
            batch_size: Expected batch size for each planned receive entry.
            merge_fn: Optional custom function for merging received shards.
            infer_batch_size_fn: Optional function used to infer shard batch size during
                validation.
            timeout_time: Maximum time in seconds to wait before finalizing partial
                results.
            recv_queue_size: Number of receive queue entries used when building the
                receive plan.

        Returns:

            A merged payload and its split sizes. If only one shard is received, the
            current implementation returns that shard directly.
        """
        from rlinf.scheduler import (
            decoupled_build_recv_plan,
            get_batch_size,
            get_group_world_size,
            merge_batches,
        )

        world_size = get_group_world_size(self._manager_proxy, group_name)
        plan = decoupled_build_recv_plan(
            src_group_name=group_name,
            dst_group_name=self.worker_address.root_group_name,
            recv_rank=None,
            src_world_size=self._world_size,
            dst_world_size=world_size,
            tag=tag,
            route_key=route_key,
            batch_size=batch_size,
            recv_queue_size=recv_queue_size,
        )

        def _finalize(received_items: list[Any]):
            if not received_items:
                assert False, "received_items is empty"

            # get the tag from the received_items
            _, _, _, tag = split_channel_message(received_items[0]["batch_index"])

            assert tag in self.batch_router, (
                f"{tag=} need to be already in the batch_router"
            )
            # Save the batch_index to the batch_router.
            list_received_items = []
            for item in received_items:
                batch_index = item["batch_index"]
                received_item = item["batch"]
                list_received_items.append(received_item)
                # Save the batch_index to the batch_router.
                self.batch_router[tag].append(batch_index)
            received_items = list_received_items
            split_sizes = [
                get_batch_size(item, infer_batch_size_fn) for item in received_items
            ]

            if merge_fn is not None:
                return merge_fn(received_items), split_sizes
            if len(received_items) == 1:
                return received_items[0]
            return merge_batches(received_items), split_sizes

        timeout_time = timeout_time + time.time()
        get_items = None
        max_item_num = len(plan.entries)
        get_item_num = 0
        received_items = []
        while get_item_num < max_item_num:
            # get the items
            if get_items is None:
                get_items = channel.get(
                    key=plan.entries[get_item_num].key, async_op=True
                )
            else:
                # Now, the worker is getting a item, sleep to wait
                await asyncio.sleep(0.0001)

            # handle the get_items finish
            if get_items.done():
                # save the data and init the get_items to get next data
                received_items.append(await get_items.async_wait())
                get_items = None
                get_item_num = get_item_num + 1

            # handle the timeout case
            if time.time() >= timeout_time:
                max_item_num = get_item_num
                if get_items is not None:
                    received_items.append(await get_items.async_wait())
                    get_items = None
                    get_item_num = get_item_num + 1

        return _finalize(received_items)

    def send_to_recorded_batch_routes(
        self,
        group_name: str,
        channel: Any | None,
        data: Any,
        *,
        route_key: Any = None,
        tag: str | None = None,
        split_fn: Optional[Callable[[Any, list[int]], list[Any]]] = None,
        split_sizes: list[int],
    ):
        """Send split batch results back using recorded batch routes.

        This method is used after a previous receive call has populated
        ``self.batch_router[tag]`` with batch indices from incoming messages.
        The outgoing ``data`` is split according to ``split_sizes`` and each shard is
        sent to the rank encoded in the corresponding recorded batch index.

        Each outgoing channel item is a dict with ``batch_index`` (recorded batch
        index) and ``batch`` (payload shard).

        After all shards are queued, the recorded batch indices for ``tag`` are cleared
        to avoid reusing stale routes.

        Args:
            group_name: Destination worker group name.
            channel: Channel used to send the split payloads.
            data: Payload to split and send.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag whose recorded batch indices should be consumed.
            split_fn: Optional custom splitter. If omitted, ``split_batch`` is used.
            split_sizes: Batch sizes used to split ``data``. Must have the same length
                as ``self.batch_router[tag]``.

        Returns:

            AsyncRouteWork wrapping the async channel put operations.
        """
        from rlinf.scheduler import build_send_key, split_batch
        from rlinf.scheduler.collective import AsyncRouteWork

        assert tag in self.batch_router, (
            f"{tag=} need to be already in the batch_router"
        )

        assert len(self.batch_router[tag]) > 0, f"{self.batch_router[tag]=} is empty"
        assert len(self.batch_router[tag]) == len(split_sizes), (
            f"{self.batch_router[tag]=} length should equal {split_sizes=} length"
        )

        payloads = (
            split_fn(data, split_sizes)
            if split_fn is not None
            else split_batch(data, split_sizes)
        )

        works = []
        for i, payload in enumerate(payloads):
            batch_index = self.batch_router[tag][i]
            send_rank, _, mode, _ = split_channel_message(batch_index)
            # After enabling env_decoupled_mode, the data sending format is as follows:
            # {
            #     "batch_index": batch_index,
            #     "batch": batch,
            # }
            # The batch_index is the index of the batch in the data.
            # The batch is the data to send.
            # batch_index: {send_rank}_{batch_idx}_{mode}_{tag}
            # The send_rank is the rank of the worker that originally sent the data.
            # The batch_idx is the index of the batch in the data.
            # The tag is the tag of the data.
            senditem = {
                "batch_index": batch_index,
                "batch": payload,
            }
            key = build_send_key(
                src_group_name=self.worker_address.root_group_name,
                dst_group_name=group_name,
                src_rank=None,
                dst_rank=send_rank,
                tag=tag if mode is None else f"{mode}_{tag}",
                route_key=route_key,
            )
            work = channel.put(
                item=senditem,
                key=key,
                async_op=True,
            )
            works.append(work)

        # clear the batch_router for the next send
        self.batch_router[tag] = []
        return AsyncRouteWork(works, lambda _: None)

    def update_dagger_beta(self):
        if self.expert_model is None or not self.enable_dagger:
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.model_cfg.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.OPENPI_PYTORCH,
            SupportedModel.EVO1,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
            SupportedModel.CFG_MODEL,
        ]:
            if self.enable_dagger:
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}
            if mode == "eval" and bool(
                self.model_cfg.get("rl_head_config", {}).get(
                    "eval_audit_metadata", False
                )
            ):
                kwargs["return_semantic_features"] = True

        if SupportedModel(self.model_cfg.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.algorithm_cfg.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None and self.enable_dagger:
            # training with expert model. Beta-probability acting.
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_label_flag = True
            else:
                prediction_started = time.perf_counter()
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                prediction_s = time.perf_counter() - prediction_started
                semantic_fetch_s = getattr(
                    self.hf_model, "_last_semantic_fetch_s", None
                )
                if semantic_fetch_s is not None:
                    tag = "predict/semantic_fetch"
                    self._timer_metrics[tag] = self._timer_metrics.get(
                        tag, 0.0
                    ) + float(semantic_fetch_s)
                    self._record_fdvla_profile(
                        actions=actions,
                        result=result,
                        prediction_s=prediction_s,
                        semantic_fetch_s=float(semantic_fetch_s),
                    )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and self.enable_dagger  # only re-label in DAgger mode
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs["model_action"]
                expert_action = expert_forward_inputs["action"]
                if expert_target is not None:
                    result["forward_inputs"]["action"] = expert_action
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def _predict_microbatched(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run one policy replica on ordered row microbatches and merge on CPU."""
        batch_size = self._infer_env_batch_size(env_obs)
        micro_batch_size = self.inference_micro_batch_size
        if micro_batch_size <= 0 or batch_size <= micro_batch_size:
            return self.predict(env_obs, mode=mode)
        if self.enable_dagger:
            raise RuntimeError(
                "rollout inference microbatching is not supported with DAgger"
            )

        split_sizes = [micro_batch_size] * (batch_size // micro_batch_size)
        remainder = batch_size % micro_batch_size
        if remainder:
            split_sizes.append(remainder)

        allowed_result_keys = {
            "prev_logprobs",
            "prev_values",
            "forward_inputs",
            "intervene_flags",
            "expert_label_flag",
        }
        rollout_results = []
        expert_label_flags = []
        for obs_shard in split_dict(env_obs, split_sizes):
            actions, result = self.predict(obs_shard, mode=mode)
            unexpected_keys = set(result) - allowed_result_keys
            if unexpected_keys:
                raise RuntimeError(
                    "Cannot merge microbatched rollout result keys: "
                    f"{sorted(unexpected_keys)}"
                )
            rollout_results.append(
                RolloutResult(
                    actions=actions,
                    prev_logprobs=result.get("prev_logprobs"),
                    prev_values=result.get("prev_values"),
                    intervene_flags=result.get("intervene_flags"),
                    forward_inputs=result.get("forward_inputs", {}),
                )
            )
            expert_label_flags.append(bool(result.get("expert_label_flag", False)))

        merged = RolloutResult.merge_rollout_results(rollout_results)
        return merged.actions, {
            "prev_logprobs": merged.prev_logprobs,
            "prev_values": merged.prev_values,
            "forward_inputs": merged.forward_inputs,
            "intervene_flags": merged.intervene_flags,
            "expert_label_flag": any(expert_label_flags),
        }

    def _predict_rollout_actions(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        final_obs: dict[str, Any] | None = None,
        rlt_switch_flags: torch.Tensor | None = None,
        intervene_requested: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.rlt_feature_model is not None:
            return predict_rlt_actions(
                policy_model=self.hf_model,
                feature_model=self.rlt_feature_model,
                rlt_route=self.rlt_route,
                env_obs=env_obs,
                final_obs=final_obs,
                mode=mode,
                version=self.version,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=self.expert_model,
            )
        return self._predict_microbatched(env_obs, mode)

    def _build_rollout_result(
        self,
        actions: torch.Tensor,
        result: dict[str, Any],
        *,
        final_obs: dict[str, Any] | None = None,
    ) -> RolloutResult:
        intervene_flags = result.get("intervene_flags")
        if (
            intervene_flags is None
            and self.enable_dagger
            and result.get("expert_label_flag", False)
        ):
            intervene_flags = torch.full(
                (actions.shape[0], self.model_cfg.num_action_chunks),
                True,
                dtype=torch.bool,
                device=actions.device,
            )
        return RolloutResult(
            actions=actions,
            prev_logprobs=result["prev_logprobs"] if self.collect_prev_infos else None,
            prev_values=result["prev_values"] if self.collect_prev_infos else None,
            bootstrap_values=self.get_bootstrap_values(final_obs),
            intervene_flags=intervene_flags,
            forward_inputs=result["forward_inputs"],
            versions=torch.full_like(
                result["prev_logprobs"],
                float(self.version),
                dtype=torch.float32,
            ),
        )

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if not (
            hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head")
        ):
            return None
        with torch.no_grad():
            actions, result = self._predict_rollout_actions(final_obs)
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""

        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.hf_model.state_dict(),
                recv=recv_func,
                send=send_func,
            )

        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(applied_version)

        gc.collect()
        self.torch_platform.empty_cache()

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        self.update_dagger_beta()
        for _ in range(self.n_train_chunk_steps):
            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_from(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="train_rollout_results",
                    route_key=stage_id,
                    async_op=True,
                    batch_size=self.train_batch_size,
                    merge_fn=self._merge_obs_batches,
                    infer_batch_size_fn=self._infer_env_batch_size,
                ).async_wait()
                actions, result = self._predict_rollout_actions(
                    env_output["obs"],
                    final_obs=env_output.get("final_obs", None),
                    rlt_switch_flags=env_output.get("rlt_switch_flags", None),
                    intervene_requested=env_output.get("intervene_flags", None),
                )

                rollout_result = self._build_rollout_result(
                    actions,
                    result,
                    final_obs=env_output.get("final_obs", None),
                )
                self.send_to(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=rollout_result,
                    tag="train_rollout_results",
                    route_key=stage_id,
                    async_op=True,
                    batch_size=self.train_batch_size,
                    split_fn=self._split_rollout_result,
                )
        for stage_id in range(self.num_pipeline_stages):
            env_output = await self.recv_from(
                group_name=self.cfg.env.group_name,
                channel=input_channel,
                tag="train_rollout_results",
                route_key=stage_id,
                async_op=True,
                batch_size=self.train_batch_size,
                merge_fn=self._merge_obs_batches,
                infer_batch_size_fn=self._infer_env_batch_size,
            ).async_wait()
            actions, result = self._predict_rollout_actions(
                env_output["obs"],
                final_obs=env_output.get("final_obs", None),
                rlt_switch_flags=env_output.get("rlt_switch_flags", None),
                intervene_requested=env_output.get("intervene_flags", None),
            )

            if self.enable_opd:
                # OPD keeps this path separate to retain student action tokens for post-rollout teacher logprobs.
                rollout_result = self._build_rollout_result(
                    actions,
                    result,
                    final_obs=env_output.get("final_obs", None),
                )
            else:
                rollout_result = RolloutResult(
                    actions=actions,
                    prev_values=(
                        result["prev_values"] if self.collect_prev_infos else None
                    ),
                    bootstrap_values=self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    ),
                    forward_inputs=(
                        result["forward_inputs"]
                        if self.rlt_feature_model is not None
                        else {}
                    ),
                )
            self.send_to(
                group_name=self.cfg.env.group_name,
                channel=output_channel,
                data=rollout_result,
                tag="train_rollout_results",
                route_key=stage_id,
                async_op=True,
                batch_size=self.train_batch_size,
                split_fn=self._split_rollout_result,
            )

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(input_channel, output_channel)

        if self.enable_offload:
            self.offload_model()

    @Worker.timer("evaluate")
    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        if self.env_decoupled_mode:
            while True:
                (
                    env_output,
                    split_sizes,
                ) = await self.recv_from_and_record_batch_routes_with_timeout(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="rollout_results",
                    batch_size=self.eval_batch_size,
                    merge_fn=self._merge_obs_batches,
                    infer_batch_size_fn=self._infer_env_batch_size,
                    timeout_time=0.02,
                    recv_queue_size=self.rollout_queue_size,
                )
                actions, result = self._predict_rollout_actions(
                    env_output["obs"],
                    mode="eval",
                    final_obs=env_output.get("final_obs", None),
                    rlt_switch_flags=env_output.get("rlt_switch_flags", None),
                    intervene_requested=env_output.get("intervene_flags", None),
                )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().contiguous()
                rollout_result = RolloutResult(
                    actions=actions,
                    forward_inputs=result.get("forward_inputs", {}),
                )
                self.send_to_recorded_batch_routes(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=rollout_result,
                    tag="rollout_results",
                    split_fn=self._split_rollout_result,
                    split_sizes=split_sizes,
                )
        else:
            for _ in tqdm(
                range(self.eval_rollout_epoch),
                desc="Evaluating Rollout Epochs",
                disable=(self._rank != 0),
            ):
                for _ in range(self.n_eval_chunk_steps):
                    for stage_id in range(self.num_pipeline_stages):
                        env_output = await self.recv_from(
                            group_name=self.cfg.env.group_name,
                            channel=input_channel,
                            tag="eval_rollout_results",
                            route_key=stage_id,
                            async_op=True,
                            batch_size=self.eval_batch_size,
                            merge_fn=self._merge_obs_batches,
                            infer_batch_size_fn=self._infer_env_batch_size,
                        ).async_wait()
                        actions, result = self._predict_rollout_actions(
                            env_output["obs"],
                            mode="eval",
                            final_obs=env_output.get("final_obs", None),
                            rlt_switch_flags=env_output.get("rlt_switch_flags", None),
                            intervene_requested=env_output.get("intervene_flags", None),
                        )
                        if isinstance(actions, torch.Tensor):
                            actions = actions.detach().cpu().contiguous()
                        rollout_result = RolloutResult(
                            actions=actions,
                            forward_inputs=result.get("forward_inputs", {}),
                        )
                        self.send_to(
                            group_name=self.cfg.env.group_name,
                            channel=output_channel,
                            data=rollout_result,
                            tag="eval_rollout_results",
                            route_key=stage_id,
                            async_op=True,
                            batch_size=self.eval_batch_size,
                            split_fn=self._split_rollout_result,
                        )

            if self.enable_offload:
                self.offload_model()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.to("cpu")
        if self.expert_model is not None:
            self.expert_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.to(self.device)
        if self.expert_model is not None:
            self.expert_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    def _merge_optional_flag_tensors(
        self,
        obs_dicts: list[dict[str, Any]],
        flags_list: list[torch.Tensor | None],
    ) -> torch.Tensor | None:
        if not any(flags is not None for flags in flags_list):
            return None
        ref_flags = next(flags for flags in flags_list if flags is not None)
        filled_flags = []
        for obs_dict, flags in zip(obs_dicts, flags_list):
            if flags is None:
                batch_size = self._infer_env_batch_size(obs_dict)
                fill_shape = (batch_size, *ref_flags.shape[1:])
                filled_flags.append(torch.zeros(fill_shape, dtype=ref_flags.dtype))
            else:
                filled_flags.append(flags)
        return torch.cat(filled_flags, dim=0)

    def _merge_obs_batches(self, obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]
        rlt_switch_flags_list = [
            obs_batch.get("rlt_switch_flags", None) for obs_batch in obs_batches
        ]
        intervene_flags_list = [
            obs_batch.get("intervene_flags", None) for obs_batch in obs_batches
        ]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = []
            for obs_dict, final_obs in zip(obs_dicts, final_obs_list):
                if final_obs is None:
                    final_obs_or_obs.append(obs_dict)
                    continue
                final_obs_with_metadata = dict(final_obs)
                for key, value in obs_dict.items():
                    if key.startswith("__rlinf_semantic_"):
                        final_obs_with_metadata.setdefault(key, value)
                final_obs_or_obs.append(final_obs_with_metadata)
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {
            "obs": merged_obs,
            "final_obs": merged_final_obs,
            "rlt_switch_flags": self._merge_optional_flag_tensors(
                obs_dicts, rlt_switch_flags_list
            ),
            "intervene_flags": self._merge_optional_flag_tensors(
                obs_dicts, intervene_flags_list
            ),
        }

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_intervene_flags = _split_optional_tensor(rollout_result.intervene_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                    if value is not None
                }
                for idx in range(len(sizes))
            ]
        )

        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                intervene_flags=split_intervene_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    def set_global_step(self, global_step: int):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
