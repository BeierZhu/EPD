"""
Trajectory collection utilities for PPO-based EPD solver training.

This module bridges the policy network (Stage 3) with the original
EPD solver by:
  * Managing prompt/seed streams for reproducible rollouts.
  * Sampling entire EPD parameter tables from the policy.
  * Adapting sampled tables to the predictor interface expected by
    `solvers.epd_sampler`.
  * Invoking the diffusion model to generate images and caching all
    metadata required by subsequent PPO stages.

Only non-invasive additions are made so the legacy EPD codebase
remains untouched.
"""

import csv
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from solvers import epd_sampler
from torch_utils.download_util import check_file_by_key

from .policy import EPDParamPolicy, PolicyOutput, PolicySample


# ---------------------------------------------------------------------------
# Configuration structures


class RLRunnerConfig(object):
    """Static configuration for EPD rollouts."""

    def __init__(
        self,
        policy,
        net,
        num_steps,
        num_points,
        device,
        guidance_type="cfg",
        guidance_rate=7.5,
        schedule_type="discrete",
        schedule_rho=1.0,
        dataset_name="ms_coco",
        precision=torch.float32,
        prompt_csv=None,
        rloo_k=1,
        rng_seed=None,
        verbose=False,
        model_source="ldm",
        backend="ldm",
        backend_config=None,
        rank=0,
        world_size=1,
        sigma_min=None,
        sigma_max=None,
    ):
        self.policy = policy
        self.net = net
        self.num_steps = num_steps
        self.num_points = num_points
        self.device = device
        self.guidance_type = guidance_type
        self.guidance_rate = guidance_rate
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.dataset_name = dataset_name
        self.precision = precision
        self.prompt_csv = Path(prompt_csv) if prompt_csv is not None else None
        self.rloo_k = rloo_k
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.model_source = model_source
        self.backend = backend
        cfg = backend_config if backend_config is not None else {}
        if isinstance(cfg, dict):
            self.backend_config = dict(cfg)
        else:
            try:
                self.backend_config = dict(cfg)  # type: ignore[arg-type]
            except Exception:
                self.backend_config = {}
        self.rank = rank
        self.world_size = max(1, world_size)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max


class RolloutBatch(object):
    """Container for a rollout batch returned by the runner."""

    def __init__(
        self,
        images,
        prompts,
        seeds,
        policy_output,
        policy_sample,
        log_prob,
        entropy_pos,
        entropy_weight,
        step_indices,
        latents,
        metadata=None,
    ):
        self.images = images
        self.prompts = prompts
        self.seeds = seeds
        self.policy_output = policy_output
        self.policy_sample = policy_sample
        self.log_prob = log_prob
        self.entropy_pos = entropy_pos
        self.entropy_weight = entropy_weight
        self.step_indices = step_indices
        self.latents = latents
        self.metadata = metadata if metadata is not None else {}


# ---------------------------------------------------------------------------
# Prompt / seed helpers


def _load_prompts(prompt_csv: Optional[Path]) -> List[str]:
    if prompt_csv is not None:
        path = Path(prompt_csv)
    else:
        prompt_path, _ = check_file_by_key("prompts")
        path = Path(prompt_path)

    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                prompts.append(text)
    if not prompts:
        raise RuntimeError(f"No prompts were loaded from {path}.")
    return prompts


def _repeat_prompts(prompts: Sequence[str], rloo_k: int) -> List[str]:
    if rloo_k <= 1:
        return list(prompts)
    expanded: List[str] = []
    for p in prompts:
        expanded.extend([p] * rloo_k)
    return expanded


# ---------------------------------------------------------------------------
# Predictor adapter


class _PolicyTableModule(nn.Module):
    """Stores per-step tables to mimic the EPD predictor interface."""

    def __init__(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        num_points: int,
        num_steps: int,
        schedule_type: str,
        schedule_rho: float,
        dataset_name: str,
        guidance_type: str,
        guidance_rate: float,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.dataset_name = dataset_name
        self.guidance_type = guidance_type
        self.guidance_rate = guidance_rate
        self.afs = False
        self.scale_dir = 0.0
        self.scale_time = 0.0
        self.predict_x0 = False
        self.lower_order_final = True
        self.fcn = False

        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

    def forward(self, batch_size: int, step_idx: int):
        pos = self.positions[:, step_idx].unsqueeze(-1).unsqueeze(-1)
        w = self.weights[:, step_idx].unsqueeze(-1).unsqueeze(-1)
        ones = torch.ones_like(pos)
        return pos, ones, ones, w


class PolicyPredictorAdapter(nn.Module):
    """
    Emulate the distilled EPD predictor interface using policy samples.
    """

    def __init__(
        self,
        policy_sample: PolicySample,
        config: RLRunnerConfig,
    ) -> None:
        super().__init__()
        self.num_points = config.num_points
        self.num_steps = config.num_steps
        self.guidance_type = config.guidance_type
        self.guidance_rate = config.guidance_rate
        self.dataset_name = config.dataset_name
        self.schedule_type = config.schedule_type
        self.schedule_rho = config.schedule_rho
        self.afs = False
        self.scale_dir = 0.0
        self.scale_time = 0.0
        self.predict_x0 = False
        self.lower_order_final = True
        self.fcn = False

        positions = policy_sample.positions
        weights = policy_sample.weights
        if positions.dim() == 2:
            positions = positions.unsqueeze(1)
        if weights.dim() == 2:
            weights = weights.unsqueeze(1)

        positions = positions.to(dtype=torch.float32)
        weights = weights.to(dtype=torch.float32)

        self.tables = _PolicyTableModule(
            positions,
            weights,
            num_points=config.num_points,
            num_steps=config.num_steps,
            schedule_type=config.schedule_type,
            schedule_rho=config.schedule_rho,
            dataset_name=config.dataset_name,
            guidance_type=config.guidance_type,
            guidance_rate=config.guidance_rate,
        )

        self.module = self.tables

    def forward(self, batch_size: int, step_idx: int):
        return self.tables(batch_size, step_idx)


# ---------------------------------------------------------------------------
# Rollout runner


class EPDRolloutRunner:
    """Generate PPO rollouts by combining the policy and the EPD solver."""

    def __init__(self, config: RLRunnerConfig):
        self.config = config
        self.policy = config.policy
        self._policy_module = (
            config.policy.module if hasattr(config.policy, "module") else config.policy
        )
        self.net = config.net
        self.device = config.device
        self.rank = getattr(config, "rank", 0)
        self.world_size = getattr(config, "world_size", 1)
        self.backend = getattr(config, "backend", getattr(self.net, "backend", "ldm"))
        cfg = getattr(config, "backend_config", {})
        self.backend_config = dict(cfg) if isinstance(cfg, dict) else {}

        self.prompts = _load_prompts(config.prompt_csv)
        self.prompt_cursor = self.rank % len(self.prompts)
        self.rloo_k = max(1, config.rloo_k)

        seed_value = config.rng_seed if config.rng_seed is not None else int(time.time())
        self.seed_rng = np.random.RandomState(seed_value)

        self.step_indices = torch.arange(config.num_steps - 1, dtype=torch.long, device=self.device)

    def set_prompt_index(self, index: int) -> None:
        self.prompt_cursor = index % len(self.prompts)

    def _sample_prompts_and_seeds(self, batch_size: int) -> Tuple[List[str], List[int]]:
        if batch_size % self.rloo_k != 0:
            raise RuntimeError(
                f"batch_size ({batch_size}) must be divisible by rloo_k ({self.rloo_k})."
            )
        prompts: List[str] = []
        seeds: List[int] = []
        unique_prompts = batch_size // self.rloo_k
        start_index = self.prompt_cursor
        for idx in range(unique_prompts):
            prompt_index = (start_index + idx * self.world_size) % len(self.prompts)
            prompt = self.prompts[prompt_index]
            for _ in range(self.rloo_k):
                prompts.append(prompt)
                seed = int(self.seed_rng.randint(0, 2**32 - 1))
                seeds.append(seed)
        self.prompt_cursor = (start_index + unique_prompts * self.world_size) % len(self.prompts)
        return prompts, seeds

    def _prepare_latents(self, seeds: Sequence[int], shape: Tuple[int, ...]) -> torch.Tensor:
        latents = []
        for seed in seeds:
            g = torch.Generator(device=self.device).manual_seed(seed)
            latents.append(torch.randn(shape[1:], generator=g, device=self.device))
        return torch.stack(latents, dim=0)

    def _policy_step(self, batch_size: int) -> Tuple[PolicyOutput, PolicySample]:
        step_idx = self.step_indices
        step_idx_batch = step_idx.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
        policy_output = self.policy(step_idx_batch)
        intervals = self.config.num_steps - 1
        policy_output.alpha_pos = policy_output.alpha_pos.reshape(batch_size, intervals, self.config.num_points + 1)
        policy_output.alpha_weight = policy_output.alpha_weight.reshape(batch_size, intervals, self.config.num_points)
        policy_output.log_alpha_pos = policy_output.log_alpha_pos.reshape(batch_size, intervals, self.config.num_points + 1)
        policy_output.log_alpha_weight = policy_output.log_alpha_weight.reshape(batch_size, intervals, self.config.num_points)

        flattened_output = PolicyOutput(
            alpha_pos=policy_output.alpha_pos.reshape(-1, self.config.num_points + 1),
            alpha_weight=policy_output.alpha_weight.reshape(-1, self.config.num_points),
            log_alpha_pos=policy_output.log_alpha_pos.reshape(-1, self.config.num_points + 1),
            log_alpha_weight=policy_output.log_alpha_weight.reshape(-1, self.config.num_points),
        )

        sample = self._policy_module.sample_table(flattened_output)
        intervals = self.config.num_steps - 1
        sample.positions = sample.positions.reshape(batch_size, intervals, self.config.num_points)
        sample.weights = sample.weights.reshape(batch_size, intervals, self.config.num_points)
        sample.segments = sample.segments.reshape(batch_size, intervals, self.config.num_points + 1)
        sample.log_prob = sample.log_prob.reshape(batch_size, intervals).sum(dim=-1)
        sample.entropy_pos = sample.entropy_pos.reshape(batch_size, intervals).sum(dim=-1)
        sample.entropy_weight = sample.entropy_weight.reshape(batch_size, intervals).sum(dim=-1)

        return policy_output, sample

    def _prepare_conditions(self, prompts: Sequence[str]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        condition = None
        unconditional = None
        class_labels = None

        backend_type = getattr(self.net, "backend", self.backend)
        if backend_type == "sd3":
            prompts_list = list(prompts)
            if self.config.guidance_rate == 1.0:
                negative_prompt = None
            else:
                base_negative = self.backend_config.get("negative_prompt", "")
                if isinstance(base_negative, list):
                    if len(base_negative) != len(prompts_list):
                        raise RuntimeError("Length of backend_config.negative_prompt must match batch size.")
                    negative_prompt = base_negative
                else:
                    negative_prompt = [str(base_negative)] * len(prompts_list)
            condition = self.net.prepare_condition(
                prompt=prompts_list,
                negative_prompt=negative_prompt,
                guidance_scale=self.config.guidance_rate,
            )
            return condition, None, None

        if getattr(self.net, "label_dim", 0):
            if self.config.model_source == "ldm":
                prompts_list = list(prompts)
                condition = self.net.model.get_learned_conditioning(prompts_list)
                if self.config.guidance_rate != 1.0:
                    unconditional = self.net.model.get_learned_conditioning(len(prompts_list) * [""])
                condition = condition.to(self.device)
                if unconditional is not None:
                    unconditional = unconditional.to(self.device)
            elif self.config.model_source == "adm":
                raise NotImplementedError("ADM model_source conditioning not yet supported in runner")

        return condition, unconditional, class_labels

    def rollout(self, batch_size: int) -> RolloutBatch:
        prompts, seeds = self._sample_prompts_and_seeds(batch_size)

        latents_shape = (batch_size, self.net.img_channels, self.net.img_resolution, self.net.img_resolution)
        latents = self._prepare_latents(seeds, latents_shape)

        policy_output, policy_sample = self._policy_step(batch_size)

        predictor = PolicyPredictorAdapter(policy_sample, self.config).to(self.device)

        condition, unconditional_condition, class_labels = self._prepare_conditions(prompts)

        sigma_min = self.config.sigma_min if self.config.sigma_min is not None else getattr(self.net, "sigma_min", None)
        sigma_max = self.config.sigma_max if self.config.sigma_max is not None else getattr(self.net, "sigma_max", None)
        if sigma_min is None or sigma_max is None:
            raise RuntimeError("sigma_min and sigma_max must be defined for the selected backend.")

        start_time = time.time()
        try:
            with torch.no_grad():
                images, _ = epd_sampler(
                    net=self.net,
                    latents=latents,
                    class_labels=class_labels,
                    condition=condition,
                    unconditional_condition=unconditional_condition,
                    num_steps=self.config.num_steps,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    schedule_type=self.config.schedule_type,
                    schedule_rho=self.config.schedule_rho,
                    guidance_type=self.config.guidance_type,
                    guidance_rate=self.config.guidance_rate,
                    predictor=predictor,
                    afs=False,
                    scale_dir=0,
                    scale_time=0,
                    train=False,
                    verbose=self.config.verbose,
                )
        except RuntimeError as err:
            torch.cuda.empty_cache()
            metadata = {
                "status": "error",
                "exception": repr(err),
                "prompts": prompts,
                "seeds": seeds,
            }
            raise RuntimeError(metadata) from err

        duration = time.time() - start_time

        images = images.to(dtype=self.config.precision)

        metadata = {
            "status": "ok",
            "duration": duration,
            "num_prompts": len(prompts),
        }

        return RolloutBatch(
            images=images,
            prompts=prompts,
            seeds=seeds,
            policy_output=policy_output,
            policy_sample=policy_sample,
            log_prob=policy_sample.log_prob,
            entropy_pos=policy_sample.entropy_pos,
            entropy_weight=policy_sample.entropy_weight,
            step_indices=self.step_indices,
            latents=latents,
            metadata=metadata,
        )
