"""
Stage 6: PPO trainer integrating policy, rollout runner, and HPS rewards.

This module provides a lightweight implementation of PPO with RLOO
advantage estimation tailored to the EPD solver setting. The design takes
inspiration from TPDM's trainer but avoids external dependencies so it can
operate entirely within the existing RLEPD codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.optim import AdamW, Optimizer
from types import SimpleNamespace
from torch.nn.parallel import DistributedDataParallel

from .policy import EPDParamPolicy
from .rl_runner import EPDRolloutRunner, RolloutBatch
from .reward_base import RewardAdapter, RewardMetadata


@dataclass
class PPOTrainerConfig:
    """Hyper-parameters for PPO training."""

    device: torch.device
    rollout_batch_size: int
    rloo_k: int
    ppo_epochs: int = 2
    minibatch_size: int = 4
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    clip_range: float = 0.2
    kl_coef: float = 0.05
    entropy_coef: float = 0.0
    normalize_advantages: bool = True
    max_grad_norm: Optional[float] = 1.0
    decode_rgb: bool = True
    image_value_range: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive.")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive.")
        if self.rloo_k <= 0:
            raise ValueError("rloo_k must be positive.")


class PPOTrainer:
    """
    Perform PPO updates over EPD parameter tables using RLOO advantages.
    """

    def __init__(
        self,
        policy: EPDParamPolicy | DistributedDataParallel,
        runner: EPDRolloutRunner,
        reward: RewardAdapter,
        config: PPOTrainerConfig,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        if isinstance(policy, DistributedDataParallel):
            self.policy = policy
            self._policy_module = policy.module
        else:
            self.policy = policy.to(config.device)
            self._policy_module = self.policy
        self.runner = runner
        self.reward = reward
        self.config = config
        self.scheduler = scheduler

        self.policy.train()

        if optimizer is None:
            optimizer = AdamW(
                self.policy.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        self.optimizer = optimizer

        self._device = config.device
        self._step = 0

        runner_k = getattr(getattr(self.runner, "config", SimpleNamespace()), "rloo_k", config.rloo_k)
        if runner_k != config.rloo_k:
            raise ValueError("PPOTrainerConfig.rloo_k must match the rollout runner configuration.")

    def train_step(self) -> Dict[str, float]:
        """
        Run a single PPO iteration: collect rollouts, score rewards, and
        update the policy parameters.
        """

        batch = self.runner.rollout(self.config.rollout_batch_size)
        rewards, reward_meta = self._compute_rewards(batch)
        advantages = self._compute_advantages(rewards)
        stats = self._ppo_update(batch, advantages)

        stats.update(self._reward_logging_stats(rewards, reward_meta))
        stats["step"] = float(self._step)
        self._step += 1
        return stats

    # ------------------------------------------------------------------ #
    # Rollout utilities
    # ------------------------------------------------------------------ #

    def _compute_rewards(self, batch: RolloutBatch) -> Tuple[torch.Tensor, RewardMetadata]:
        images = self._prepare_images(batch)
        result = self.reward.score_tensor(images, batch.prompts, return_metadata=True)
        if isinstance(result, tuple):
            scores, metadata = result
        else:
            scores = result
            metadata = {}
        rewards = scores.to(self._device).detach()
        return rewards, metadata

    def _reward_logging_stats(self, rewards: torch.Tensor, metadata: RewardMetadata) -> Dict[str, float]:
        mean_value = rewards.mean().item()
        std_value = rewards.std(unbiased=False).item() if rewards.numel() > 1 else 0.0
        stats = {
            "mixed_reward_mean": mean_value,
            "mixed_reward_std": std_value,
        }

        raw_scores: Dict[str, torch.Tensor] = {}
        if isinstance(metadata, dict):
            raw_candidate = metadata.get("raw_scores", {})
            if isinstance(raw_candidate, dict):
                raw_scores = {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in raw_candidate.items()
                }

        def metric_mean(name: str, fallback: Optional[float] = None) -> float:
            value = raw_scores.get(name)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return float(value.mean().item())
            if fallback is not None:
                return fallback
            return float("nan")

        stats["hps_mean"] = metric_mean("hps", fallback=mean_value if not raw_scores else None)
        stats["aesthetic_mean"] = metric_mean("aesthetic")
        stats["clip_mean"] = metric_mean("clip")
        stats["imagereward_mean"] = metric_mean("imagereward")
        stats["pickscore_mean"] = metric_mean("pickscore")
        return stats

    def _prepare_images(self, batch: RolloutBatch) -> torch.Tensor:
        images = batch.images
        if self.config.decode_rgb:
            backend = getattr(getattr(self, "runner", None), "net", None)
            # Prefer SD3-style backend.vae_decode; fall back to LDM decode_first_stage.
            decoder = getattr(backend, "vae_decode", None)
            if decoder is None:
                decoder = getattr(getattr(backend, "model", None), "decode_first_stage", None)

            if callable(decoder):
                with torch.no_grad():
                    images = decoder(images)
        images = torch.clamp((images + 1.0) / 2.0, self.config.image_value_range[0], self.config.image_value_range[1])
        return images.to(self.reward.config.device if hasattr(self.reward, "config") else self._device)

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        k = self.config.rloo_k
        if k == 1:
            advantages = rewards.clone()
        else:
            num_prompts = rewards.numel() // k
            reshaped = rewards.view(k, num_prompts)
            baseline = (reshaped.sum(dim=0, keepdim=True) - reshaped) / (k - 1)
            advantages = (reshaped - baseline).reshape(-1)

        advantages = advantages.to(self._device)
        if self.config.normalize_advantages and advantages.numel() > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
            advantages = (advantages - adv_mean) / adv_std
        return advantages.detach()

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #

    def _ppo_update(self, batch: RolloutBatch, advantages: torch.Tensor) -> Dict[str, float]:
        policy_sample = batch.policy_sample
        step_indices = batch.step_indices.to(self._device)
        segments = policy_sample.segments.to(self._device).detach()
        weights = policy_sample.weights.to(self._device).detach()
        old_log_prob = batch.log_prob.to(self._device).detach()
        advantages = advantages.detach()

        batch_size = old_log_prob.shape[0]
        intervals = segments.shape[1]

        base_alpha_pos = torch.exp(
            self._policy_module.base_log_alpha_pos.index_select(0, step_indices)
        )
        base_alpha_weight = torch.exp(
            self._policy_module.base_log_alpha_weight.index_select(0, step_indices)
        )

        stats_accum = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "kl": 0.0,
            "ratio": 0.0,
            "grad_norm": 0.0,
        }
        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self._device)
            for start in range(0, batch_size, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, batch_size)
                mb_indices = permutation[start:end]
                if mb_indices.numel() == 0:
                    continue
                update_stats = self._ppo_minibatch_update(
                    mb_indices,
                    step_indices,
                    segments,
                    weights,
                    base_alpha_pos,
                    base_alpha_weight,
                    old_log_prob,
                    advantages,
                )
                for key, value in update_stats.items():
                    stats_accum[key] += value
                num_updates += 1

        if num_updates > 0:
            for key in stats_accum:
                stats_accum[key] /= num_updates
        return stats_accum

    def _ppo_minibatch_update(
        self,
        mb_indices: torch.Tensor,
        step_indices: torch.Tensor,
        segments: torch.Tensor,
        weights: torch.Tensor,
        base_alpha_pos: torch.Tensor,
        base_alpha_weight: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        mb_size = mb_indices.numel()
        intervals = segments.shape[1]
        num_points = weights.shape[-1]

        mb_segments = segments.index_select(0, mb_indices)
        mb_weights = weights.index_select(0, mb_indices)
        mb_advantages = advantages.index_select(0, mb_indices)
        mb_old_logprob = old_log_prob.index_select(0, mb_indices)

        repeated_indices = step_indices.unsqueeze(0).repeat(mb_size, 1).reshape(-1)
        flat_output = self.policy(repeated_indices)

        flat_segments = mb_segments.reshape(-1, num_points + 1)
        flat_weights = mb_weights.reshape(-1, num_points)

        new_logprob_flat = self._policy_module.log_prob(flat_output, flat_segments, flat_weights)
        new_logprob = new_logprob_flat.view(mb_size, intervals).sum(dim=-1)

        ratio = torch.exp(new_logprob - mb_old_logprob)

        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
        policy_loss_unclipped = -mb_advantages * ratio
        policy_loss_clipped = -mb_advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

        ref_alpha_pos = base_alpha_pos.unsqueeze(0).repeat(mb_size, 1, 1).reshape(-1, num_points + 1)
        ref_alpha_weight = base_alpha_weight.unsqueeze(0).repeat(mb_size, 1, 1).reshape(-1, num_points)
        kl_flat = self._policy_module._dirichlet_kl(flat_output.alpha_pos, ref_alpha_pos) + self._policy_module._dirichlet_kl(
            flat_output.alpha_weight, ref_alpha_weight
        )
        kl = kl_flat.view(mb_size, intervals).sum(dim=-1)
        kl_mean = kl.mean()

        entropy_flat = self._policy_module.entropy(flat_output)
        entropy = entropy_flat.view(mb_size, intervals).sum(dim=-1)
        entropy_mean = entropy.mean()

        loss = policy_loss + self.config.kl_coef * kl_mean - self.config.entropy_coef * entropy_mean

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        grad_norm_value = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else (
            float(grad_norm) if grad_norm is not None else 0.0
        )

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "kl": float(kl_mean.item()),
            "ratio": float(ratio.mean().item()),
            "grad_norm": grad_norm_value,
        }
