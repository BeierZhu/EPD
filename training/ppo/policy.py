"""
Policy network for PPO-based fine-tuning of EPD parameter tables.

The network maps each coarse diffusion step to Dirichlet concentration
vectors describing:
    * Position segments (K+1 values that integrate to 1.0 and recover
      monotonically increasing intermediate locations `r` via cumsum).
    * Gradient weights (K values on the simplex).

Cold-start tables (see `cold_start.py`) provide per-step Dirichlet
concentrations which are stored as buffers and used as the policy's
reference/initialisation. The learnable network outputs residuals on
log-concentration space so that, at initialisation, the policy mean
exactly matches the distilled EPD table.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


@dataclass
class PolicyOutput:
    """Container for per-step Dirichlet concentrations."""

    alpha_pos: torch.Tensor  # shape: (batch, num_points + 1)
    alpha_weight: torch.Tensor  # shape: (batch, num_points)
    log_alpha_pos: torch.Tensor  # auxiliary tensor (same shape as alpha_pos)
    log_alpha_weight: torch.Tensor  # auxiliary tensor (same shape as alpha_weight)


@dataclass
class PolicySample:
    """Result of sampling a parameter table from the policy."""

    positions: torch.Tensor  # shape: (batch, num_points)
    weights: torch.Tensor  # shape: (batch, num_points)
    segments: torch.Tensor  # sampled simplex segments for positions
    log_prob: torch.Tensor  # shape: (batch,)
    entropy_pos: torch.Tensor  # shape: (batch,)
    entropy_weight: torch.Tensor  # shape: (batch,)


class ResidualBlock(nn.Module):
    """Layer-norm + SiLU + linear residual block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = F.silu(x)
        x = self.linear(x)
        return residual + x


class EPDParamPolicy(nn.Module):
    """
    Predict Dirichlet parameters for EPD solver tables.

    Parameters
    ----------
    num_steps:
        Total number of coarse diffusion steps (including the initial state).
    num_points:
        Number of parallel intermediate points per step (K).
    hidden_dim:
        Width of the shared MLP.
    num_layers:
        Number of residual blocks processing the combined embeddings.
    context_dim:
        Optional per-step context dimensionality (e.g. global hyper-parameters,
        textual features). When >0 a linear projection is applied and added to
        step embeddings.
    dirichlet_alpha_eps:
        Minimum concentration to stabilise Dirichlet sampling.
    dirichlet_init:
        Optional cold-start Dirichlet parameters (see Stage 2). When provided,
        the policy exactly reproduces the distilled table at initialisation.
    """

    def __init__(
        self,
        num_steps: int,
        num_points: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        context_dim: int = 0,
        dirichlet_alpha_eps: float = 1e-5,
        dirichlet_init: Optional["DirichletInit"] = None,
    ) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2 (start and end).")
        if num_points < 1:
            raise ValueError("num_points must be positive.")

        self.num_steps = num_steps
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.dirichlet_alpha_eps = dirichlet_alpha_eps
        self.context_dim = context_dim

        self.step_embed = nn.Embedding(num_steps - 1, hidden_dim)
        if context_dim > 0:
            self.context_proj = nn.Linear(context_dim, hidden_dim)
        else:
            self.context_proj = None

        blocks = [ResidualBlock(hidden_dim) for _ in range(num_layers)]
        self.blocks = nn.ModuleList(blocks)

        out_dim = (num_points + 1) + num_points
        self.output_linear = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

        # Baseline log concentrations (buffers, no gradients).
        base_alpha_pos, base_alpha_weight = self._build_default_dirichlet(dirichlet_init)
        self.register_buffer("base_log_alpha_pos", base_alpha_pos.log())
        self.register_buffer("base_log_alpha_weight", base_alpha_weight.log())

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def forward(
        self,
        step_indices: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """
        Compute Dirichlet concentrations for the requested coarse steps.

        Parameters
        ----------
        step_indices:
            Tensor of shape (batch,) with integers in [0, num_steps-2].
        context:
            Optional tensor of shape (batch, context_dim). Pass None when no
            additional conditioning is required.
        """

        if step_indices.dim() != 1:
            raise ValueError("step_indices must be a 1-D tensor.")
        if context is not None and context.shape[0] != step_indices.shape[0]:
            raise ValueError("context batch dimension must match step_indices.")
        if context is not None and context.shape[-1] != self.context_dim:
            raise ValueError(f"context last dimension must be {self.context_dim}.")

        x = self.step_embed(step_indices)
        if self.context_proj is not None and context is not None:
            x = x + self.context_proj(context)

        for block in self.blocks:
            x = block(x)

        deltas = self.output_linear(x)
        delta_pos, delta_weight = torch.split(
            deltas, [self.num_points + 1, self.num_points], dim=-1
        )

        base_pos = self.base_log_alpha_pos.index_select(0, step_indices)
        base_weight = self.base_log_alpha_weight.index_select(0, step_indices)

        log_alpha_pos = base_pos + delta_pos
        log_alpha_weight = base_weight + delta_weight

        alpha_pos = torch.exp(log_alpha_pos).clamp_min(self.dirichlet_alpha_eps)
        alpha_weight = torch.exp(log_alpha_weight).clamp_min(self.dirichlet_alpha_eps)

        return PolicyOutput(
            alpha_pos=alpha_pos,
            alpha_weight=alpha_weight,
            log_alpha_pos=log_alpha_pos,
            log_alpha_weight=log_alpha_weight,
        )

    @torch.no_grad()
    def mean_table(self, policy_output: PolicyOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Dirichlet means to positions and weights.

        Returns
        -------
        positions:
            Tensor of shape (batch, num_points) with strictly increasing values.
        weights:
            Tensor of shape (batch, num_points) summing to 1 along the last axis.
        """

        mean_segments = self._normalize(policy_output.alpha_pos)
        mean_weights = self._normalize(policy_output.alpha_weight)
        positions = torch.cumsum(mean_segments[..., :-1], dim=-1)
        return positions, mean_weights

    def sample_table(
        self,
        policy_output: PolicyOutput,
        generator: Optional[torch.Generator] = None,
    ) -> PolicySample:
        """
        Draw a sample table (positions + weights) and compute log-probabilities.
        """

        dir_pos = Dirichlet(policy_output.alpha_pos)
        dir_weight = Dirichlet(policy_output.alpha_weight)

        if generator is not None:
            torch.random.set_rng_state(generator.get_state())
            segments = dir_pos.rsample()
            torch.random.set_rng_state(generator.get_state())
            weights = dir_weight.rsample()
        else:
            segments = dir_pos.rsample()
            weights = dir_weight.rsample()

        positions = torch.cumsum(segments[..., :-1], dim=-1)

        log_prob_pos = dir_pos.log_prob(segments)
        log_prob_weight = dir_weight.log_prob(weights)
        total_log_prob = log_prob_pos + log_prob_weight

        entropy_pos = dir_pos.entropy()
        entropy_weight = dir_weight.entropy()

        if not torch.isfinite(total_log_prob).all():
            raise RuntimeError("Encountered non-finite log-probability in policy sampling.")

        return PolicySample(
            positions=positions,
            weights=weights,
            segments=segments,
            log_prob=total_log_prob,
            entropy_pos=entropy_pos,
            entropy_weight=entropy_weight,
        )

    def log_prob(
        self,
        policy_output: PolicyOutput,
        segments: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-probability of provided segments/weights under the policy's
        Dirichlet distributions.
        """

        dir_pos = Dirichlet(policy_output.alpha_pos)
        dir_weight = Dirichlet(policy_output.alpha_weight)
        log_prob_pos = dir_pos.log_prob(segments)
        log_prob_weight = dir_weight.log_prob(weights)
        return log_prob_pos + log_prob_weight

    def entropy(self, policy_output: PolicyOutput) -> torch.Tensor:
        """Return the entropy of the Dirichlet factors for diagnostic purposes."""

        dir_pos = Dirichlet(policy_output.alpha_pos)
        dir_weight = Dirichlet(policy_output.alpha_weight)
        return dir_pos.entropy() + dir_weight.entropy()

    def kl_to_base(
        self,
        policy_output: PolicyOutput,
        step_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between the current policy factors and the
        cold-start (baseline) Dirichlet parameters.
        """

        base_alpha_pos, base_alpha_weight = self._get_base_alpha(step_indices)
        kl_pos = self._dirichlet_kl(policy_output.alpha_pos, base_alpha_pos)
        kl_weight = self._dirichlet_kl(policy_output.alpha_weight, base_alpha_weight)
        return kl_pos + kl_weight

    @torch.no_grad()
    def load_dirichlet_init(self, dirichlet_init: "DirichletInit") -> None:
        """
        Replace the baseline Dirichlet buffers with new cold-start parameters.
        """

        base_pos, base_weight = self._build_default_dirichlet(dirichlet_init)
        self.base_log_alpha_pos.copy_(base_pos.log())
        self.base_log_alpha_weight.copy_(base_weight.log())

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _build_default_dirichlet(
        self,
        dirichlet_init: Optional["DirichletInit"],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dirichlet_init is not None:
            alpha_pos = torch.from_numpy(dirichlet_init.alpha_pos).float()
            alpha_weight = torch.from_numpy(dirichlet_init.alpha_weight).float()
        else:
            uniform_pos = torch.full(
                (self.num_steps - 1, self.num_points + 1),
                1.0 / (self.num_points + 1),
                dtype=torch.float32,
            )
            uniform_weight = torch.full(
                (self.num_steps - 1, self.num_points),
                1.0 / self.num_points,
                dtype=torch.float32,
            )
            alpha_pos = uniform_pos
            alpha_weight = uniform_weight

        alpha_pos = alpha_pos.clamp_min(self.dirichlet_alpha_eps)
        alpha_weight = alpha_weight.clamp_min(self.dirichlet_alpha_eps)
        return alpha_pos, alpha_weight

    def _get_base_alpha(
        self,
        step_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_pos = torch.exp(self.base_log_alpha_pos.index_select(0, step_indices))
        base_weight = torch.exp(self.base_log_alpha_weight.index_select(0, step_indices))
        return base_pos, base_weight

    @staticmethod
    def _dirichlet_kl(p_alpha: torch.Tensor, q_alpha: torch.Tensor) -> torch.Tensor:
        """
        KL divergence KL(p || q) for Dirichlet distributions with parameters p_alpha, q_alpha.
        """

        sum_p = p_alpha.sum(dim=-1)
        sum_q = q_alpha.sum(dim=-1)
        term1 = torch.lgamma(sum_p) - torch.lgamma(sum_q)
        term2 = torch.lgamma(p_alpha).sum(dim=-1) - torch.lgamma(q_alpha).sum(dim=-1)
        digamma_diff = torch.digamma(p_alpha) - torch.digamma(sum_p.unsqueeze(-1))
        term3 = ((p_alpha - q_alpha) * digamma_diff).sum(dim=-1)
        return term1 - term2 + term3

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        total = tensor.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return tensor / total


# Imported lazily to avoid circular dependencies in type checkers.
try:  # pragma: no cover - type checking helper
    from training.ppo.cold_start import DirichletInit  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    DirichletInit = None  # type: ignore
