"""
Policy network for PPO-based fine-tuning of EPD parameter tables.

The network maps each coarse diffusion step to Dirichlet concentration
vectors describing:
    * Position segments (K+1 values that integrate to 1.0 and recover
      monotonically increasing intermediate locations `r` via cumsum).
    * Gradient weights (K values on the simplex).
    * Optional scale_dir / scale_time factors, modeled independently via
      log-normal distributions in log-space.

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
from torch.distributions import Dirichlet, LogNormal


@dataclass
class PolicyOutput:
    """Container for per-step Dirichlet concentrations."""

    alpha_pos: torch.Tensor  # shape: (batch, num_points + 1)
    alpha_weight: torch.Tensor  # shape: (batch, num_points)
    log_alpha_pos: torch.Tensor  # auxiliary tensor (same shape as alpha_pos)
    log_alpha_weight: torch.Tensor  # auxiliary tensor (same shape as alpha_weight)
    scale_dir_loc: Optional[torch.Tensor] = None  # log-space mean, shape: (batch, num_points)
    scale_dir_log_std: Optional[torch.Tensor] = None  # log-space std, shape: (batch, num_points)
    scale_time_loc: Optional[torch.Tensor] = None  # log-space mean, shape: (batch, num_points)
    scale_time_log_std: Optional[torch.Tensor] = None  # log-space std, shape: (batch, num_points)


@dataclass
class PolicySample:
    """Result of sampling a parameter table from the policy."""

    positions: torch.Tensor  # shape: (batch, num_points)
    weights: torch.Tensor  # shape: (batch, num_points)
    segments: torch.Tensor  # sampled simplex segments for positions
    log_prob: torch.Tensor  # shape: (batch,)
    entropy_pos: torch.Tensor  # shape: (batch,)
    entropy_weight: torch.Tensor  # shape: (batch,)
    scale_dir: Optional[torch.Tensor] = None  # shape: (batch, num_points)
    scale_time: Optional[torch.Tensor] = None  # shape: (batch, num_points)
    entropy_scale_dir: Optional[torch.Tensor] = None  # shape: (batch,)
    entropy_scale_time: Optional[torch.Tensor] = None  # shape: (batch,)


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
    use_scale_dir / use_scale_time:
        Whether to model scale_dir / scale_time with independent log-normal
        factors (per step, per point). When enabled, the policy outputs
        log-space means and standard deviations.
    scale_dir_init / scale_time_init:
        Optional baseline scale tables (shape: num_steps-1 x num_points).
        When provided, they define the initial log-normal means.
    scale_log_std_init:
        Initial log-standard deviation (log space) used for scale factors.
    scale_log_std_min / scale_log_std_max:
        Clamp range for log-standard deviation to keep sampling stable.
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
        use_scale_dir: bool = False,
        use_scale_time: bool = False,
        scale_dir_init: Optional[torch.Tensor] = None,
        scale_time_init: Optional[torch.Tensor] = None,
        scale_log_std_init: float = -3.0,
        scale_log_std_min: float = -7.0,
        scale_log_std_max: float = 2.0,
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
        self.use_scale_dir = bool(use_scale_dir or scale_dir_init is not None)
        self.use_scale_time = bool(use_scale_time or scale_time_init is not None)
        self.scale_log_std_min = float(scale_log_std_min)
        self.scale_log_std_max = float(scale_log_std_max)
        self.scale_eps = 1e-6

        self.step_embed = nn.Embedding(num_steps - 1, hidden_dim)
        if context_dim > 0:
            self.context_proj = nn.Linear(context_dim, hidden_dim)
        else:
            self.context_proj = None

        blocks = [ResidualBlock(hidden_dim) for _ in range(num_layers)]
        self.blocks = nn.ModuleList(blocks)

        out_dim = (num_points + 1) + num_points
        if self.use_scale_dir:
            out_dim += 2 * num_points
        if self.use_scale_time:
            out_dim += 2 * num_points
        self.output_linear = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

        # Baseline log concentrations (buffers, no gradients).
        base_alpha_pos, base_alpha_weight = self._build_default_dirichlet(dirichlet_init)
        self.register_buffer("base_log_alpha_pos", base_alpha_pos.log())
        self.register_buffer("base_log_alpha_weight", base_alpha_weight.log())

        # Baseline log-scale means/stds for scale_dir / scale_time.
        if self.use_scale_dir:
            base_scale_dir = self._build_default_scale(scale_dir_init, "scale_dir")
            base_log_std_dir = torch.full_like(base_scale_dir, float(scale_log_std_init))
            self.register_buffer("base_log_scale_dir", base_scale_dir.log())
            self.register_buffer("base_log_std_dir", base_log_std_dir)
        if self.use_scale_time:
            base_scale_time = self._build_default_scale(scale_time_init, "scale_time")
            base_log_std_time = torch.full_like(base_scale_time, float(scale_log_std_init))
            self.register_buffer("base_log_scale_time", base_scale_time.log())
            self.register_buffer("base_log_std_time", base_log_std_time)

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
        offset = 0
        delta_pos = deltas[..., offset : offset + self.num_points + 1]
        offset += self.num_points + 1
        delta_weight = deltas[..., offset : offset + self.num_points]
        offset += self.num_points

        base_pos = self.base_log_alpha_pos.index_select(0, step_indices)
        base_weight = self.base_log_alpha_weight.index_select(0, step_indices)

        log_alpha_pos = base_pos + delta_pos
        log_alpha_weight = base_weight + delta_weight

        alpha_pos = torch.exp(log_alpha_pos).clamp_min(self.dirichlet_alpha_eps)
        alpha_weight = torch.exp(log_alpha_weight).clamp_min(self.dirichlet_alpha_eps)

        scale_dir_loc = None
        scale_dir_log_std = None
        if self.use_scale_dir:
            delta_scale_dir_loc = deltas[..., offset : offset + self.num_points]
            offset += self.num_points
            delta_scale_dir_log_std = deltas[..., offset : offset + self.num_points]
            offset += self.num_points
            base_scale_dir = self.base_log_scale_dir.index_select(0, step_indices)
            base_log_std_dir = self.base_log_std_dir.index_select(0, step_indices)
            scale_dir_loc = base_scale_dir + delta_scale_dir_loc
            scale_dir_log_std = (base_log_std_dir + delta_scale_dir_log_std).clamp(
                min=self.scale_log_std_min, max=self.scale_log_std_max
            )

        scale_time_loc = None
        scale_time_log_std = None
        if self.use_scale_time:
            delta_scale_time_loc = deltas[..., offset : offset + self.num_points]
            offset += self.num_points
            delta_scale_time_log_std = deltas[..., offset : offset + self.num_points]
            offset += self.num_points
            base_scale_time = self.base_log_scale_time.index_select(0, step_indices)
            base_log_std_time = self.base_log_std_time.index_select(0, step_indices)
            scale_time_loc = base_scale_time + delta_scale_time_loc
            scale_time_log_std = (base_log_std_time + delta_scale_time_log_std).clamp(
                min=self.scale_log_std_min, max=self.scale_log_std_max
            )

        return PolicyOutput(
            alpha_pos=alpha_pos,
            alpha_weight=alpha_weight,
            log_alpha_pos=log_alpha_pos,
            log_alpha_weight=log_alpha_weight,
            scale_dir_loc=scale_dir_loc,
            scale_dir_log_std=scale_dir_log_std,
            scale_time_loc=scale_time_loc,
            scale_time_log_std=scale_time_log_std,
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

    @torch.no_grad()
    def mean_scales(
        self, policy_output: PolicyOutput
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute log-normal means for scale_dir / scale_time, if enabled.
        """

        scale_dir = None
        scale_time = None
        if self.use_scale_dir and policy_output.scale_dir_loc is not None:
            dist_dir = LogNormal(policy_output.scale_dir_loc, torch.exp(policy_output.scale_dir_log_std))
            scale_dir = dist_dir.mean
        if self.use_scale_time and policy_output.scale_time_loc is not None:
            dist_time = LogNormal(policy_output.scale_time_loc, torch.exp(policy_output.scale_time_log_std))
            scale_time = dist_time.mean
        return scale_dir, scale_time

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
        dir_scale_dir = None
        dir_scale_time = None

        if self.use_scale_dir and policy_output.scale_dir_loc is not None:
            scale_dir_std = torch.exp(policy_output.scale_dir_log_std)
            dir_scale_dir = LogNormal(policy_output.scale_dir_loc, scale_dir_std)
        if self.use_scale_time and policy_output.scale_time_loc is not None:
            scale_time_std = torch.exp(policy_output.scale_time_log_std)
            dir_scale_time = LogNormal(policy_output.scale_time_loc, scale_time_std)

        if generator is not None:
            # Use the provided generator for all draws while preserving the global RNG state.
            current_state = torch.random.get_rng_state()
            try:
                torch.random.set_rng_state(generator.get_state())
                segments = dir_pos.rsample()
                weights = dir_weight.rsample()
                scale_dir = dir_scale_dir.rsample() if dir_scale_dir is not None else None
                scale_time = dir_scale_time.rsample() if dir_scale_time is not None else None
                generator.set_state(torch.random.get_rng_state())
            finally:
                torch.random.set_rng_state(current_state)
        else:
            segments = dir_pos.rsample()
            weights = dir_weight.rsample()
            scale_dir = dir_scale_dir.rsample() if dir_scale_dir is not None else None
            scale_time = dir_scale_time.rsample() if dir_scale_time is not None else None

        positions = torch.cumsum(segments[..., :-1], dim=-1)

        log_prob_pos = dir_pos.log_prob(segments)
        log_prob_weight = dir_weight.log_prob(weights)
        total_log_prob = log_prob_pos + log_prob_weight
        entropy_scale_dir = None
        entropy_scale_time = None
        if dir_scale_dir is not None and scale_dir is not None:
            log_prob_scale_dir = dir_scale_dir.log_prob(scale_dir).sum(dim=-1)
            total_log_prob = total_log_prob + log_prob_scale_dir
            entropy_scale_dir = dir_scale_dir.entropy().sum(dim=-1)
        if dir_scale_time is not None and scale_time is not None:
            log_prob_scale_time = dir_scale_time.log_prob(scale_time).sum(dim=-1)
            total_log_prob = total_log_prob + log_prob_scale_time
            entropy_scale_time = dir_scale_time.entropy().sum(dim=-1)

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
            scale_dir=scale_dir,
            scale_time=scale_time,
            entropy_scale_dir=entropy_scale_dir,
            entropy_scale_time=entropy_scale_time,
        )

    def log_prob(
        self,
        policy_output: PolicyOutput,
        segments: torch.Tensor,
        weights: torch.Tensor,
        scale_dir: Optional[torch.Tensor] = None,
        scale_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log-probability of provided segments/weights under the policy's
        Dirichlet distributions (plus optional log-normal scale factors).
        """

        dir_pos = Dirichlet(policy_output.alpha_pos)
        dir_weight = Dirichlet(policy_output.alpha_weight)
        log_prob_pos = dir_pos.log_prob(segments)
        log_prob_weight = dir_weight.log_prob(weights)
        total = log_prob_pos + log_prob_weight

        if self.use_scale_dir and policy_output.scale_dir_loc is not None:
            if scale_dir is None:
                raise ValueError("scale_dir samples must be provided when use_scale_dir is True.")
            dist_dir = LogNormal(policy_output.scale_dir_loc, torch.exp(policy_output.scale_dir_log_std))
            total = total + dist_dir.log_prob(scale_dir).sum(dim=-1)
        if self.use_scale_time and policy_output.scale_time_loc is not None:
            if scale_time is None:
                raise ValueError("scale_time samples must be provided when use_scale_time is True.")
            dist_time = LogNormal(policy_output.scale_time_loc, torch.exp(policy_output.scale_time_log_std))
            total = total + dist_time.log_prob(scale_time).sum(dim=-1)

        return total

    def entropy(self, policy_output: PolicyOutput) -> torch.Tensor:
        """Return the entropy of the Dirichlet factors for diagnostic purposes."""

        dir_pos = Dirichlet(policy_output.alpha_pos)
        dir_weight = Dirichlet(policy_output.alpha_weight)
        total = dir_pos.entropy() + dir_weight.entropy()
        if self.use_scale_dir and policy_output.scale_dir_loc is not None:
            dist_dir = LogNormal(policy_output.scale_dir_loc, torch.exp(policy_output.scale_dir_log_std))
            total = total + dist_dir.entropy().sum(dim=-1)
        if self.use_scale_time and policy_output.scale_time_loc is not None:
            dist_time = LogNormal(policy_output.scale_time_loc, torch.exp(policy_output.scale_time_log_std))
            total = total + dist_time.entropy().sum(dim=-1)
        return total

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
        total = kl_pos + kl_weight

        if self.use_scale_dir and policy_output.scale_dir_loc is not None:
            base_loc, base_log_std = self._get_base_scale_dir(step_indices)
            total = total + self._lognormal_kl(
                policy_output.scale_dir_loc, policy_output.scale_dir_log_std, base_loc, base_log_std
            )
        if self.use_scale_time and policy_output.scale_time_loc is not None:
            base_loc, base_log_std = self._get_base_scale_time(step_indices)
            total = total + self._lognormal_kl(
                policy_output.scale_time_loc, policy_output.scale_time_log_std, base_loc, base_log_std
            )

        return total

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

    def _build_default_scale(
        self,
        scale_init: Optional[torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        if scale_init is None:
            scale = torch.ones(
                (self.num_steps - 1, self.num_points),
                dtype=torch.float32,
            )
        else:
            scale = torch.as_tensor(scale_init, dtype=torch.float32)
            expected = (self.num_steps - 1, self.num_points)
            if scale.shape != expected:
                raise ValueError(f"{name} init shape {scale.shape} does not match expected {expected}.")
        return scale.clamp_min(self.scale_eps)

    def _get_base_alpha(
        self,
        step_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_pos = torch.exp(self.base_log_alpha_pos.index_select(0, step_indices))
        base_weight = torch.exp(self.base_log_alpha_weight.index_select(0, step_indices))
        return base_pos, base_weight

    def _get_base_scale_dir(self, step_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_loc = self.base_log_scale_dir.index_select(0, step_indices)
        base_log_std = self.base_log_std_dir.index_select(0, step_indices)
        return base_loc, base_log_std

    def _get_base_scale_time(self, step_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_loc = self.base_log_scale_time.index_select(0, step_indices)
        base_log_std = self.base_log_std_time.index_select(0, step_indices)
        return base_loc, base_log_std

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
    def _lognormal_kl(
        p_loc: torch.Tensor,
        p_log_std: torch.Tensor,
        q_loc: torch.Tensor,
        q_log_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence between log-normal distributions via their underlying normals.
        """

        p_var = torch.exp(2 * p_log_std)
        q_var = torch.exp(2 * q_log_std)
        diff = p_loc - q_loc
        kl = 0.5 * ((p_var + diff.pow(2)) / q_var - 1.0 + 2.0 * (q_log_std - p_log_std))
        return kl.sum(dim=-1)

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        total = tensor.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return tensor / total


# Imported lazily to avoid circular dependencies in type checkers.
try:  # pragma: no cover - type checking helper
    from training.ppo.cold_start import DirichletInit  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    DirichletInit = None  # type: ignore
