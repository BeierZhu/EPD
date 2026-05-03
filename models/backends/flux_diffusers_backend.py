"""
FLUX backend adapter that wraps the official diffusers pipeline.

Like the SD3 adapter, this class hides diffusers-specific details behind
the legacy `(net(x, t, condition=...))` solver interface. The rest of the
EPD codebase continues to operate on 4D latent maps even though FLUX
internally packs them into patch sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    from diffusers import FluxPipeline
except ImportError as exc:  # pragma: no cover - handled explicitly for clarity
    raise ImportError(
        "FluxPipeline is unavailable. Install a diffusers build that includes FLUX support. "
        "On Python 3.9 this usually means an installed site-packages diffusers with FLUX; "
        "the vendored diffusers tree in this repo targets Python >= 3.10."
    ) from exc


PromptType = Union[str, Sequence[str]]


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Match the diffusers helper that maps sequence length to scheduler `mu`."""

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _is_empty_negative_prompt(prompt: Optional[PromptType]) -> bool:
    if prompt is None:
        return True
    if isinstance(prompt, str):
        return prompt == ""
    return all(item is None or str(item) == "" for item in prompt)


@dataclass
class FluxConditioning:
    """Container for cached FLUX text embeddings and guidance metadata."""

    prompt_embeds: torch.FloatTensor
    pooled_prompt_embeds: torch.FloatTensor
    text_ids: torch.FloatTensor
    guidance_scale: float
    num_images_per_prompt: int = 1


class FluxDiffusersBackend(nn.Module):
    """
    Thin adapter over `diffusers.FluxPipeline`.

    The public contract mirrors the SD3 backend so existing solver code can
    treat FLUX as another flow-matching backend.
    """

    def __init__(
        self,
        model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
        *,
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        guidance_scale: float = 3.5,
        enable_model_cpu_offload: bool = False,
        max_sequence_length: int = 512,
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        use_safetensors: Optional[bool] = True,
        token: Optional[str] = None,
        pipeline_kwargs: Optional[dict] = None,
        flowmatch_mu: Optional[float] = None,
        resolution: int = 1024,
    ) -> None:
        super().__init__()
        if resolution != 1024:
            raise ValueError(f"resolution must be 1024 for FLUX.1-dev in this checkout; got {resolution}")

        self.default_guidance_scale = float(guidance_scale)
        self.max_sequence_length = int(max_sequence_length)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.requested_resolution = int(resolution)

        load_kwargs = {
            "torch_dtype": torch_dtype,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        if variant is not None:
            load_kwargs["variant"] = variant
        if use_safetensors is not None:
            load_kwargs["use_safetensors"] = use_safetensors
        if token is not None:
            load_kwargs["token"] = token
        if pipeline_kwargs:
            load_kwargs.update(pipeline_kwargs)

        print(
            f"[FluxBackend] Loading pipeline model='{model_name_or_path}' "
            f"device={self.device} dtype={torch_dtype} offload={enable_model_cpu_offload}"
        )
        self.pipeline = FluxPipeline.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        print("[FluxBackend] Pipeline weights loaded.")
        if enable_model_cpu_offload:
            if not hasattr(self.pipeline, "enable_model_cpu_offload"):
                raise RuntimeError("Installed diffusers FluxPipeline does not expose enable_model_cpu_offload().")
            self.pipeline.enable_model_cpu_offload()
            self._using_model_offload = True
            print("[FluxBackend] Enabled model CPU offload.")
        else:
            self.pipeline.to(self.device)
            self._using_model_offload = False
            print(f"[FluxBackend] Moved pipeline to device {self.device}.")

        transformer_cfg = self.pipeline.transformer.config
        vae_sf = getattr(self.pipeline, "vae_scale_factor", None)
        if vae_sf is None:
            vae_sf = 8

        # FLUX packs 2x2 latent patches, so the effective latent map is rounded
        # to the nearest size divisible by `vae_scale_factor * 2`.
        if self.requested_resolution % (vae_sf * 2) != 0:
            raise ValueError(
                f"resolution {self.requested_resolution} must be divisible by {vae_sf * 2} for FLUX packing."
            )
        self.output_resolution = self.requested_resolution
        self.latent_resolution = 2 * (self.output_resolution // (vae_sf * 2))
        self.img_resolution = self.latent_resolution  # legacy field used throughout EPD
        self.img_channels = int(transformer_cfg.in_channels // 4)
        self.label_dim = False
        self.backend = "flux"
        self.backend_config = {
            "resolution": self.output_resolution,
            "latent_resolution": self.latent_resolution,
        }

        scheduler = self.pipeline.scheduler
        self.sigma_min = float(getattr(scheduler, "sigma_min", 0.0))
        self.sigma_max = float(getattr(scheduler, "sigma_max", 1.0))
        self.flow_shift = float(getattr(scheduler, "shift", 1.0))
        scheduler_cfg = getattr(scheduler, "config", {})
        self.flowmatch_use_dynamic_shifting = bool(getattr(scheduler_cfg, "use_dynamic_shifting", False))
        self.flowmatch_base_seq_len = int(getattr(scheduler_cfg, "base_image_seq_len", 256))
        self.flowmatch_max_seq_len = int(getattr(scheduler_cfg, "max_image_seq_len", 4096))
        self.flowmatch_base_shift = float(getattr(scheduler_cfg, "base_shift", 0.5))
        self.flowmatch_max_shift = float(getattr(scheduler_cfg, "max_shift", 1.15))
        self.default_flowmatch_mu = flowmatch_mu
        if self.default_flowmatch_mu is None and self.flowmatch_use_dynamic_shifting:
            self.default_flowmatch_mu = self._compute_default_mu()
        self.current_flowmatch_mu: Optional[float] = None

    def prepare_condition(
        self,
        prompt: PromptType,
        negative_prompt: Optional[PromptType] = None,
        *,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: Optional[int] = None,
    ) -> FluxConditioning:
        if not _is_empty_negative_prompt(negative_prompt):
            raise ValueError(
                "FLUX v1 integration in this checkout does not implement true CFG / negative prompt branches."
            )

        gs = float(guidance_scale) if guidance_scale is not None else self.default_guidance_scale
        max_seq = int(max_sequence_length or self.max_sequence_length)
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.pipeline._execution_device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_seq,
        )
        return FluxConditioning(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            guidance_scale=gs,
            num_images_per_prompt=num_images_per_prompt,
        )

    def forward(  # type: ignore[override]
        self,
        latents: torch.FloatTensor,
        t: Union[float, torch.Tensor],
        *,
        condition: Optional[FluxConditioning] = None,
        unconditional_condition: Optional[FluxConditioning] = None,
    ) -> torch.FloatTensor:
        """
        Compute `x - t * v_theta` so solvers can keep using `(x - denoised) / t`.
        """
        if condition is None:
            raise ValueError("FluxDiffusersBackend requires a conditioning object.")
        if unconditional_condition is not None:
            raise ValueError("FLUX backend does not use `unconditional_condition`; pass guidance via prepare_condition.")

        exec_device = self.pipeline._execution_device
        latent_dtype = self.pipeline.transformer.dtype
        latents = latents.to(device=exec_device, dtype=latent_dtype)
        batch_size, _, height, width = latents.shape
        packed_latents = self.pipeline._pack_latents(latents, batch_size, self.img_channels, height, width)
        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size,
            height // 2,
            width // 2,
            exec_device,
            condition.prompt_embeds.dtype,
        )

        velocity = self._run_transformer_step(packed_latents, latent_image_ids, t, condition)
        sigma = self._format_sigmas(t, batch_size, packed_latents.dtype, packed_latents.device).view(batch_size, 1, 1)
        denoised_packed = packed_latents - sigma * velocity
        denoised = self._unpack_latents(denoised_packed, height, width)
        self._maybe_free_model_hooks()
        return denoised

    __call__ = forward

    def vae_decode(self, latents: torch.FloatTensor, output_type: str = "tensor"):
        latents = latents.to(self.pipeline._execution_device, dtype=self.pipeline.transformer.dtype)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        if output_type == "tensor":
            return image
        return self.pipeline.image_processor.postprocess(image, output_type=output_type)

    def _compute_default_mu(self, height: Optional[int] = None, width: Optional[int] = None) -> Optional[float]:
        if not self.flowmatch_use_dynamic_shifting:
            return None
        h = height or self.img_resolution
        w = width or self.img_resolution
        image_seq_len = (h // 2) * (w // 2)
        return _calculate_shift(
            image_seq_len=image_seq_len,
            base_seq_len=self.flowmatch_base_seq_len,
            max_seq_len=self.flowmatch_max_seq_len,
            base_shift=self.flowmatch_base_shift,
            max_shift=self.flowmatch_max_shift,
        )

    def resolve_flowmatch_mu(
        self,
        *,
        height: Optional[int] = None,
        width: Optional[int] = None,
        override: Optional[float] = None,
    ) -> Optional[float]:
        if override is not None:
            return float(override)
        if self.default_flowmatch_mu is not None:
            return float(self.default_flowmatch_mu)
        return self._compute_default_mu(height=height, width=width)

    def make_flowmatch_schedule(
        self,
        num_steps: int,
        *,
        device: Optional[torch.device] = None,
        mu: Optional[float] = None,
    ) -> torch.Tensor:
        scheduler = self.pipeline.scheduler
        scheduler_device = self.pipeline._execution_device
        resolved_mu = self.resolve_flowmatch_mu(override=mu)
        scheduler_kwargs = {}
        if resolved_mu is not None:
            scheduler_kwargs["mu"] = resolved_mu
        scheduler.set_timesteps(num_inference_steps=num_steps, device=scheduler_device, **scheduler_kwargs)
        sigmas = scheduler.sigmas[:-1]
        sigmas = sigmas.to(device=device or scheduler_device)
        self.current_flowmatch_mu = resolved_mu
        return sigmas

    def _run_transformer_step(
        self,
        packed_latents: torch.Tensor,
        latent_image_ids: torch.Tensor,
        t: Union[float, torch.Tensor],
        condition: FluxConditioning,
    ) -> torch.Tensor:
        exec_device = self.pipeline._execution_device
        batch_size = packed_latents.shape[0]
        sigma = self._format_sigmas(t, batch_size, packed_latents.dtype, packed_latents.device)

        guidance = None
        if bool(getattr(self.pipeline.transformer.config, "guidance_embeds", False)):
            guidance = torch.full([batch_size], condition.guidance_scale, device=exec_device, dtype=torch.float32)

        joint_kwargs = getattr(self.pipeline, "_joint_attention_kwargs", None)
        noise_pred = self.pipeline.transformer(
            hidden_states=packed_latents,
            timestep=sigma,
            guidance=guidance,
            pooled_projections=condition.pooled_prompt_embeds,
            encoder_hidden_states=condition.prompt_embeds,
            txt_ids=condition.text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_kwargs,
            return_dict=False,
        )[0]
        return noise_pred.to(device=exec_device, dtype=packed_latents.dtype)

    def _maybe_free_model_hooks(self) -> None:
        if not getattr(self, "_using_model_offload", False):
            return
        try:
            self.pipeline.maybe_free_model_hooks()
        except Exception:
            pass

    @staticmethod
    def _format_sigmas(
        t: Union[float, torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([float(t)], device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        elif t.ndim == 1 and t.shape[0] != batch_size:
            raise ValueError(f"Timestep tensor shape {t.shape} does not match batch size {batch_size}.")
        elif t.ndim > 1:
            t = t.reshape(batch_size)
        return t

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, channels = latents.shape
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, height, width)
