#!/usr/bin/env python3
"""
Minimal FLUX baseline sampler.

- `flux` / `flowmatch` / `euler` use the official diffusers FLUX Euler rollout.
- `edm` / `dpm` / `dpm2` / `heun` / `ipndm` / `ddim` reuse the project solvers
  through `FluxDiffusersBackend`.
"""

from __future__ import annotations

import os
import re
from typing import List, Sequence

import click
import torch
from torchvision.utils import make_grid, save_image

from sample import _prepare_flux_condition, create_model_flux
from training.loss import get_solver_fn
from training.ppo.pipeline_utils import load_prompts_file, resolve_flux_runtime_metadata


class StackedRandomGenerator:
    def __init__(self, device: torch.device, seeds: Sequence[int]):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def _load_prompts(prompt: str | None, prompt_file: str | None, count: int) -> List[str]:
    if prompt is not None:
        return [prompt] * count
    if prompt_file:
        lines = load_prompts_file(prompt_file)
        reps = (count + len(lines) - 1) // len(lines)
        return (lines * reps)[:count]
    return [""] * count


@click.command()
@click.option(
    "--sampler",
    type=click.Choice(["flux", "flowmatch", "euler", "edm", "dpm", "dpm2", "heun", "ipndm", "ddim"], case_sensitive=False),
    default="flux",
    show_default=True,
    help="Sampler backend. flux/flowmatch/euler use the official diffusers FLUX Euler baseline.",
)
@click.option("--num-steps", type=int, default=28, show_default=True, help="Number of inference steps.")
@click.option("--batch", "max_batch_size", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--seeds", type=parse_int_list, default="0", show_default=True, help="Random seeds list/range.")
@click.option("--prompt", type=str, default=None, help="Single prompt to use for all seeds.")
@click.option(
    "--prompt-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Text file or CSV with 'text' column.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Diffusers model repo or local snapshot path. Defaults to FLUX_MODEL_PATH, then black-forest-labs/FLUX.1-dev.",
)
@click.option("--outdir", type=str, default="./samples/flux_baseline", show_default=True, help="Output directory.")
@click.option("--grid", type=bool, default=False, help="Whether to save a grid image per batch.")
@click.option(
    "--subdirs",
    is_flag=True,
    default=True,
    help="Create subdirectory for every 1000 seeds.",
)
@click.option("--guidance-type", type=click.Choice(["cfg"], case_sensitive=False), default="cfg", show_default=True)
@click.option("--guidance-rate", type=float, default=3.5, show_default=True, help="Embedded FLUX guidance scale.")
@click.option(
    "--resolution",
    type=click.Choice(["1024"], case_sensitive=False),
    default="1024",
    show_default=True,
    help="Square image resolution.",
)
@click.option("--schedule-type", type=click.Choice(["flowmatch"], case_sensitive=False), default="flowmatch", show_default=True)
@click.option("--schedule-rho", type=float, default=7.0, show_default=True, help="Kept for solver interface parity.")
@click.option("--afs", is_flag=True, default=False, help="Apply AFS on first step (for generic solvers).")
@click.option("--inner-steps", type=int, default=None, help="Inner steps for DPM/Heun.")
@click.option("--solver-r", type=float, default=0.5, show_default=True, help="DPM relaxation factor.")
@click.option("--max-order", type=int, default=4, show_default=True, help="Max order for IPNDM.")
@click.option("--ddim-eta", type=float, default=0.0, show_default=True, help="DDIM eta parameter.")
@click.option("--ddim-steps", type=int, default=None, help="Override number of DDIM steps.")
def main(
    sampler: str,
    num_steps: int,
    max_batch_size: int,
    seeds: List[int],
    prompt: str | None,
    prompt_file: str | None,
    outdir: str,
    grid: bool,
    subdirs: bool,
    model_id: str,
    guidance_type: str,
    guidance_rate: float,
    resolution: str,
    schedule_type: str,
    schedule_rho: float,
    afs: bool,
    inner_steps: int | None,
    solver_r: float,
    max_order: int,
    ddim_eta: float,
    ddim_steps: int | None,
) -> None:
    sampler_choice = sampler.lower()
    sampler_mode = "flowmatch" if sampler_choice in {"flux", "flowmatch", "euler"} else sampler_choice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_resolution = int(resolution)
    model_id = model_id or os.environ.get("FLUX_MODEL_PATH") or "black-forest-labs/FLUX.1-dev"

    backend_cfg = {
        "model_id": model_id,
        "model_name_or_path": model_id,
        "resolution": target_resolution,
    }
    resolved_flux = resolve_flux_runtime_metadata(
        backend_options=backend_cfg,
        resolution=target_resolution,
    )
    backend_cfg = dict(resolved_flux["backend_options"])
    backend, _ = create_model_flux(
        dataset_name="ms_coco",
        guidance_type=guidance_type,
        guidance_rate=guidance_rate,
        device=device,
        backend_config=backend_cfg,
    )
    if not hasattr(backend, "round_sigma"):
        backend.round_sigma = lambda x: x
    pipe = backend.pipeline
    pipe.set_progress_bar_config(disable=True)

    prompts = _load_prompts(prompt, prompt_file, len(seeds))
    seeds_tensor = torch.as_tensor(seeds)
    num_batches = ((len(seeds) - 1) // max_batch_size) + 1
    all_batches = seeds_tensor.tensor_split(num_batches)

    sampler_config = {
        "sampler_requested": sampler_choice,
        "sampler_impl": "official_flux_euler" if sampler_mode == "flowmatch" else f"project_{sampler_mode}",
        "num_steps": num_steps,
        "guidance_rate": guidance_rate,
        "schedule_type": schedule_type,
        "schedule_rho": schedule_rho,
        "sigma_min": float(resolved_flux["sigma_min"]),
        "sigma_max": float(resolved_flux["sigma_max"]),
        "flowmatch_mu": float(resolved_flux["flowmatch_mu"]),
        "flowmatch_shift": float(resolved_flux["flowmatch_shift"]),
        "backend": backend.backend,
        "resolution": getattr(backend, "output_resolution", backend.img_resolution),
        "backend_config": backend.backend_config,
    }
    print("[sampler] configuration:")
    for key, value in sampler_config.items():
        print(f"  - {key}: {value}")

    solver_fn = None if sampler_mode == "flowmatch" else get_solver_fn(sampler_mode)

    os.makedirs(outdir, exist_ok=True)

    batch_start = 0
    for batch_idx, batch_seeds in enumerate(all_batches):
        batch_size = int(batch_seeds.numel())
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_start += batch_size

        if sampler_mode == "flowmatch":
            generators = [torch.Generator(device).manual_seed(int(seed.item()) % (1 << 32)) for seed in batch_seeds]
            with torch.no_grad():
                result = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_rate,
                    generator=generators,
                    height=target_resolution,
                    width=target_resolution,
                    output_type="pt",
                )
            # `output_type="pt"` is already postprocessed by diffusers into [0, 1].
            images = torch.clamp(result.images, 0, 1)
        else:
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                [len(batch_seeds), backend.img_channels, backend.img_resolution, backend.img_resolution],
                device=device,
                dtype=pipe.transformer.dtype,
            )
            condition = _prepare_flux_condition(
                backend,
                batch_prompts,
                guidance_rate,
                backend_cfg,
            )

            solver_kwargs = dict(
                num_steps=int(ddim_steps) if sampler_mode == "ddim" and ddim_steps is not None else num_steps,
                sigma_min=float(resolved_flux["sigma_min"]),
                sigma_max=float(resolved_flux["sigma_max"]),
                schedule_type=schedule_type,
                schedule_rho=schedule_rho,
                afs=afs,
                predictor=None,
                train=False,
            )
            if sampler_mode == "dpm":
                solver_kwargs["inner_steps"] = inner_steps if inner_steps is not None else 2
                solver_kwargs["r"] = solver_r
            if sampler_mode == "heun":
                solver_kwargs["inner_steps"] = inner_steps if inner_steps is not None else 3
            if sampler_mode == "ipndm":
                solver_kwargs["max_order"] = max_order
            if sampler_mode == "ddim":
                solver_kwargs["ddim_eta"] = ddim_eta

            with torch.no_grad():
                samples, _ = solver_fn(
                    net=backend,
                    latents=latents,
                    condition=condition,
                    **solver_kwargs,
                )
                images = backend.vae_decode(samples)
                images = torch.clamp(images / 2 + 0.5, 0, 1)

        if grid:
            grid_img = make_grid(images, nrow=int(len(images) ** 0.5) or 1, padding=0)
            save_image(grid_img, os.path.join(outdir, f"grid_batch{batch_idx:04d}.png"))

        images_np = (images * 255).round().to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        for seed_val, image_np in zip(batch_seeds.tolist(), images_np):
            image_dir = os.path.join(outdir, f"{seed_val - seed_val % 1000:06d}") if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{seed_val:06d}.png")
            from PIL import Image

            Image.fromarray(image_np, "RGB").save(image_path)

    print("FLUX baseline sampling done.")


if __name__ == "__main__":
    main()
