#!/usr/bin/env python3
"""
Minimal FLUX sampler that replays an exported EPD predictor table.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Sequence

import click
import torch
from torchvision.utils import make_grid, save_image

from sample import _prepare_flux_condition, create_model_flux
from training.loss import get_solver_fn
from training.networks import EPD_predictor
from training.ppo.pipeline_utils import (
    load_prompts_file,
    resolve_flux_runtime_metadata,
    resolve_predictor_path,
)


class StackedRandomGenerator:
    """Batch-aware RNG helper so each seed gets its own generator."""

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
        lines = (lines * reps)[:count]
        return lines
    return [""] * count


def _load_predictor(path: Path, device: torch.device) -> EPD_predictor:
    import pickle

    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    predictor = snapshot["model"].to(device).eval()
    return predictor


@click.command()
@click.option(
    "--predictor",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    required=True,
    help="EPD predictor .pkl or RL run directory.",
)
@click.option("--seeds", type=parse_int_list, default="0", show_default=True, help="Seed list/range.")
@click.option("--prompt", type=str, default=None, help="Single prompt for all seeds.")
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False), default=None, help="Text/CSV prompt file.")
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Override FLUX.1-dev model repo or local snapshot path.",
)
@click.option("--outdir", type=str, default="./samples/flux_epd", show_default=True, help="Output directory.")
@click.option("--grid", is_flag=True, default=False, help="Save grid per batch.")
@click.option("--max-batch-size", type=click.IntRange(min=1), default=1, show_default=True, help="Batch size.")
@click.option(
    "--resolution",
    type=click.Choice(["1024"], case_sensitive=False),
    default=None,
    help="Override resolution; FLUX v1 only supports 1024 and must match predictor metadata.",
)
def main(
    predictor: str,
    seeds: List[int],
    prompt: str | None,
    prompt_file: str | None,
    model_id: str | None,
    outdir: str,
    grid: bool,
    max_batch_size: int,
    resolution: str | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor_path = resolve_predictor_path(predictor)
    predictor_module = _load_predictor(predictor_path, device=device)

    cli_resolution = int(resolution) if resolution is not None else None
    predictor_resolution = getattr(predictor_module, "img_resolution", None)
    backend_cfg = {}
    if isinstance(getattr(predictor_module, "backend_config", None), dict):
        backend_cfg = dict(predictor_module.backend_config)
    model_override = model_id or os.environ.get("FLUX_MODEL_PATH")
    if model_override:
        backend_cfg["model_name_or_path"] = model_override
        backend_cfg["model_id"] = model_override
    resolved_flux = resolve_flux_runtime_metadata(
        backend_options=backend_cfg,
        resolution=cli_resolution or predictor_resolution or backend_cfg.get("resolution"),
        sigma_min=getattr(predictor_module, "sigma_min", None),
        sigma_max=getattr(predictor_module, "sigma_max", None),
        flowmatch_mu=getattr(predictor_module, "flowmatch_mu", None),
        flowmatch_shift=getattr(predictor_module, "flowmatch_shift", None),
    )
    backend_cfg = dict(resolved_flux["backend_options"])
    if cli_resolution is not None and predictor_resolution is not None and cli_resolution != predictor_resolution:
        raise RuntimeError(
            f"Resolution override ({cli_resolution}) does not match predictor metadata ({predictor_resolution})."
        )
    resolved_resolution = resolved_flux["resolution"]
    if int(resolved_resolution) != 1024:
        raise RuntimeError(f"FLUX v1 sampler only supports resolution 1024, got {resolved_resolution}.")
    backend_cfg["resolution"] = 1024

    backend, _ = create_model_flux(
        guidance_rate=predictor_module.guidance_rate,
        guidance_type=predictor_module.guidance_type,
        backend_config=backend_cfg,
        device=device,
    )

    num_steps = predictor_module.num_steps
    schedule_type = predictor_module.schedule_type or "flowmatch"
    sigma_min = float(resolved_flux["sigma_min"])
    sigma_max = float(resolved_flux["sigma_max"])

    sampler_config = {
        "num_steps": num_steps,
        "guidance_rate": predictor_module.guidance_rate,
        "schedule_type": schedule_type,
        "schedule_rho": predictor_module.schedule_rho,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "flowmatch_mu": float(resolved_flux["flowmatch_mu"]),
        "flowmatch_shift": float(resolved_flux["flowmatch_shift"]),
        "backend": backend.backend,
        "resolution": getattr(backend, "output_resolution", backend.img_resolution),
        "backend_config": backend.backend_config,
    }
    print("[sampler] configuration:")
    for k, v in sampler_config.items():
        print(f"  - {k}: {v}")

    prompts = _load_prompts(prompt, prompt_file, len(seeds))
    seeds_tensor = torch.as_tensor(seeds)
    num_batches = ((len(seeds) - 1) // max_batch_size) + 1
    all_batches = seeds_tensor.tensor_split(num_batches)
    sampler_name = getattr(predictor_module, "sampler_stu", "epd") or "epd"
    if sampler_name == "epd":
        sampler_name = "epd_parallel"
    sampler_fn = get_solver_fn(sampler_name)
    afs = bool(getattr(predictor_module, "afs", False))

    os.makedirs(outdir, exist_ok=True)

    batch_start = 0
    for batch_idx, batch_seeds in enumerate(all_batches):
        batch_size = int(batch_seeds.numel())
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_start += batch_size
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [len(batch_seeds), backend.img_channels, backend.img_resolution, backend.img_resolution],
            device=device,
            dtype=backend.pipeline.transformer.dtype,
        )

        condition = _prepare_flux_condition(
            backend,
            batch_prompts,
            predictor_module.guidance_rate,
            backend_cfg,
        )

        with torch.no_grad():
            samples, _ = sampler_fn(
                net=backend,
                latents=latents,
                condition=condition,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                schedule_type=schedule_type,
                schedule_rho=predictor_module.schedule_rho,
                guidance_type=predictor_module.guidance_type,
                guidance_rate=predictor_module.guidance_rate,
                predictor=predictor_module if sampler_name.startswith("epd") else None,
                afs=afs,
                return_inters=False,
                train=False,
            )
            images = backend.vae_decode(samples)

        images = torch.clamp(images / 2 + 0.5, 0, 1)

        if grid:
            grid_img = make_grid(images, nrow=int(len(images) ** 0.5) or 1, padding=0)
            save_image(grid_img, os.path.join(outdir, f"grid_batch{batch_idx:04d}.png"))

        images_np = (images * 255).round().to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        for seed_val, image_np in zip(batch_seeds.tolist(), images_np):
            image_dir = os.path.join(outdir, f"{seed_val - seed_val % 1000:06d}")
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{seed_val:06d}.png")
            from PIL import Image

            Image.fromarray(image_np, "RGB").save(image_path)

    print("FLUX EPD sampling done.")


if __name__ == "__main__":
    main()
