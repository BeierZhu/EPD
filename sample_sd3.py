#!/usr/bin/env python3
"""
Minimal SD3 sampler that replays an exported EPD predictor table.

Usage example:
    python sample_sd3.py --predictor exps/fake-sd3-9/network-snapshot-000005.pkl \
        --seeds 1 --outdir samples/test --prompt "A maglev train going vertically downward in high speed, New York Times photojournalism."
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import List, Sequence

import click
import torch
from torchvision.utils import make_grid, save_image

from sample import create_model_sd3
from solvers import epd_sampler
from solver_utils import get_schedule
from training.networks import EPD_predictor


# -----------------------------------------------------------------------------
# Helpers (trimmed from sample.py)


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
        path = Path(prompt_file)
        lines: List[str] = []
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "text" in row and row["text"].strip():
                        lines.append(row["text"].strip())
        else:
            with path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise RuntimeError(f"No prompts found in '{prompt_file}'.")
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


# -----------------------------------------------------------------------------
# CLI


@click.command()
@click.option("--predictor", type=click.Path(exists=True, dir_okay=False), required=True, help="EPD predictor .pkl.")
@click.option("--seeds", type=parse_int_list, default="0-3", show_default=True, help="Seed list/range.")
@click.option("--prompt", type=str, default=None, help="Single prompt for all seeds.")
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False), default=None, help="Text/CSV prompt file.")
@click.option("--outdir", type=str, default="./samples/sd3_epd", show_default=True, help="Output directory.")
@click.option("--grid", is_flag=True, default=False, help="Save grid per batch.")
@click.option("--max-batch-size", type=click.IntRange(min=1), default=4, show_default=True, help="Batch size.")
@click.option(
    "--resolution",
    type=click.Choice(["512", "1024"], case_sensitive=False),
    default=None,
    help="Override resolution; defaults to predictor/back-end config.",
)
def main(
    predictor: str,
    seeds: List[int],
    prompt: str | None,
    prompt_file: str | None,
    outdir: str,
    grid: bool,
    max_batch_size: int,
    resolution: str | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor_path = Path(predictor)
    predictor_module = _load_predictor(predictor_path, device=device)

    cli_resolution: int | None = int(resolution) if resolution is not None else None
    predictor_resolution = getattr(predictor_module, "img_resolution", None)
    backend_cfg = {}
    if isinstance(getattr(predictor_module, "backend_config", None), dict):
        backend_cfg = dict(predictor_module.backend_config)
    backend_cfg.setdefault("flowmatch_mu", getattr(predictor_module, "flowmatch_mu", None))
    backend_cfg.setdefault("sigma_min", getattr(predictor_module, "sigma_min", None))
    backend_cfg.setdefault("sigma_max", getattr(predictor_module, "sigma_max", None))
    if cli_resolution is not None and predictor_resolution is not None and cli_resolution != predictor_resolution:
        raise RuntimeError(
            f"Resolution override ({cli_resolution}) does not match predictor metadata ({predictor_resolution})."
        )
    cfg_resolution = backend_cfg.get("resolution")
    if cfg_resolution is not None:
        cfg_resolution = int(cfg_resolution)
    resolved_resolution = cli_resolution or cfg_resolution or predictor_resolution or 1024
    backend_cfg["resolution"] = int(resolved_resolution)

    backend, _ = create_model_sd3(
        guidance_rate=predictor_module.guidance_rate,
        backend_config=backend_cfg,
        device=device,
    )

    num_steps = predictor_module.num_steps
    schedule_type = predictor_module.schedule_type or "flowmatch"
    sigma_min = getattr(predictor_module, "sigma_min", None) or backend.sigma_min
    sigma_max = getattr(predictor_module, "sigma_max", None) or backend.sigma_max

    # Log sampler configuration for reproducibility/debugging.
    sampler_config = {
        "num_steps": num_steps,
        "guidance_rate": predictor_module.guidance_rate,
        "schedule_type": schedule_type,
        "schedule_rho": predictor_module.schedule_rho,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "flowmatch_mu": backend.resolve_flowmatch_mu(override=backend_cfg.get("flowmatch_mu")),
        "flowmatch_shift": backend.flow_shift,
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

    os.makedirs(outdir, exist_ok=True)

    for batch_idx, batch_seeds in enumerate(all_batches):
        batch_prompts = prompts[
            batch_idx * batch_seeds.numel() : batch_idx * batch_seeds.numel() + batch_seeds.numel()
        ]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [len(batch_seeds), backend.img_channels, backend.img_resolution, backend.img_resolution],
            device=device,
            dtype=backend.pipeline.transformer.dtype,
        )

        condition = backend.prepare_condition(
            prompt=batch_prompts,
            negative_prompt=None if predictor_module.guidance_rate == 1.0 else [""] * len(batch_prompts),
            guidance_scale=predictor_module.guidance_rate,
        )

        with torch.no_grad():
            images, _ = epd_sampler(
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
                predictor=predictor_module,
                afs=False,
                return_inters=False,
                train=False,
            )
            images = backend.vae_decode(images)

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

    print("SD3 EPD sampling done.")


if __name__ == "__main__":
    main()
