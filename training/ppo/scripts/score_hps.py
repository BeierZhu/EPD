"""
Utility CLI to compute HPS scores for a directory of generated images.

Usage example:
    python -m training.ppo.scripts.score_hps \\
        --images exps/<run-id>/samples \\
        --prompts prompts.csv \\
        --weights weights/HPS_v2.1_compressed.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

from training.ppo.reward_hps import RewardHPS, RewardHPSConfig

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm.*")


def _load_prompts(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            prompts = [row.get("text", "").strip() for row in reader if row.get("text")]
    else:
        with path.open("r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise RuntimeError(f"No prompts could be read from {path}.")
    return prompts


def _collect_images(directory: Path, pattern: str) -> List[Path]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise RuntimeError(f"No image files matching {pattern} were found under {directory}.")
    return files


def _build_reward(args: argparse.Namespace) -> RewardHPS:
    device = torch.device(args.device)
    cfg = RewardHPSConfig(
        device=device,
        batch_size=args.batch_size,
        enable_amp=not args.disable_amp,
        weights_path=args.weights,
        cache_dir=args.cache_dir,
    )
    return RewardHPS(cfg)


def _summarize(scores: torch.Tensor) -> dict:
    stats = {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
    }
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute HPS scores for generated images.")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing generated images.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern to match image files (default: *.png).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Prompt list file (CSV must include a text column, or plain text with one prompt per line).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/HPS_v2.1_compressed.pt"),
        help="Path to HPSv2 weights.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("weights"),
        help="Cache directory for HPS weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run HPS evaluation.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable AMP and force FP32 evaluation.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional: write results to a JSON file.",
    )

    args = parser.parse_args(argv)
    images_dir = args.images.resolve()
    prompts_path = args.prompts.resolve()

    image_files = _collect_images(images_dir, args.pattern)
    prompts = _load_prompts(prompts_path)

    if len(image_files) != len(prompts):
        raise RuntimeError(
            f"Number of images ({len(image_files)}) does not match number of prompts ({len(prompts)})."
        )

    reward = _build_reward(args)
    scores, meta = reward.score_paths(image_files, prompts, return_metadata=True)
    stats = _summarize(scores)

    result = {
        "images_dir": str(images_dir),
        "pattern": args.pattern,
        "prompts_file": str(prompts_path),
        "stats": stats,
        "metadata": meta,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
