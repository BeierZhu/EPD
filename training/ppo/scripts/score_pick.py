"""
Utility CLI to compute PickScore scores for a directory of generated images.

Usage example:
    python -m training.ppo.scripts.score_pick \\
        --images exps/<run-id>/samples \\
        --prompts prompts.csv \\
        --weights weights/PickScore_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
import os
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image

from training.ppo.reward_models.pickscore import PickScoreModel

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


def _summarize(scores: torch.Tensor) -> dict:
    stats = {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
    }
    return stats


def _build_reward(args: argparse.Namespace):
    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))

    processor_ref = os.environ.get("PICKSCORE_PROCESSOR_PATH", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    weights_path = args.weights.expanduser()
    model_ref = str(weights_path)
    if not weights_path.exists():
        model_ref = os.environ.get("PICKSCORE_MODEL_PATH", "yuvalkirstain/PickScore_v1")

    reward = PickScoreModel(
        device=args.device,
        processor_name_or_path=processor_ref,
        model_pretrained_name_or_path=model_ref,
    )
    return reward, processor_ref, model_ref


def _score(
    reward,
    image_files: Sequence[Path],
    prompts: Sequence[str],
    batch_size: int,
    device_hint: str,
) -> tuple[torch.Tensor, dict]:
    if len(image_files) != len(prompts):
        raise RuntimeError(f"Number of images ({len(image_files)}) does not match number of prompts ({len(prompts)}).")

    values: List[float] = []
    start = time.time()

    for offset in range(0, len(prompts), batch_size):
        chunk_paths = image_files[offset : offset + batch_size]
        chunk_prompts = prompts[offset : offset + batch_size]
        for path, prompt in zip(chunk_paths, chunk_prompts):
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                score = reward(prompt, [image])[0]
            values.append(float(score))

    duration = time.time() - start
    tensor = torch.tensor(values, dtype=torch.float32)
    metadata = {
        "duration": duration,
        "num_images": len(values),
        "batch_size": batch_size,
        "device": str(getattr(reward, "device", device_hint)),
    }
    return tensor, metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute PickScore scores for generated images.")
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
        default=Path("weights/PickScore_v1"),
        help="PickScore model checkpoint path (can be empty; defaults to HuggingFace weights).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("weights"),
        help="Cache directory for model-related files.",
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
        help="Device to run PickScore evaluation.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Kept for score_hps compatibility; PickScore does not support AMP toggling.",
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

    reward, processor_ref, model_ref = _build_reward(args)
    scores, meta = _score(reward, image_files, prompts, args.batch_size, args.device)
    stats = _summarize(scores)

    meta["processor"] = processor_ref
    meta["model"] = model_ref

    result = {
        "images_dir": str(images_dir),
        "pattern": args.pattern,
        "prompts_file": str(prompts_path),
        "stats": stats,
        "metadata": meta,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
