"""
HPSv2-based reward adapter for PPO training.

This module exposes a lightweight wrapper around the official HPSv2
implementation so that rollout batches can be evaluated on-device and
batched efficiently.
"""

import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

HPS_LOCAL_PATH = Path(__file__).resolve().parents[1] / "reward_models" / "HPSv2"
if HPS_LOCAL_PATH.exists() and str(HPS_LOCAL_PATH) not in sys.path:
    sys.path.insert(0, str(HPS_LOCAL_PATH))

try:
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import root_path as HPS_ROOT_DEFAULT, hps_version_map
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import HPSv2 package. Ensure the local HPSv2 repo is on PYTHONPATH."
    ) from exc

from huggingface_hub import hf_hub_download


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


@dataclass
class RewardHPSConfig:
    """Static configuration for the HPS reward evaluator."""

    device: Optional[torch.device] = None
    precision: torch.dtype = torch.float32
    batch_size: int = 8
    hps_version: str = "v2.1"
    weights_path: Optional[Union[str, Path]] = None
    cache_dir: Optional[Union[str, Path]] = None
    enable_amp: bool = True
    image_value_range: Tuple[float, float] = (0.0, 1.0)

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardHPS:
    """Evaluate rollout batches with the Human Preference Score v2 model."""

    def __init__(self, config: RewardHPSConfig) -> None:
        self.config = config
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded_version = None
        self._weights_path = None
        self._device = self.config.resolve_device()
        self._amp_dtype = torch.float16 if self.config.enable_amp else torch.float32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_tensor(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Score a batch of images provided as tensors."""

        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [N,C,H,W], got {images.shape}.")
        if images.shape[0] != len(prompts):
            raise ValueError("Number of images and prompts must match.")

        self._ensure_model_loaded()

        scores: List[torch.Tensor] = []
        batch = batch_size or self.config.batch_size
        start = time.time()

        for offset in range(0, len(prompts), batch):
            chunk_prompts = prompts[offset : offset + batch]
            chunk_tensor = images[offset : offset + len(chunk_prompts)]
            chunk_scores = self._score_chunk(chunk_tensor, chunk_prompts)
            scores.append(chunk_scores)

        scores_tensor = torch.cat(scores, dim=0)
        duration = time.time() - start

        if return_metadata:
            metadata = {
                "duration": duration,
                "num_images": len(prompts),
                "batch_size": batch,
                "device": str(self._device),
            }
            return scores_tensor, metadata
        return scores_tensor

    def score_paths(
        self,
        paths: Sequence[Union[str, Path]],
        prompts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Score images given as filesystem paths."""

        pil_images = [Image.open(path).convert("RGB") for path in paths]
        return self.score_pil(
            pil_images,
            prompts,
            batch_size=batch_size,
            return_metadata=return_metadata,
        )

    def score_pil(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Score images provided as in-memory PIL images."""

        tensors = torch.stack([pil_to_tensor(img.convert("RGB")) for img in images]).to(dtype=torch.float32)
        tensors = tensors / 255.0
        return self.score_tensor(
            tensors,
            prompts,
            batch_size=batch_size,
            return_metadata=return_metadata,
        )

    def clear(self) -> None:
        """Release cached model to free memory."""

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
        weights_path = (
            Path(self.config.weights_path)
            if self.config.weights_path is not None
            else None
        )

        if weights_path is None:
            cache_root = (
                Path(cache_dir)
                if cache_dir is not None
                else Path(os.environ.get("HPS_ROOT", HPS_ROOT_DEFAULT))
            )
            cache_root.mkdir(parents=True, exist_ok=True)
            version_key = hps_version_map[self.config.hps_version]
            resolved_path: Optional[str] = None
            if _dist_ready():
                if dist.get_rank() == 0:
                    resolved_path = hf_hub_download(
                        "xswu/HPSv2",
                        version_key,
                        cache_dir=str(cache_root),
                    )
                dist.barrier()
                if dist.get_rank() != 0:
                    resolved_path = hf_hub_download(
                        "xswu/HPSv2",
                        version_key,
                        cache_dir=str(cache_root),
                        local_files_only=True,
                    )
            else:
                resolved_path = hf_hub_download(
                    "xswu/HPSv2",
                    version_key,
                    cache_dir=str(cache_root),
                )
            if resolved_path is None:
                raise RuntimeError("Failed to resolve HPS weights path.")
            weights_path = Path(resolved_path)
        else:
            weights_path = Path(weights_path)
            if _dist_ready():
                dist.barrier()

        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            device=self._device,
            precision="amp" if self.config.enable_amp else "fp32",
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        state = torch.load(weights_path, map_location=self._device)
        model.load_state_dict(state["state_dict"], strict=True)
        model = model.to(self._device)
        model.eval()

        tokenizer = get_tokenizer("ViT-H-14")

        self._model = model
        self._preprocess = preprocess_val
        self._tokenizer = tokenizer
        self._weights_path = str(weights_path)
        self._loaded_version = self.config.hps_version

    def _score_chunk(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        assert self._model is not None
        assert self._preprocess is not None
        assert self._tokenizer is not None

        device = self._device
        model = self._model

        if images.device != device:
            chunk = images.to(device)
        else:
            chunk = images

        chunk = chunk.to(dtype=torch.float32)
        min_val, max_val = self.config.image_value_range
        chunk = torch.clamp(chunk, min=min_val, max=max_val)

        pil_images = [to_pil_image(img.cpu()) for img in chunk]
        processed = torch.stack([self._preprocess(img) for img in pil_images]).to(device)
        tokenized = self._tokenizer(list(prompts)).to(device)

        if self.config.enable_amp and device.type == "cuda":
            autocast_ctx = torch.cuda.amp.autocast(dtype=self._amp_dtype)
        else:
            autocast_ctx = nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                # Use the bound __call__ so tests that patch the instance method with a
                # mock still intercept this invocation.
                model_callable = getattr(model, "__call__", None)
                if callable(model_callable):
                    outputs = model_callable(processed, tokenized)
                else:
                    outputs = model(processed, tokenized)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logits = image_features @ text_features.T
                chunk_scores = torch.diagonal(logits).detach().cpu()
        return chunk_scores
