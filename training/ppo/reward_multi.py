"""
Multi-metric reward adapter that aggregates several preference models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from .reward_hps import RewardHPS, RewardHPSConfig
from .reward_models.aesthetic_predictor_v2 import AestheticV2Model
from .reward_models.imagereward import ImageReward
from .reward_models.imagereward.utils import load as load_imagereward
from .reward_models.pickscore import PickScoreModel


@dataclass
class RewardMetricWeights:
    hps: float = 1.0
    pickscore: float = 1.0
    imagereward: float = 1.0
    clip: float = 1.0
    aesthetic: float = 1.0

    def total(self) -> float:
        return sum(max(weight, 0.0) for weight in (self.hps, self.pickscore, self.imagereward, self.clip, self.aesthetic))


@dataclass
class RewardMultiMetricPaths:
    imagereward_checkpoint: Optional[Union[str, Path]] = None
    imagereward_med_config: Optional[Union[str, Path]] = None
    imagereward_cache_dir: Optional[Union[str, Path]] = None
    clip_cache_dir: Optional[Union[str, Path]] = None
    aesthetic_clip_path: Optional[Union[str, Path]] = None
    aesthetic_predictor_path: Optional[Union[str, Path]] = None


@dataclass
class RewardMultiMetricConfig:
    device: Optional[torch.device] = None
    batch_size: int = 8
    image_value_range: Tuple[float, float] = (0.0, 1.0)
    weights: RewardMetricWeights = field(default_factory=RewardMetricWeights)
    hps: RewardHPSConfig = field(default_factory=RewardHPSConfig)
    pickscore_model_name_or_path: str = "yuvalkirstain/PickScore_v1"
    pickscore_processor_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    paths: RewardMultiMetricPaths = field(default_factory=RewardMultiMetricPaths)

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardMultiMetric:
    """Aggregate multiple reward models into a single normalized signal."""

    def __init__(self, config: RewardMultiMetricConfig) -> None:
        self.config = config
        self._device = config.resolve_device()

        hps_config = config.hps
        if hps_config.device is None:
            hps_config = replace(hps_config, device=self._device)
        self._hps = RewardHPS(hps_config)
        self._hps_config = hps_config

        self._pickscore: Optional[PickScoreModel] = None
        self._imagereward: Optional[ImageReward] = None
        self._clip_model = None
        self._clip_preprocess = None
        self._aesthetic: Optional[AestheticV2Model] = None

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
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [N,C,H,W], got {images.shape}.")
        if images.shape[0] != len(prompts):
            raise ValueError("Number of images and prompts must match.")

        weights = self.config.weights
        total_weight = weights.total()
        if total_weight <= 0.0:
            raise ValueError("At least one reward weight must be positive.")

        images = images.to(torch.float32)
        min_val, max_val = self.config.image_value_range
        images = torch.clamp(images, min=min_val, max=max_val)

        metadata: Dict[str, Dict[str, torch.Tensor]] = {
            "raw_scores": {},
            "normalized_scores": {},
        }

        contributions: List[torch.Tensor] = []
        batch = batch_size or self.config.batch_size

        if weights.hps > 0.0:
            hps_scores = self._hps.score_tensor(images, prompts, batch_size=batch)
            hps_scores = hps_scores.to(dtype=torch.float32)
            metadata["raw_scores"]["hps"] = hps_scores.clone()
            hps_normalized = torch.clamp(hps_scores, 0.0, 1.0)
            metadata["normalized_scores"]["hps"] = hps_normalized.clone()
            contributions.append(hps_normalized * weights.hps)

        pil_images = [to_pil_image(img.cpu()).convert("RGB") for img in images]

        if weights.pickscore > 0.0:
            pick_scores = self._score_pickscore(prompts, pil_images, batch)
            metadata["raw_scores"]["pickscore"] = pick_scores.clone()
            pick_normalized = torch.clamp(pick_scores / 26.0, 0.0, 1.0)
            metadata["normalized_scores"]["pickscore"] = pick_normalized.clone()
            contributions.append(pick_normalized * weights.pickscore)

        if weights.imagereward > 0.0:
            image_rewards = self._score_imagereward(prompts, pil_images)
            metadata["raw_scores"]["imagereward"] = image_rewards.clone()
            image_normalized = torch.clamp(image_rewards / 10.0, 0.0, 1.0)
            metadata["normalized_scores"]["imagereward"] = image_normalized.clone()
            contributions.append(image_normalized * weights.imagereward)

        if weights.clip > 0.0:
            clip_scores = self._score_clip(prompts, pil_images, batch)
            metadata["raw_scores"]["clip"] = clip_scores.clone()
            clip_normalized = torch.clamp(clip_scores, 0.0, 1.0)
            metadata["normalized_scores"]["clip"] = clip_normalized.clone()
            contributions.append(clip_normalized * weights.clip)

        if weights.aesthetic > 0.0:
            aesthetic_scores = self._score_aesthetic(pil_images, batch)
            metadata["raw_scores"]["aesthetic"] = aesthetic_scores.clone()
            aesthetic_normalized = torch.clamp(aesthetic_scores / 10.0, 0.0, 1.0)
            metadata["normalized_scores"]["aesthetic"] = aesthetic_normalized.clone()
            contributions.append(aesthetic_normalized * weights.aesthetic)

        if not contributions:
            raise RuntimeError("No reward contributions were produced; check weight configuration.")

        final = torch.zeros(images.shape[0], dtype=torch.float32)
        for contrib in contributions:
            final += contrib
        final = final / total_weight

        if return_metadata:
            metadata["weights"] = {
                "hps": weights.hps,
                "pickscore": weights.pickscore,
                "imagereward": weights.imagereward,
                "clip": weights.clip,
                "aesthetic": weights.aesthetic,
            }
            return final, metadata
        return final

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_pickscore(self, prompts: Sequence[str], images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        model = self._ensure_pickscore()
        scores: List[float] = []
        for prompt, image in zip(prompts, images):
            output = model([prompt], [image])
            scores.append(float(output[0]))
        return torch.tensor(scores, dtype=torch.float32)

    def _score_imagereward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> torch.Tensor:
        model = self._ensure_imagereward()
        values: List[float] = []
        for prompt, image in zip(prompts, images):
            values.append(float(model.score(prompt, image)))
        return torch.tensor(values, dtype=torch.float32)

    def _score_clip(self, prompts: Sequence[str], images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        clip_model, preprocess = self._ensure_clip()

        scores: List[torch.Tensor] = []
        for offset in range(0, len(prompts), batch_size):
            chunk_prompts = prompts[offset : offset + batch_size]
            chunk_images = images[offset : offset + len(chunk_prompts)]

            text_inputs = clip.tokenize(chunk_prompts, truncate=True).to(self._device)
            image_inputs = torch.stack([preprocess(img) for img in chunk_images], dim=0).to(self._device)

            with torch.no_grad():
                text_features = clip_model.encode_text(text_inputs)
                image_features = clip_model.encode_image(image_inputs)
                text_features = F.normalize(text_features, dim=-1)
                image_features = F.normalize(image_features, dim=-1)
                chunk_scores = (text_features * image_features).sum(dim=-1)

            scores.append(chunk_scores.detach().cpu())

        if scores:
            return torch.cat(scores, dim=0).to(torch.float32)
        return torch.empty(0, dtype=torch.float32)

    def _score_aesthetic(self, images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        model = self._ensure_aesthetic()
        results: List[torch.Tensor] = []
        for offset in range(0, len(images), batch_size):
            chunk = images[offset : offset + batch_size]
            with torch.no_grad():
                chunk_scores = model(chunk).squeeze(-1)
            results.append(chunk_scores.detach().cpu())

        if results:
            return torch.cat(results, dim=0).to(torch.float32)
        return torch.empty(0, dtype=torch.float32)

    def _ensure_pickscore(self) -> PickScoreModel:
        if self._pickscore is None:
            self._pickscore = PickScoreModel(
                device=str(self._device),
                processor_name_or_path=self.config.pickscore_processor_name_or_path,
                model_pretrained_name_or_path=self.config.pickscore_model_name_or_path,
            )
        return self._pickscore

    def _ensure_imagereward(self) -> ImageReward:
        if self._imagereward is None:
            checkpoint = (
                str(self.config.paths.imagereward_checkpoint) if self.config.paths.imagereward_checkpoint else "ImageReward-v1.0"
            )
            cache_dir = (
                str(self.config.paths.imagereward_cache_dir)
                if self.config.paths.imagereward_cache_dir
                else None
            )
            med_config = (
                str(self.config.paths.imagereward_med_config) if self.config.paths.imagereward_med_config else None
            )
            self._imagereward = load_imagereward(
                name=checkpoint,
                device=str(self._device),
                download_root=cache_dir,
                med_config=med_config,
            )
        return self._imagereward

    def _ensure_clip(self):
        if self._clip_model is None or self._clip_preprocess is None:
            download_root = (
                str(self.config.paths.clip_cache_dir) if self.config.paths.clip_cache_dir else None
            )
            self._clip_model, self._clip_preprocess = clip.load(
                "ViT-L/14",
                device=self._device,
                download_root=download_root,
                jit=False,
            )
            if self._device.type != "cpu":
                clip.model.convert_weights(self._clip_model)
            self._clip_model.eval()
        return self._clip_model, self._clip_preprocess

    def _ensure_aesthetic(self) -> AestheticV2Model:
        if self._aesthetic is None:
            predictor_path = self.config.paths.aesthetic_predictor_path
            if predictor_path is None:
                raise RuntimeError("Aesthetic predictor path must be provided when its weight is non-zero.")
            self._aesthetic = AestheticV2Model(
                clip_path=self.config.paths.aesthetic_clip_path,
                predictor_path=str(predictor_path),
                device=self._device,
            )
            self._aesthetic = self._aesthetic.eval()
        return self._aesthetic
