from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration values are invalid."""


def _parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if lowered.startswith("0x"):
            return int(value, 16)
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_override(raw: Dict[str, Any], key: str, value: str) -> None:
    parts = key.split(".")
    node = raw
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = _parse_override_value(value)


def load_raw_config(path: Path, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML structure must be a mapping.")
    merged = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must be in key=value form: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ConfigError(f"Override key is empty: {item}")
        _apply_override(merged, key, value)
    return merged


@dataclass
class RunConfig:
    output_root: Path
    run_name: str
    seed: int
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_id: Optional[str] = None
    run_dir: Optional[Path] = None

    def assign_run_dir(self) -> None:
        if self.run_id is None:
            self.run_id = f"{self.timestamp}-{self.run_name}"
        self.run_dir = (self.output_root / self.run_id).resolve()


@dataclass
class DataConfig:
    predictor_snapshot: Path
    prompt_csv: Optional[Path] = None


@dataclass
class ModelConfig:
    dataset_name: str
    guidance_type: str
    guidance_rate: float
    schedule_type: str
    schedule_rho: float
    num_steps: Optional[int] = None
    num_points: Optional[int] = None
    backend: str = "ldm"
    resolution: Optional[int] = None
    backend_options: Dict[str, Any] = field(default_factory=dict)
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    flowmatch_mu: Optional[float] = None
    flowmatch_shift: Optional[float] = None


@dataclass
class RewardMultiPickScoreConfig:
    model: str
    processor: str


@dataclass
class RewardMultiImageRewardConfig:
    checkpoint: Optional[Path] = None
    med_config: Optional[Path] = None
    cache_dir: Optional[Path] = None


@dataclass
class RewardMultiClipConfig:
    cache_dir: Optional[Path] = None


@dataclass
class RewardMultiAestheticConfig:
    predictor_path: Path
    clip_path: Optional[Path] = None


@dataclass
class RewardMultiWeightsConfig:
    hps: float = 1.0
    pickscore: float = 1.0
    imagereward: float = 1.0
    clip: float = 1.0
    aesthetic: float = 1.0


@dataclass
class RewardMultiConfig:
    weights: RewardMultiWeightsConfig
    pickscore: RewardMultiPickScoreConfig
    imagereward: RewardMultiImageRewardConfig
    clip: RewardMultiClipConfig
    aesthetic: RewardMultiAestheticConfig


@dataclass
class RewardConfig:
    type: str
    weights_path: Path
    batch_size: int
    enable_amp: bool
    hps_version: str = "v2.1"
    cache_dir: Optional[Path] = None
    multi: Optional[RewardMultiConfig] = None


@dataclass
class PPOConfig:
    rollout_batch_size: int
    rloo_k: int
    ppo_epochs: int
    minibatch_size: int
    learning_rate: float
    clip_range: float
    kl_coef: float
    entropy_coef: float
    max_grad_norm: Optional[float]
    decode_rgb: bool
    steps: int
    dirichlet_concentration: float


@dataclass
class LoggingConfig:
    log_interval: int
    save_interval: int


@dataclass
class FullConfig:
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    reward: RewardConfig
    ppo: PPOConfig
    logging: LoggingConfig
    raw: Dict[str, Any] = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        run_dict = asdict(self.run)
        run_dict["output_root"] = str(run_dict["output_root"])
        if run_dict.get("run_dir") is not None:
            run_dict["run_dir"] = str(run_dict["run_dir"])
        return {
            "run": run_dict,
            "data": {
                "predictor_snapshot": str(self.data.predictor_snapshot),
                "prompt_csv": str(self.data.prompt_csv) if self.data.prompt_csv else None,
            },
            "model": asdict(self.model),
            "reward": {
                "type": self.reward.type,
                "weights_path": str(self.reward.weights_path),
                "batch_size": self.reward.batch_size,
                "enable_amp": self.reward.enable_amp,
                "hps_version": self.reward.hps_version,
                "cache_dir": str(self.reward.cache_dir) if self.reward.cache_dir else None,
                "multi": self._multi_to_dict(),
            },
            "ppo": asdict(self.ppo),
            "logging": asdict(self.logging),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def _multi_to_dict(self) -> Optional[dict]:
        if self.reward.multi is None:
            return None
        multi = self.reward.multi
        return {
            "weights": {
                "hps": multi.weights.hps,
                "pickscore": multi.weights.pickscore,
                "imagereward": multi.weights.imagereward,
                "clip": multi.weights.clip,
                "aesthetic": multi.weights.aesthetic,
            },
            "pickscore": {
                "model": multi.pickscore.model,
                "processor": multi.pickscore.processor,
            },
            "imagereward": {
                "checkpoint": str(multi.imagereward.checkpoint) if multi.imagereward.checkpoint else None,
                "med_config": str(multi.imagereward.med_config) if multi.imagereward.med_config else None,
                "cache_dir": str(multi.imagereward.cache_dir) if multi.imagereward.cache_dir else None,
            },
            "clip": {
                "cache_dir": str(multi.clip.cache_dir) if multi.clip.cache_dir else None,
            },
            "aesthetic": {
                "predictor_path": str(multi.aesthetic.predictor_path),
                "clip_path": str(multi.aesthetic.clip_path) if multi.aesthetic.clip_path else None,
            },
        }


def build_config(raw: Dict[str, Any]) -> FullConfig:
    def require(section: str) -> Dict[str, Any]:
        if section not in raw or not isinstance(raw[section], dict):
            raise ConfigError(f"Missing configuration section: {section}")
        return raw[section]

    run_raw = require("run")
    data_raw = require("data")
    model_raw = require("model")
    reward_raw = require("reward")
    ppo_raw = require("ppo")
    logging_raw = require("logging")

    run = RunConfig(
        output_root=Path(run_raw.get("output_root", "exps")).expanduser(),
        run_name=str(run_raw.get("run_name", "rl_run")),
        seed=int(run_raw.get("seed", 0)),
    )
    data = DataConfig(
        predictor_snapshot=Path(data_raw["predictor_snapshot"]).expanduser(),
        prompt_csv=Path(data_raw["prompt_csv"]).expanduser() if data_raw.get("prompt_csv") else None,
    )
    backend_value = str(model_raw.get("backend", "ldm"))
    backend_options_raw = model_raw.get("backend_options", {})
    if backend_options_raw is None:
        backend_options_raw = {}
    if not isinstance(backend_options_raw, dict):
        raise ConfigError("model.backend_options must be a mapping when provided.")
    backend_options = copy.deepcopy(backend_options_raw)
    if model_raw.get("resolution") is not None:
        backend_options.setdefault("resolution", int(model_raw["resolution"]))

    model = ModelConfig(
        dataset_name=str(model_raw.get("dataset_name", "ms_coco")),
        guidance_type=str(model_raw.get("guidance_type", "cfg")),
        guidance_rate=float(model_raw.get("guidance_rate", 7.5)),
        schedule_type=str(model_raw.get("schedule_type", "discrete")),
        schedule_rho=float(model_raw.get("schedule_rho", 1.0)),
        num_steps=model_raw.get("num_steps"),
        num_points=model_raw.get("num_points"),
        backend=backend_value,
        resolution=int(model_raw["resolution"]) if model_raw.get("resolution") is not None else None,
        backend_options=backend_options,
        sigma_min=float(model_raw["sigma_min"]) if model_raw.get("sigma_min") is not None else None,
        sigma_max=float(model_raw["sigma_max"]) if model_raw.get("sigma_max") is not None else None,
        flowmatch_mu=float(model_raw["flowmatch_mu"]) if model_raw.get("flowmatch_mu") is not None else None,
        flowmatch_shift=float(model_raw["flowmatch_shift"]) if model_raw.get("flowmatch_shift") is not None else None,
    )
    reward_type = str(reward_raw.get("type", "hps")).lower()
    weights_path = Path(reward_raw["weights_path"]).expanduser()
    reward_batch_size = int(reward_raw.get("batch_size", 2))
    reward_enable_amp = bool(reward_raw.get("enable_amp", True))
    hps_version = str(reward_raw.get("hps_version", "v2.1"))
    cache_dir = Path(reward_raw["cache_dir"]).expanduser() if reward_raw.get("cache_dir") else None

    multi_cfg: Optional[RewardMultiConfig] = None
    if reward_type == "multi":
        multi_raw = reward_raw.get("multi")
        if not isinstance(multi_raw, dict):
            raise ConfigError("reward.multi must be provided as a mapping when reward.type=multi.")

        weights_raw = multi_raw.get("weights", {})
        if not isinstance(weights_raw, dict):
            raise ConfigError("reward.multi.weights must be a mapping.")
        weights_cfg = RewardMultiWeightsConfig(
            hps=float(weights_raw.get("hps", 1.0)),
            pickscore=float(weights_raw.get("pickscore", 1.0)),
            imagereward=float(weights_raw.get("imagereward", 1.0)),
            clip=float(weights_raw.get("clip", 1.0)),
            aesthetic=float(weights_raw.get("aesthetic", 1.0)),
        )

        pick_raw = multi_raw.get("pickscore", {})
        if not isinstance(pick_raw, dict):
            raise ConfigError("reward.multi.pickscore must be a mapping.")
        pick_cfg = RewardMultiPickScoreConfig(
            model=str(pick_raw.get("model", "yuvalkirstain/PickScore_v1")),
            processor=str(pick_raw.get("processor", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")),
        )

        image_raw = multi_raw.get("imagereward", {})
        if not isinstance(image_raw, dict):
            raise ConfigError("reward.multi.imagereward must be a mapping.")
        image_cfg = RewardMultiImageRewardConfig(
            checkpoint=Path(image_raw["checkpoint"]).expanduser() if image_raw.get("checkpoint") else None,
            med_config=Path(image_raw["med_config"]).expanduser() if image_raw.get("med_config") else None,
            cache_dir=Path(image_raw["cache_dir"]).expanduser() if image_raw.get("cache_dir") else None,
        )

        clip_raw = multi_raw.get("clip", {})
        if not isinstance(clip_raw, dict):
            raise ConfigError("reward.multi.clip must be a mapping.")
        clip_cfg = RewardMultiClipConfig(
            cache_dir=Path(clip_raw["cache_dir"]).expanduser() if clip_raw.get("cache_dir") else None
        )

        aesthetic_raw = multi_raw.get("aesthetic")
        if not isinstance(aesthetic_raw, dict):
            raise ConfigError("reward.multi.aesthetic must be provided when reward.type=multi.")
        if "predictor_path" not in aesthetic_raw:
            raise ConfigError("reward.multi.aesthetic.predictor_path is required.")
        aesthetic_cfg = RewardMultiAestheticConfig(
            predictor_path=Path(aesthetic_raw["predictor_path"]).expanduser(),
            clip_path=Path(aesthetic_raw["clip_path"]).expanduser() if aesthetic_raw.get("clip_path") else None,
        )

        multi_cfg = RewardMultiConfig(
            weights=weights_cfg,
            pickscore=pick_cfg,
            imagereward=image_cfg,
            clip=clip_cfg,
            aesthetic=aesthetic_cfg,
        )
    elif reward_type != "hps":
        raise ConfigError(f"Unsupported reward.type: {reward_type}")

    reward = RewardConfig(
        type=reward_type,
        weights_path=weights_path,
        batch_size=reward_batch_size,
        enable_amp=reward_enable_amp,
        hps_version=hps_version,
        cache_dir=cache_dir,
        multi=multi_cfg,
    )
    ppo = PPOConfig(
        rollout_batch_size=int(ppo_raw.get("rollout_batch_size", 4)),
        rloo_k=int(ppo_raw.get("rloo_k", 2)),
        ppo_epochs=int(ppo_raw.get("ppo_epochs", 1)),
        minibatch_size=int(ppo_raw.get("minibatch_size", 2)),
        learning_rate=float(ppo_raw.get("learning_rate", 5e-5)),
        clip_range=float(ppo_raw.get("clip_range", 0.2)),
        kl_coef=float(ppo_raw.get("kl_coef", 0.01)),
        entropy_coef=float(ppo_raw.get("entropy_coef", 0.0)),
        max_grad_norm=(
            float(ppo_raw["max_grad_norm"]) if ppo_raw.get("max_grad_norm") is not None else None
        ),
        decode_rgb=bool(ppo_raw.get("decode_rgb", True)),
        steps=int(ppo_raw.get("steps", 10)),
        dirichlet_concentration=float(ppo_raw.get("dirichlet_concentration", 200.0)),
    )
    logging = LoggingConfig(
        log_interval=int(logging_raw.get("log_interval", 1)),
        save_interval=int(logging_raw.get("save_interval", 5)),
    )
    run.assign_run_dir()
    return FullConfig(run=run, data=data, model=model, reward=reward, ppo=ppo, logging=logging, raw=raw)


def validate_config(config: FullConfig, check_paths: bool = True) -> None:
    if config.model.backend.lower() == "sd3":
        res = config.model.resolution
        if res is None:
            res = config.model.backend_options.get("resolution") if isinstance(config.model.backend_options, dict) else None
        if res is not None and res not in (512, 1024):
            raise ConfigError("SD3 resolution must be 512 or 1024.")
    if config.ppo.rollout_batch_size <= 0:
        raise ConfigError("rollout_batch_size must be positive.")
    if config.ppo.rollout_batch_size % config.ppo.rloo_k != 0:
        raise ConfigError("rollout_batch_size must be a multiple of rloo_k.")
    if config.ppo.minibatch_size <= 0 or config.ppo.minibatch_size > config.ppo.rollout_batch_size:
        raise ConfigError("minibatch_size must be in (0, rollout_batch_size].")
    if config.reward.batch_size <= 0 or config.reward.batch_size > config.ppo.rollout_batch_size:
        raise ConfigError("reward.batch_size must be in (0, rollout_batch_size].")
    if config.ppo.steps <= 0:
        raise ConfigError("ppo.steps must be positive.")
    if check_paths:
        if not config.data.predictor_snapshot.is_file():
            raise ConfigError(f"Predictor snapshot not found: {config.data.predictor_snapshot}")
        if not config.reward.weights_path.is_file():
            raise ConfigError(f"HPS weights not found: {config.reward.weights_path}")
        if config.data.prompt_csv and not config.data.prompt_csv.is_file():
            raise ConfigError(f"Prompt CSV not found: {config.data.prompt_csv}")
        if config.reward.type == "multi":
            if config.reward.multi is None:
                raise ConfigError("reward.multi must be set when reward.type=multi.")
            weights = config.reward.multi.weights
            total_weight = (
                max(weights.hps, 0.0)
                + max(weights.pickscore, 0.0)
                + max(weights.imagereward, 0.0)
                + max(weights.clip, 0.0)
                + max(weights.aesthetic, 0.0)
            )
            if total_weight <= 0.0:
                raise ConfigError("The sum of reward.multi.weights must be positive.")
            aesthetic = config.reward.multi.aesthetic
            if not aesthetic.predictor_path.is_file():
                raise ConfigError(f"Aesthetic predictor weights not found: {aesthetic.predictor_path}")
            image_cfg = config.reward.multi.imagereward
            if image_cfg.checkpoint and not image_cfg.checkpoint.is_file():
                raise ConfigError(f"ImageReward checkpoint not found: {image_cfg.checkpoint}")
            if image_cfg.med_config and not image_cfg.med_config.is_file():
                raise ConfigError(f"ImageReward med_config not found: {image_cfg.med_config}")


def pretty_format_config(config: FullConfig) -> str:
    return yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True)
