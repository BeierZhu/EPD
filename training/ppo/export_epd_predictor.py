"""
Stage 8: Export PPO 策略的 Dirichlet 均值为标准 EPD predictor。

本模块提供：
    * 将 `EPDParamPolicy` 的均值表转换成 `training.networks.EPD_predictor`
      权重结构（含必要的 logits / 参数）。
    * 根据 Stage 7 run-dir 读取配置、checkpoint，并生成网络快照 +
      `training_options.json` 摘要 + manifest。
    * 命令行接口，方便独立执行导出流程后交由 `sample.py` 回放。

注意：本文件仅负责生成所需文件；不负责实际运行 sample 或测试。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml

from training.networks import EPD_predictor
from training.ppo.cold_start import (
    EPDTable,
    _sanitize_table_arrays,
    build_dirichlet_params,
    load_predictor_table,
)
from training.ppo.pipeline_utils import resolve_flux_runtime_metadata
from training.ppo.policy import EPDParamPolicy


# ---------------------------------------------------------------------------
# Exceptions & dataclasses


class ExportError(RuntimeError):
    """Raised when export prerequisites are missing or inconsistent."""


@dataclass
class ExportResult:
    """Paths written during a successful export."""

    snapshot_path: Path
    training_options_path: Path
    manifest_path: Optional[Path]


# ---------------------------------------------------------------------------
# Core helpers


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _load_resolved_config(run_dir: Path) -> Mapping[str, object]:
    config_path = run_dir / "configs" / "resolved_config.yaml"
    if not config_path.is_file():
        raise ExportError(f"未找到 Stage 7 配置文件: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ExportError(f"解析配置失败: {config_path}")
    return data


def _load_training_options_template(snapshot_path: Path) -> Dict[str, object]:
    candidate = snapshot_path.parent / "training_options.json"
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError as exc:  # pragma: no cover - unexpected
                raise ExportError(f"无法解析 {candidate}: {exc}") from exc
    # 构造最小模板
    return {
        "loss_kwargs": {"class_name": "training.loss.EPD_loss"},
        "pred_kwargs": {"class_name": "training.networks.EPD_predictor"},
    }


def _find_latest_policy_checkpoint(run_dir: Path) -> Optional[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    pattern = re.compile(r"^policy-(?:step)?(\d+)\.pt$")
    candidates: List[Tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        step = int(match.group(1))
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _parse_step_from_name(name: str) -> Optional[int]:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def _infer_policy_arch_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    arch: Dict[str, int] = {}
    embed = state_dict.get("step_embed.weight")
    if isinstance(embed, torch.Tensor):
        arch["hidden_dim"] = embed.shape[1]
    block_indices = []
    pattern = re.compile(r"^blocks\.(\d+)\.")
    for key in state_dict.keys():
        match = pattern.match(key)
        if match:
            block_indices.append(int(match.group(1)))
    if block_indices:
        arch["num_layers"] = max(block_indices) + 1
    context_w = state_dict.get("context_proj.weight")
    if isinstance(context_w, torch.Tensor):
        arch["context_dim"] = context_w.shape[1]
    return arch


def _infer_policy_scale_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Tuple[bool, bool]:
    use_scale_dir = "base_log_scale_dir" in state_dict
    use_scale_time = "base_log_scale_time" in state_dict
    return use_scale_dir, use_scale_time


def _instantiate_policy(
    table: EPDTable,
    config: Mapping[str, object],
    device: torch.device,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> EPDParamPolicy:
    ppo_cfg = config.get("ppo", {}) if isinstance(config, Mapping) else {}
    dirichlet_c = float(ppo_cfg.get("dirichlet_concentration", 200.0))
    scale_std = float(ppo_cfg.get("scale_std", 0.05))

    hidden_dim = 128
    num_layers = 2
    context_dim = 0
    use_scale_dir = False
    use_scale_time = False

    if isinstance(state_dict, Mapping):
        inferred = _infer_policy_arch_from_state_dict(state_dict)
        hidden_dim = int(inferred.get("hidden_dim", hidden_dim))
        num_layers = int(inferred.get("num_layers", num_layers))
        context_dim = int(inferred.get("context_dim", context_dim))
        use_scale_dir, use_scale_time = _infer_policy_scale_from_state_dict(state_dict)
    else:
        use_scale_dir = bool(ppo_cfg.get("train_scale_dir", False) or table.scale_dir is not None)
        use_scale_time = bool(ppo_cfg.get("train_scale_time", False) or table.scale_time is not None)

    init = build_dirichlet_params(table, concentration=dirichlet_c)
    policy = EPDParamPolicy(
        num_steps=table.num_steps,
        num_points=table.num_points,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        context_dim=context_dim,
        dirichlet_init=init,
        use_scale_dir=use_scale_dir,
        use_scale_time=use_scale_time,
        scale_log_std_init=math.log(scale_std),
    ).to(device)
    return policy


def _calc_policy_means(policy: EPDParamPolicy, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    was_training = policy.training
    policy = policy.to(device)
    policy.eval()
    with torch.no_grad():
        step_idx = torch.arange(policy.num_steps - 1, device=device, dtype=torch.long)
        output = policy(step_idx)
        positions, weights = policy.mean_table(output)
    if was_training:
        policy.train()
    return positions, weights


def _calc_policy_scales(
    policy: EPDParamPolicy, device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    was_training = policy.training
    policy = policy.to(device)
    policy.eval()
    with torch.no_grad():
        step_idx = torch.arange(policy.num_steps - 1, device=device, dtype=torch.long)
        output = policy(step_idx)
        scale_dir, scale_time = policy.mean_scales(output)
    if was_training:
        policy.train()
    return scale_dir, scale_time


def _sanitize_tables(
    positions: torch.Tensor,
    weights: torch.Tensor,
    scale_dir: Optional[torch.Tensor] = None,
    scale_time: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, int]]:
    pos_np = positions.detach().cpu().numpy()
    weight_np = weights.detach().cpu().numpy()
    scale_dir_np = scale_dir.detach().cpu().numpy() if scale_dir is not None else None
    scale_time_np = scale_time.detach().cpu().numpy() if scale_time is not None else None
    pos_np, weight_np, scale_dir_np, scale_time_np, reordered, adjusted = _sanitize_table_arrays(
        pos_np, weight_np, scale_dir_np, scale_time_np
    )
    meta = {
        "reordered_rows": int(np.count_nonzero(reordered)),
        "adjusted_rows": int(np.count_nonzero(adjusted)),
    }
    pos_tensor = torch.from_numpy(pos_np).to(positions.device)
    weight_tensor = torch.from_numpy(weight_np).to(weights.device)
    scale_dir_tensor = torch.from_numpy(scale_dir_np).to(positions.device) if scale_dir_np is not None else None
    scale_time_tensor = (
        torch.from_numpy(scale_time_np).to(positions.device) if scale_time_np is not None else None
    )
    return pos_tensor, weight_tensor, scale_dir_tensor, scale_time_tensor, meta


def _weights_to_logits(weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    clamped = torch.clamp(weights, min=eps)
    return torch.log(clamped)


def _positions_to_logits(positions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    clamped = torch.clamp(positions, min=eps, max=1 - eps)
    return torch.logit(clamped)


def _scale_to_params(scale: torch.Tensor, scale_range: float, eps: float = 1e-6) -> torch.Tensor:
    if scale_range <= 0:
        return torch.zeros_like(scale)
    normalized = (scale - (1 - scale_range)) / (2 * scale_range)
    normalized = torch.clamp(normalized, min=eps, max=1 - eps)
    return 2.0 * torch.logit(normalized)


def _prepare_pred_kwargs(
    base_options: Mapping[str, object],
    run_config: Mapping[str, object],
    table: EPDTable,
) -> Dict[str, object]:
    pred_kwargs = dict(base_options.get("pred_kwargs", {}))
    pred_kwargs.setdefault("class_name", "training.networks.EPD_predictor")

    model_cfg = run_config.get("model", {}) if isinstance(run_config, Mapping) else {}
    table_meta = table.metadata if isinstance(table.metadata, Mapping) else {}
    pred_kwargs["num_steps"] = int(table.num_steps)
    pred_kwargs["num_points"] = int(table.num_points)
    pred_kwargs["dataset_name"] = _first_non_none(
        model_cfg.get("dataset_name"),
        pred_kwargs.get("dataset_name"),
        table_meta.get("dataset_name"),
    )
    pred_kwargs["guidance_type"] = _first_non_none(
        model_cfg.get("guidance_type"),
        pred_kwargs.get("guidance_type"),
        table_meta.get("guidance_type"),
    )
    pred_kwargs["guidance_rate"] = _first_non_none(
        model_cfg.get("guidance_rate"),
        pred_kwargs.get("guidance_rate"),
        table_meta.get("guidance_rate"),
    )
    pred_kwargs["schedule_type"] = _first_non_none(
        model_cfg.get("schedule_type"),
        pred_kwargs.get("schedule_type"),
        table_meta.get("schedule_type"),
    )
    pred_kwargs["schedule_rho"] = _first_non_none(
        model_cfg.get("schedule_rho"),
        pred_kwargs.get("schedule_rho"),
        table_meta.get("schedule_rho"),
    )
    pred_kwargs["sigma_min"] = _first_non_none(
        model_cfg.get("sigma_min"),
        pred_kwargs.get("sigma_min"),
        table_meta.get("sigma_min"),
    )
    pred_kwargs["sigma_max"] = _first_non_none(
        model_cfg.get("sigma_max"),
        pred_kwargs.get("sigma_max"),
        table_meta.get("sigma_max"),
    )
    pred_kwargs["flowmatch_mu"] = _first_non_none(
        model_cfg.get("flowmatch_mu"),
        pred_kwargs.get("flowmatch_mu"),
        table_meta.get("flowmatch_mu"),
    )
    pred_kwargs["flowmatch_shift"] = _first_non_none(
        model_cfg.get("flowmatch_shift"),
        pred_kwargs.get("flowmatch_shift"),
        table_meta.get("flowmatch_shift"),
    )
    pred_kwargs["afs"] = bool(pred_kwargs.get("afs", False))
    pred_kwargs["scale_dir"] = float(pred_kwargs.get("scale_dir", 0.0))
    pred_kwargs["scale_time"] = float(pred_kwargs.get("scale_time", 0.0))
    pred_kwargs["predict_x0"] = bool(pred_kwargs.get("predict_x0", False))
    pred_kwargs["lower_order_final"] = bool(pred_kwargs.get("lower_order_final", True))
    pred_kwargs["sampler_stu"] = pred_kwargs.get("sampler_stu", "epd")
    pred_kwargs["sampler_tea"] = pred_kwargs.get("sampler_tea", "dpm")
    pred_kwargs["M"] = pred_kwargs.get("M", 1)
    pred_kwargs["alpha"] = pred_kwargs.get("alpha", 10.0)
    pred_kwargs["fcn"] = bool(pred_kwargs.get("fcn", False))
    pred_kwargs["max_order"] = pred_kwargs.get("max_order", 2)
    backend = _first_non_none(model_cfg.get("backend"), pred_kwargs.get("backend"), table_meta.get("backend"), "ldm")
    if backend is None:
        backend = "ldm"
    pred_kwargs["backend"] = str(backend)
    backend_options = model_cfg.get("backend_options", pred_kwargs.get("backend_config", {}))
    if backend_options is None:
        backend_options = {}
    elif not isinstance(backend_options, Mapping):
        raise ExportError("model.backend_options must be a mapping when provided.")
    else:
        backend_options = dict(backend_options)
    res_meta = _first_non_none(
        model_cfg.get("resolution") if isinstance(model_cfg, Mapping) else None,
        pred_kwargs.get("img_resolution"),
        table_meta.get("img_resolution"),
        table_meta.get("resolution"),
    )
    if res_meta is not None:
        try:
            backend_options.setdefault("resolution", int(res_meta))
        except Exception:
            raise ExportError("model.resolution must be an integer when provided.")
    pred_kwargs["backend_config"] = backend_options
    if res_meta is None:
        res_meta = _first_non_none(backend_options.get("resolution"), table_meta.get("img_resolution"), table_meta.get("resolution"))
    if res_meta is not None:
        try:
            pred_kwargs["img_resolution"] = int(res_meta)
        except Exception:
            raise ExportError("model.resolution must be an integer when provided.")
    if str(pred_kwargs["backend"]).lower() == "flux":
        resolved_flux = resolve_flux_runtime_metadata(
            backend_options=backend_options,
            resolution=_first_non_none(pred_kwargs.get("img_resolution"), table_meta.get("img_resolution"), table_meta.get("resolution")),
            sigma_min=_first_non_none(pred_kwargs.get("sigma_min"), table_meta.get("sigma_min")),
            sigma_max=_first_non_none(pred_kwargs.get("sigma_max"), table_meta.get("sigma_max")),
            flowmatch_mu=_first_non_none(pred_kwargs.get("flowmatch_mu"), table_meta.get("flowmatch_mu")),
            flowmatch_shift=_first_non_none(pred_kwargs.get("flowmatch_shift"), table_meta.get("flowmatch_shift")),
        )
        pred_kwargs["backend_config"] = dict(resolved_flux["backend_options"])
        pred_kwargs["img_resolution"] = resolved_flux["resolution"]
        pred_kwargs["sigma_min"] = resolved_flux["sigma_min"]
        pred_kwargs["sigma_max"] = resolved_flux["sigma_max"]
        pred_kwargs["flowmatch_mu"] = resolved_flux["flowmatch_mu"]
        pred_kwargs["flowmatch_shift"] = resolved_flux["flowmatch_shift"]
    return pred_kwargs


def _instantiate_predictor(pred_kwargs: Mapping[str, object]) -> EPD_predictor:
    kwargs = dict(pred_kwargs)
    kwargs.pop("class_name", None)
    return EPD_predictor(**kwargs)


def _read_latest_metrics(run_dir: Path) -> Optional[Mapping[str, object]]:
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    if not metrics_path.is_file():
        return None
    last_line: Optional[str] = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def _format_manifest(
    run_dir: Path,
    checkpoint: Path,
    step: Optional[int],
    sanitize_meta: Mapping[str, int],
    metrics: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    manifest = {
        "run_dir": str(run_dir),
        "policy_checkpoint": str(checkpoint),
        "export_step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sanitized_rows": {
            "reordered": int(sanitize_meta.get("reordered_rows", 0)),
            "adjusted": int(sanitize_meta.get("adjusted_rows", 0)),
        },
    }
    if metrics:
        manifest["latest_metrics"] = {
            key: metrics.get(key)
            for key in ("step", "mixed_reward_mean", "mixed_reward_std", "kl", "policy_loss", "elapsed_sec")
            if key in metrics
        }
    return manifest


def _write_snapshot(predictor: EPD_predictor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {"model": predictor.cpu()}
    with output_path.open("wb") as handle:
        pickle.dump(snapshot, handle)


def _write_training_options(payload: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_manifest(payload: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Public export API


def export_policy_mean_to_predictor(
    run_dir: Path,
    checkpoint: Optional[Path] = None,
    *,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    include_manifest: bool = True,
) -> ExportResult:
    """
    将给定 run-dir 下的策略 checkpoint 导出为 EPD predictor。

    Parameters
    ----------
    run_dir:
        Stage 7 训练输出目录 (`exps/<timestamp>-<run_name>`).
    checkpoint:
        策略参数 (`.pt`)；若未提供则自动选取 `checkpoints/policy-step*.pt` 中步数最大的文件。
    output_dir:
        导出文件保存目录；默认为 run_dir / "export".
    device:
        加载策略/计算均值时使用的 PyTorch 设备字符串；默认 `cpu`。
    include_manifest:
        是否写出 manifest JSON。
    """

    run_dir = Path(run_dir).resolve()
    if checkpoint is None:
        checkpoint = _find_latest_policy_checkpoint(run_dir)
        if checkpoint is None:
            raise ExportError("未找到策略 checkpoint（期待 checkpoints/policy-stepXXXXX.pt）。")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    if not checkpoint.is_file():
        raise ExportError(f"checkpoint 不存在: {checkpoint}")

    config_dict = _load_resolved_config(run_dir)
    data_cfg = config_dict.get("data", {})
    snapshot_path_str = data_cfg.get("predictor_snapshot")
    if not snapshot_path_str:
        raise ExportError("配置中缺少 data.predictor_snapshot 字段。")
    snapshot_path = Path(snapshot_path_str).expanduser()
    if not snapshot_path.is_file():
        raise ExportError(f"指定的 predictor 快照不存在: {snapshot_path}")

    table = load_predictor_table(snapshot_path, map_location="cpu", allow_scale=False)

    torch_device = torch.device(device or "cpu")

    state_dict = torch.load(checkpoint, map_location=torch_device)
    if not isinstance(state_dict, Mapping):
        raise ExportError(f"载入的 checkpoint 不是 state_dict: {checkpoint}")

    policy = _instantiate_policy(table, config_dict, torch_device, state_dict=state_dict)
    policy.load_state_dict(state_dict, strict=True)

    positions, weights = _calc_policy_means(policy, torch_device)
    scale_dir_mean, scale_time_mean = _calc_policy_scales(policy, torch_device)
    positions, weights, scale_dir_mean, scale_time_mean, sanitize_meta = _sanitize_tables(
        positions, weights, scale_dir_mean, scale_time_mean
    )

    r_logits = _positions_to_logits(positions)
    weight_logits = _weights_to_logits(weights)

    base_options = _load_training_options_template(snapshot_path)
    pred_kwargs = _prepare_pred_kwargs(base_options, config_dict, table)
    scale_dir_range = float(pred_kwargs.get("scale_dir", 0.0))
    scale_time_range = float(pred_kwargs.get("scale_time", 0.0))

    predictor = _instantiate_predictor(pred_kwargs)
    predictor = predictor.to(torch_device)
    with torch.no_grad():
        predictor.r_params.copy_(r_logits)
        predictor.weight_s.copy_(weight_logits)
        if hasattr(predictor, "scale_dir_params"):
            if scale_dir_mean is not None and scale_dir_range > 0:
                predictor.scale_dir_params.copy_(_scale_to_params(scale_dir_mean, scale_dir_range))
            else:
                predictor.scale_dir_params.zero_()
        if hasattr(predictor, "scale_time_params"):
            if scale_time_mean is not None and scale_time_range > 0:
                predictor.scale_time_params.copy_(_scale_to_params(scale_time_mean, scale_time_range))
            else:
                predictor.scale_time_params.zero_()
    predictor.sigma_min = pred_kwargs.get("sigma_min")
    predictor.sigma_max = pred_kwargs.get("sigma_max")
    predictor.flowmatch_mu = pred_kwargs.get("flowmatch_mu")
    predictor.flowmatch_shift = pred_kwargs.get("flowmatch_shift")

    training_options = dict(base_options)
    training_options["pred_kwargs"] = pred_kwargs
    training_options["export_info"] = {
        "source_run": str(run_dir),
        "policy_checkpoint": str(checkpoint),
        "sanitized_rows": sanitize_meta,
    }

    export_dir = Path(output_dir) if output_dir is not None else (run_dir / "export")
    export_dir.mkdir(parents=True, exist_ok=True)

    step = _parse_step_from_name(checkpoint.name)
    suffix = f"step{step:06d}" if step is not None else "latest"
    snapshot_path_out = export_dir / f"network-snapshot-export-{suffix}.pkl"
    training_options_path_out = export_dir / f"training_options-export-{suffix}.json"

    _write_snapshot(predictor, snapshot_path_out)
    _write_training_options(training_options, training_options_path_out)

    manifest_path: Optional[Path] = None
    if include_manifest:
        metrics = _read_latest_metrics(run_dir)
        manifest = _format_manifest(run_dir, checkpoint, step, sanitize_meta, metrics)
        manifest_path = export_dir / f"export-manifest-{suffix}.json"
        _write_manifest(manifest, manifest_path)

    return ExportResult(
        snapshot_path=snapshot_path_out,
        training_options_path=training_options_path_out,
        manifest_path=manifest_path,
    )


# ---------------------------------------------------------------------------
# CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 PPO 策略导出为 EPD predictor。")
    parser.add_argument("run_dir", type=Path, help="Stage 7 训练输出目录（含 configs/、logs/ 等子目录）。")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="策略 checkpoint (*.pt)，默认选择 checkpoints/ 下最新的 policy-step*.pt。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="导出文件保存路径（默认: <run_dir>/export）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="加载策略与计算均值的 PyTorch 设备 (默认: cpu)。",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="不写出 manifest JSON。",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = export_policy_mean_to_predictor(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        include_manifest=not args.no_manifest,
    )
    print("[Stage 8] 导出完成：")
    print(f"  snapshot: {result.snapshot_path}")
    print(f"  training_options: {result.training_options_path}")
    if result.manifest_path:
        print(f"  manifest: {result.manifest_path}")


if __name__ == "__main__":  # pragma: no cover
    main()


'''

python -m training.ppo.export_epd_predictor exps/20251222-171726-sd15_k20 \
    --checkpoint checkpoints/policy-step000040.pt


'''
