"""
Utilities for reading distilled EPD predictor tables and initializing
Dirichlet priors for PPO cold-start.

Stage 2 focuses on:
  * Loading predictor checkpoints (.pkl/.npz) and extracting table data.
  * Converting positions/weights to Dirichlet means.
  * Constructing concentration vectors with configurable sharpness.
  * Providing fallbacks and round-trip helpers for later validation.

The actual RL stages will import these helpers to seed the policy network
with a distribution that closely matches the distilled baseline.
"""

import dataclasses
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Iterable[float]]
PathLike = Union[str, Path]


@dataclass
class EPDTable:
    """Container for a single EPD distilled parameter table."""

    positions: np.ndarray  # shape: (num_steps-1, num_points)
    weights: np.ndarray  # shape: (num_steps-1, num_points)
    scale_dir: Optional[np.ndarray] = None  # shape matches weights, optional
    scale_time: Optional[np.ndarray] = None  # shape matches weights, optional
    num_steps: int = 0
    num_points: int = 0
    schedule_type: Optional[str] = None
    schedule_rho: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.scale_dir is not None:
            self.scale_dir = np.asarray(self.scale_dir, dtype=np.float64)
        if self.scale_time is not None:
            self.scale_time = np.asarray(self.scale_time, dtype=np.float64)
        if self.num_steps == 0:
            self.num_steps = self.positions.shape[0] + 1
        if self.num_points == 0:
            self.num_points = self.positions.shape[1]
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        expected_shape = (self.num_steps - 1, self.num_points)
        if self.positions.shape != expected_shape:
            raise ValueError(
                f"positions shape {self.positions.shape} does not match expected {expected_shape}"
            )
        if self.weights.shape != expected_shape:
            raise ValueError(
                f"weights shape {self.weights.shape} does not match expected {expected_shape}"
            )
        if self.scale_dir is not None and self.scale_dir.shape != expected_shape:
            raise ValueError(
                f"scale_dir shape {self.scale_dir.shape} does not match expected {expected_shape}"
            )
        if self.scale_time is not None and self.scale_time.shape != expected_shape:
            raise ValueError(
                f"scale_time shape {self.scale_time.shape} does not match expected {expected_shape}"
            )

    def to_json_serializable(self) -> Dict[str, Any]:
        """Return a JSON-ready summary of core statistics."""
        summary = {
            "num_steps": self.num_steps,
            "num_points": self.num_points,
            "schedule_type": self.schedule_type,
            "schedule_rho": self.schedule_rho,
        }
        summary.update(self.metadata)
        return summary


@dataclass
class DirichletInit:
    """Dirichlet concentration tensors derived from a cold-start table."""

    alpha_pos: np.ndarray  # shape: (num_steps-1, num_points+1)
    alpha_weight: np.ndarray  # shape: (num_steps-1, num_points)
    mean_pos_segments: np.ndarray  # same shape as alpha_pos but normalized
    mean_weights: np.ndarray  # same shape as alpha_weight but normalized
    invalid_pos_rows: np.ndarray  # boolean mask over rows with fallback
    invalid_weight_rows: np.ndarray  # boolean mask over rows with fallback
    concentration: float = 0.0


def load_predictor_table(
    checkpoint: PathLike,
    map_location: str = "cpu",
    allow_scale: bool = True,
) -> EPDTable:
    """
    Load a distilled EPD predictor snapshot and extract its parameter table.

    Parameters
    ----------
    checkpoint:
        Path to `.pkl` or `.npz` file. `.pkl` files are expected to contain a
        Torch module (optionally wrapped in DDP/DataParallel) under keys such as
        ``model`` or ``predictor``. `.npz` files must provide arrays named
        ``r``, ``weight`` and optionally ``scale_dir``/``scale_time``.
    map_location:
        Forwarded to `torch.load` when dealing with pickle files.
    allow_scale:
        If False, the returned table will zero out `scale_dir`/`scale_time`
        even when present.
    """

    checkpoint = Path(checkpoint)
    suffix = checkpoint.suffix.lower()
    if suffix not in {".pkl", ".npz"}:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")

    if suffix == ".npz":
        return _load_table_from_npz(checkpoint, allow_scale=allow_scale)
    return _load_table_from_pkl(checkpoint, map_location=map_location, allow_scale=allow_scale)


def positions_to_segments(positions: np.ndarray) -> np.ndarray:
    """
    Convert monotonically increasing positions r (in (0, 1)) to simplex-length segments.

    Returns an array with one extra dimension representing the K+1 segments
    whose cumulative sum reconstructs the original positions.
    """

    positions = np.asarray(positions, dtype=np.float64)
    prefix = np.zeros((*positions.shape[:-1], 1), dtype=np.float64)
    suffix = np.ones((*positions.shape[:-1], 1), dtype=np.float64)
    augmented = np.concatenate([prefix, positions, suffix], axis=-1)
    segments = np.diff(augmented, axis=-1)
    return segments


def segments_to_positions(segments: np.ndarray) -> np.ndarray:
    """
    Inverse of `positions_to_segments`: reconstruct positions from simplex segments.
    """

    segments = np.asarray(segments, dtype=np.float64)
    if segments.shape[-1] < 2:
        raise ValueError("Segments must contain at least two elements to recover positions.")
    cumulative = np.cumsum(segments[..., :-1], axis=-1)
    return cumulative


def normalize_simplex(values: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Project positive values onto the simplex by simple renormalisation."""

    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, eps, np.inf)
    totals = clipped.sum(axis=axis, keepdims=True)
    totals = np.where(totals <= eps, 1.0, totals)
    return clipped / totals


def dirichlet_alpha_from_mean(mean: np.ndarray, concentration: float, eps: float = 1e-12) -> np.ndarray:
    """Construct Dirichlet concentration vectors given a target mean and shared concentration."""

    if concentration <= 0:
        raise ValueError("Dirichlet concentration must be positive.")
    mean = normalize_simplex(mean, eps=eps)
    return mean * float(concentration)


def build_dirichlet_params(
    table: EPDTable,
    concentration: float,
    min_segment: float = 1e-6,
    min_weight: float = 1e-6,
) -> DirichletInit:
    """
    Generate Dirichlet priors for positions and weights based on a cold-start table.

    Parameters
    ----------
    table:
        Parsed EPD table.
    concentration:
        Shared concentration coefficient `c`. Larger `c` yields lower variance.
    min_segment / min_weight:
        Thresholds for detecting degenerate rows. Any row that violates the simplex
        constraints is replaced with a uniform fallback.
    """

    segments = positions_to_segments(table.positions)
    weights = normalize_simplex(table.weights, eps=min_weight)

    invalid_pos = (segments <= min_segment).any(axis=-1)
    invalid_weight = (weights <= min_weight).any(axis=-1)

    pos_mean = segments.copy()
    weight_mean = weights.copy()

    if np.any(invalid_pos):
        uniform_pos = np.full((1, table.num_points + 1), 1.0 / (table.num_points + 1), dtype=np.float64)
        pos_mean[invalid_pos] = uniform_pos
    if np.any(invalid_weight):
        uniform_weight = np.full((1, table.num_points), 1.0 / table.num_points, dtype=np.float64)
        weight_mean[invalid_weight] = uniform_weight

    alpha_pos = dirichlet_alpha_from_mean(pos_mean, concentration)
    alpha_weight = dirichlet_alpha_from_mean(weight_mean, concentration)

    return DirichletInit(
        alpha_pos=alpha_pos,
        alpha_weight=alpha_weight,
        mean_pos_segments=pos_mean,
        mean_weights=weight_mean,
        invalid_pos_rows=invalid_pos,
        invalid_weight_rows=invalid_weight,
        concentration=concentration,
    )


def table_from_dirichlet(init: DirichletInit) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover positions and weights means from Dirichlet concentrations.

    Returns
    -------
    positions, weights :
        Arrays matching the shapes in `EPDTable` (positions without endpoints).
    """

    mean_segments = normalize_simplex(init.alpha_pos)
    mean_weights = normalize_simplex(init.alpha_weight)
    positions = segments_to_positions(mean_segments)
    return positions, mean_weights


def save_dirichlet_summary(
    init: DirichletInit,
    table: EPDTable,
    output_path: PathLike,
) -> None:
    """
    Persist a lightweight JSON summary describing the initialized Dirichlet priors.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "concentration": init.concentration,
        "invalid_rows": {
            "positions": int(np.count_nonzero(init.invalid_pos_rows)),
            "weights": int(np.count_nonzero(init.invalid_weight_rows)),
        },
        "table": table.to_json_serializable(),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers


def _load_table_from_npz(path: Path, allow_scale: bool = True) -> EPDTable:
    data = np.load(path)
    required_keys = {"r", "weight"}
    if not required_keys.issubset(set(data.keys())):
        raise ValueError(f"NPZ file {path} must contain keys {sorted(required_keys)}.")

    positions = np.asarray(data["r"], dtype=np.float64)
    weights = normalize_simplex(np.asarray(data["weight"], dtype=np.float64))

    scale_dir = None
    scale_time = None
    if allow_scale and "scale_dir" in data:
        scale_dir = np.asarray(data["scale_dir"], dtype=np.float64)
    if allow_scale and "scale_time" in data:
        scale_time = np.asarray(data["scale_time"], dtype=np.float64)

    metadata = {key: data[key].item() for key in data.files if key not in {"r", "weight", "scale_dir", "scale_time"}}
    num_steps = metadata.get("num_steps", positions.shape[0] + 1)
    num_points = metadata.get("num_points", positions.shape[1])
    schedule_type = metadata.get("schedule_type")
    schedule_rho = metadata.get("schedule_rho")

    (
        positions,
        weights,
        scale_dir,
        scale_time,
        reordered_rows,
        adjusted_rows,
    ) = _sanitize_table_arrays(positions, weights, scale_dir, scale_time)
    metadata["sanitized"] = bool(reordered_rows.any() or adjusted_rows.any())
    metadata["sanitized_rows"] = int(np.count_nonzero(reordered_rows | adjusted_rows))

    return EPDTable(
        positions=positions,
        weights=weights,
        scale_dir=scale_dir,
        scale_time=scale_time,
        num_steps=int(num_steps),
        num_points=int(num_points),
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        metadata=metadata,
    )


def _load_table_from_pkl(
    path: Path,
    map_location: str = "cpu",
    allow_scale: bool = True,
) -> EPDTable:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required to load EPD predictor pickle files. "
            "Install PyTorch in the current environment before calling load_predictor_table."
        ) from exc

    with path.open("rb") as handle:
        snapshot = pickle.load(handle)

    predictor = _extract_predictor(snapshot)
    if predictor is None:
        raise ValueError(f"Failed to locate predictor module inside {path}")

    predictor = predictor.to(map_location)
    predictor.eval()

    def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default)

    r_params = torch.sigmoid(predictor.r_params.detach())
    weight_logits = predictor.weight_s.detach()
    weights = torch.softmax(weight_logits, dim=-1)

    scale_dir = None
    if allow_scale and _get_attr(predictor, "scale_dir", 0.0):
        raw = predictor.scale_dir_params.detach()
        scale_dir = 2 * torch.sigmoid(0.5 * raw) * predictor.scale_dir + (1 - predictor.scale_dir)
    scale_time = None
    if allow_scale and _get_attr(predictor, "scale_time", 0.0):
        raw = predictor.scale_time_params.detach()
        scale_time = 2 * torch.sigmoid(0.5 * raw) * predictor.scale_time + (1 - predictor.scale_time)

    positions = r_params.cpu().numpy()
    weights = weights.cpu().numpy()
    scale_dir_np = scale_dir.cpu().numpy() if scale_dir is not None else None
    scale_time_np = scale_time.cpu().numpy() if scale_time is not None else None

    (
        positions,
        weights,
        scale_dir_np,
        scale_time_np,
        reordered_rows,
        adjusted_rows,
    ) = _sanitize_table_arrays(positions, weights, scale_dir_np, scale_time_np)

    num_steps = int(_get_attr(predictor, "num_steps", positions.shape[0] + 1))
    num_points = int(_get_attr(predictor, "num_points", positions.shape[1]))
    backend_name = _get_attr(predictor, "backend", None)
    backend_config = _get_attr(predictor, "backend_config", None)
    if isinstance(backend_config, dict):
        backend_config = dict(backend_config)
    else:
        backend_config = None

    metadata = {
        "dataset_name": _get_attr(predictor, "dataset_name", None),
        "sampler_stu": _get_attr(predictor, "sampler_stu", None),
        "guidance_rate": _get_attr(predictor, "guidance_rate", None),
        "guidance_type": _get_attr(predictor, "guidance_type", None),
        "schedule_type": _get_attr(predictor, "schedule_type", None),
        "schedule_rho": _get_attr(predictor, "schedule_rho", None),
        "sigma_min": _get_attr(predictor, "sigma_min", None),
        "sigma_max": _get_attr(predictor, "sigma_max", None),
        "flowmatch_mu": _get_attr(predictor, "flowmatch_mu", None),
        "flowmatch_shift": _get_attr(predictor, "flowmatch_shift", None),
        "afs": _get_attr(predictor, "afs", None),
        "predict_x0": _get_attr(predictor, "predict_x0", None),
        "lower_order_final": _get_attr(predictor, "lower_order_final", None),
        "backend": backend_name,
        "backend_config": backend_config,
        "img_resolution": _get_attr(predictor, "img_resolution", None),
        "resolution": _get_attr(predictor, "img_resolution", None),
        "sanitized": bool(reordered_rows.any() or adjusted_rows.any()),
        "sanitized_rows": int(np.count_nonzero(reordered_rows | adjusted_rows)),
    }

    table = EPDTable(
        positions=positions,
        weights=weights,
        scale_dir=scale_dir_np,
        scale_time=scale_time_np,
        num_steps=num_steps,
        num_points=num_points,
        schedule_type=metadata.get("schedule_type"),
        schedule_rho=metadata.get("schedule_rho"),
        metadata=metadata,
    )
    _assert_monotonic_positions(table.positions)
    return table


def _extract_predictor(snapshot: Any):
    """Retrieve the predictor module from a snapshot structure."""

    try:
        import torch.nn as nn
    except ModuleNotFoundError:
        nn = None  # type: ignore

    def unwrap(module: Any):
        if nn is not None and isinstance(module, nn.Module):
            return getattr(module, "module", module)
        return None

    module = unwrap(snapshot)
    if module is not None:
        return module
    if isinstance(snapshot, dict):
        for key in ("model", "predictor", "student", "ema"):
            if key in snapshot:
                candidate = unwrap(snapshot[key])
                if candidate is not None:
                    return candidate
        # Sometimes the module itself is stored under 'model' without wrapping.
        for value in snapshot.values():
            candidate = unwrap(value)
            if candidate is not None:
                return candidate
    return None


def _assert_monotonic_positions(positions: np.ndarray, eps: float = 1e-5) -> None:
    """Ensure all positions lie inside (0, 1) and are strictly increasing."""

    if not ((positions > 0.0 - eps) & (positions < 1.0 + eps)).all():
        raise ValueError("EPD positions contain values outside (0, 1).")
    augmented = np.concatenate(
        [np.zeros((positions.shape[0], 1), dtype=np.float64), positions, np.ones((positions.shape[0], 1), dtype=np.float64)],
        axis=-1,
    )
    diffs = np.diff(augmented, axis=-1)
    if (diffs <= 0).any():
        raise ValueError("EPD positions must be strictly increasing within each interval.")


def _sanitize_table_arrays(
    positions: np.ndarray,
    weights: np.ndarray,
    scale_dir: Optional[np.ndarray] = None,
    scale_time: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Ensure per-row monotonicity by sorting and applying minimal adjustments.

    Returns the sanitized arrays together with boolean masks indicating rows
    that required reordering or value adjustments.
    """

    positions = np.asarray(positions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    order = np.argsort(positions, axis=-1)
    base_indices = np.broadcast_to(np.arange(positions.shape[-1]), positions.shape)
    reordered_rows = np.any(order != base_indices, axis=-1)

    positions = np.take_along_axis(positions, order, axis=-1)
    weights = np.take_along_axis(weights, order, axis=-1)

    if scale_dir is not None:
        scale_dir = np.take_along_axis(scale_dir, order, axis=-1)
    if scale_time is not None:
        scale_time = np.take_along_axis(scale_time, order, axis=-1)

    adjusted_rows = np.zeros(positions.shape[0], dtype=bool)
    max_idx = positions.shape[-1]
    min_clip = eps

    for row_idx in range(positions.shape[0]):
        prev = 0.0
        row = positions[row_idx]
        for col_idx in range(max_idx):
            remaining = max_idx - (col_idx + 1)
            min_allowed = prev + min_clip
            max_allowed = 1.0 - min_clip * (remaining + 1)
            val = row[col_idx]
            clipped = np.clip(val, min_allowed, max_allowed)
            if clipped != val:
                adjusted_rows[row_idx] = True
            row[col_idx] = clipped
            prev = clipped

    _assert_monotonic_positions(positions)
    return positions, weights, scale_dir, scale_time, reordered_rows, adjusted_rows
