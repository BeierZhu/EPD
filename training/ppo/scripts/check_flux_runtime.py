#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional

import click

from training.ppo.pipeline_utils import FLUX_RUNTIME_VERSION_SPECS, resolve_predictor_path


def _read_predictor_model_ref(predictor: str) -> Optional[str]:
    import pickle

    predictor_path = resolve_predictor_path(predictor)
    with predictor_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    model = snapshot.get("model")
    backend_cfg = getattr(model, "backend_config", None)
    if isinstance(backend_cfg, dict):
        ref = backend_cfg.get("model_name_or_path") or backend_cfg.get("model_id")
        if ref:
            return str(ref)
    return None


def _prefix_ok(version: str, prefix: str) -> bool:
    return version.startswith(prefix)


def _collect_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "python": platform.python_version(),
    }
    for pkg in ("torch", "torchvision", "diffusers", "transformers", "huggingface_hub", "accelerate", "safetensors"):
        versions[pkg] = metadata.version(pkg)
    return versions


def _probe_flux_support() -> Dict[str, object]:
    try:
        dist = metadata.distribution("diffusers")
    except metadata.PackageNotFoundError as exc:
        return {
            "flux_pipeline_check_mode": "installed_package_files",
            "diffusers_spec_found": False,
            "flux_pipeline_module_found": False,
            "flux_pipeline_importable": False,
            "flux_pipeline_path": None,
            "flux_pipeline_import_error": f"PackageNotFoundError: {exc}",
        }

    candidates = (
        Path("diffusers/pipelines/flux/pipeline_flux.py"),
        Path("diffusers/pipelines/flux/__init__.py"),
    )
    located_path = None
    for relative in candidates:
        try:
            candidate = Path(dist.locate_file(relative))
        except Exception as exc:  # pragma: no cover - runtime-only guard
            return {
                "flux_pipeline_check_mode": "installed_package_files",
                "diffusers_spec_found": True,
                "flux_pipeline_module_found": False,
                "flux_pipeline_importable": False,
                "flux_pipeline_path": None,
                "flux_pipeline_import_error": f"{type(exc).__name__}: {exc}",
            }
        if candidate.is_file():
            located_path = str(candidate)
            break

    return {
        "flux_pipeline_check_mode": "installed_package_files",
        "diffusers_spec_found": True,
        "flux_pipeline_module_found": located_path is not None,
        "flux_pipeline_importable": located_path is not None,
        "flux_pipeline_path": located_path,
        "flux_pipeline_import_error": None
        if located_path is not None
        else "Could not locate diffusers/pipelines/flux/pipeline_flux.py inside the installed diffusers package.",
    }


@click.command()
@click.option("--model-id", type=str, default=None, help="Local FLUX snapshot path or HF repo id.")
@click.option("--predictor", type=click.Path(exists=True, dir_okay=True, file_okay=True), default=None, help="Optional predictor used to resolve the FLUX model reference.")
@click.option("--allow-remote", is_flag=True, default=False, help="Allow non-local HF repo ids instead of requiring a local snapshot.")
def main(model_id: Optional[str], predictor: Optional[str], allow_remote: bool) -> None:
    resolved_model = model_id
    if resolved_model is None and predictor is not None:
        resolved_model = _read_predictor_model_ref(predictor)
    if resolved_model is None:
        raise click.ClickException("Provide --model-id or --predictor.")

    versions = _collect_versions()
    mismatches = []
    for key, prefix in FLUX_RUNTIME_VERSION_SPECS.items():
        version_key = key.replace("_prefix", "")
        current = versions.get(version_key)
        if current is None:
            continue
        if not _prefix_ok(current, prefix):
            mismatches.append(f"{version_key}={current} does not match expected prefix {prefix}")

    flux_probe = _probe_flux_support()

    local_model = Path(str(resolved_model)).expanduser()
    local_exists = local_model.exists()
    if not local_exists and not allow_remote:
        raise click.ClickException(
            f"FLUX model is not local: {resolved_model}. Pass --allow-remote only after authenticating to Hugging Face."
        )

    report = {
        "model_id": str(resolved_model),
        "model_is_local": bool(local_exists),
        "allow_remote": bool(allow_remote),
        "versions": versions,
        "version_mismatches": mismatches,
        **flux_probe,
    }

    if mismatches:
        raise click.ClickException(json.dumps(report, indent=2))
    if not bool(flux_probe["flux_pipeline_importable"]):
        raise click.ClickException(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
