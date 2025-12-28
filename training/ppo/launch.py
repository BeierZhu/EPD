from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import config as cfg
from .cold_start import build_dirichlet_params, load_predictor_table
from .policy import EPDParamPolicy
from .ppo_trainer import PPOTrainer, PPOTrainerConfig
from .reward_hps import RewardHPS, RewardHPSConfig
from .reward_multi import RewardMetricWeights, RewardMultiMetric, RewardMultiMetricConfig, RewardMultiMetricPaths
from .rl_runner import EPDRolloutRunner, RLRunnerConfig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch PPO fine-tuning for EPD predictor tables.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/ppo/cfgs/sd15_base.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values (dot-separated keys).",
    )
    parser.add_argument("--run-name", type=str, help="Override run.run_name without editing YAML.")
    parser.add_argument("--max-steps", type=int, help="Override number of PPO steps.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and validate configuration, print resolved values, then exit.",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int, rank_offset: int = 0) -> None:
    full_seed = seed + rank_offset
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


def ensure_run_directory(run_config: cfg.RunConfig) -> None:
    base = run_config.output_root / run_config.run_id
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = run_config.output_root / f"{run_config.run_id}-{suffix:02d}"
        suffix += 1
    run_config.run_dir = candidate.resolve()
    (run_config.run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_config.run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_config.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_config.run_dir / "samples").mkdir(parents=True, exist_ok=True)


def _unwrap_policy(policy: EPDParamPolicy | DistributedDataParallel) -> EPDParamPolicy:
    if isinstance(policy, DistributedDataParallel):
        return policy.module  # type: ignore[return-value]
    return policy


def _policy_state_dict_cpu(policy: EPDParamPolicy | DistributedDataParallel) -> Dict[str, torch.Tensor]:
    state_dict = _unwrap_policy(policy).state_dict()
    cpu_state = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def save_policy_checkpoint(
    run_config: cfg.RunConfig,
    policy: EPDParamPolicy | DistributedDataParallel,
    step: int,
) -> Path:
    """
    Persist policy weights to checkpoints/policy-stepXXXXXX.pt.
    """

    if run_config.run_dir is None:
        raise RuntimeError("run_dir is not assigned; call ensure_run_directory first.")
    checkpoint_dir = run_config.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"policy-step{step:06d}.pt"
    state_dict = _policy_state_dict_cpu(policy)
    torch.save(state_dict, path)
    return path


def _distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> Tuple[bool, int, int, int]:
    if not dist.is_available():
        return False, 0, 1, 0

    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    if env_rank is None or env_world_size is None:
        return False, 0, 1, 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if _distributed_available():
        dist.destroy_process_group()


def reduce_metrics(metrics: Dict[str, float], world_size: int, device: torch.device) -> Dict[str, float]:
    if not _distributed_available() or world_size == 1:
        return {k: float(v) for k, v in metrics.items()}

    keys = sorted(metrics.keys())
    tensor = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(world_size)
    return {key: float(value) for key, value in zip(keys, tensor.tolist())}


def enrich_model_dimensions(full_config: cfg.FullConfig, dry_run: bool) -> Dict[str, int]:
    table = load_predictor_table(full_config.data.predictor_snapshot, map_location="cpu")
    if full_config.model.num_steps is not None and full_config.model.num_steps != table.num_steps:
        raise cfg.ConfigError(
            f"num_steps mismatch: config={full_config.model.num_steps} vs table={table.num_steps}"
        )
    if full_config.model.num_points is not None and full_config.model.num_points != table.num_points:
        raise cfg.ConfigError(
            f"num_points mismatch: config={full_config.model.num_points} vs table={table.num_points}"
        )
    full_config.model.num_steps = table.num_steps
    full_config.model.num_points = table.num_points

    # Prefer predictor metadata for backend/schedule/hyperparams when available.
    meta = table.metadata
    def _override(field: str, key: str) -> None:
        val_meta = meta.get(key, None) if isinstance(meta, dict) else None
        if val_meta is None:
            return
        val_cfg = getattr(full_config.model, field, None)
        if val_cfg is not None and val_cfg != val_meta:
            print(f"[Config] Overriding model.{field}={val_cfg} with predictor metadata {val_meta}")
        setattr(full_config.model, field, val_meta)

    _override("schedule_type", "schedule_type")
    _override("schedule_rho", "schedule_rho")
    _override("guidance_type", "guidance_type")
    _override("guidance_rate", "guidance_rate")
    _override("dataset_name", "dataset_name")
    _override("backend", "backend")
    _override("sigma_min", "sigma_min")
    _override("sigma_max", "sigma_max")
    _override("flowmatch_mu", "flowmatch_mu")
    _override("flowmatch_shift", "flowmatch_shift")
    _override("resolution", "resolution")

    backend_cfg_meta = meta.get("backend_config") if isinstance(meta, dict) else None
    if isinstance(backend_cfg_meta, dict) and backend_cfg_meta:
        if full_config.model.backend_options and full_config.model.backend_options != backend_cfg_meta:
            print(f"[Config] Overriding model.backend_options with predictor metadata")
        full_config.model.backend_options = backend_cfg_meta

    return {
        "num_steps": table.num_steps,
        "num_points": table.num_points,
        "schedule_type": table.metadata.get("schedule_type"),
        "schedule_rho": table.metadata.get("schedule_rho"),
    }


def save_config_snapshot(full_config: cfg.FullConfig) -> None:
    config_path = full_config.run.run_dir / "configs" / "resolved_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        handle.write(cfg.pretty_format_config(full_config))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser()

    try:
        raw = cfg.load_raw_config(config_path, overrides=args.override)
        if args.run_name:
            raw.setdefault("run", {})["run_name"] = args.run_name
        if args.max_steps is not None:
            raw.setdefault("ppo", {})["steps"] = args.max_steps
        full_config = cfg.build_config(raw)
        meta = enrich_model_dimensions(full_config, dry_run=args.dry_run)
        cfg.validate_config(full_config, check_paths=not args.dry_run)
    except cfg.ConfigError as err:
        print(f"[ConfigError] {err}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("==== Resolved Configuration ====")
        print(cfg.pretty_format_config(full_config))
        print("Derived from predictor snapshot:")
        print(json.dumps(meta, indent=2))
        return

    distributed, rank, world_size, local_rank = setup_distributed()
    is_master = rank == 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")

    try:
        if is_master:
            ensure_run_directory(full_config.run)
            save_config_snapshot(full_config)
        if distributed:
            dist.barrier()
            run_dir_holder = [str(full_config.run.run_dir) if full_config.run.run_dir else ""]
            dist.broadcast_object_list(run_dir_holder, src=0)
            if not is_master:
                full_config.run.run_dir = Path(run_dir_holder[0])

        seed_everything(full_config.run.seed, rank_offset=rank)

        if is_master:
            print(f"[Launch] run_dir={full_config.run.run_dir}")
            print(f"[Launch] World size: {world_size}")
        print(f"[Launch][Rank {rank}] Using device: {device}")
        if distributed:
            dist.barrier()

        if is_master:
            print("[Launch] Loading predictor snapshot...")
        predictor_table = load_predictor_table(full_config.data.predictor_snapshot, map_location="cpu")
        dirichlet_init = build_dirichlet_params(
            predictor_table,
            concentration=full_config.ppo.dirichlet_concentration,
        )
        policy = EPDParamPolicy(
            num_steps=predictor_table.num_steps,
            num_points=predictor_table.num_points,
            hidden_dim=128,
            num_layers=2,
            dirichlet_init=dirichlet_init,
        ).to(device)

        if distributed:
            ddp_kwargs: Dict[str, object] = {}
            if device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[device.index],
                    output_device=device.index,
                )
            policy = DistributedDataParallel(policy, **ddp_kwargs)

        if is_master:
            print("[Launch] Loading Stable Diffusion model...")
        from sample import create_model_backend  # Local import to avoid overhead on dry run

        backend_config = dict(full_config.model.backend_options)
        if full_config.model.flowmatch_mu is not None:
            backend_config.setdefault("flowmatch_mu", full_config.model.flowmatch_mu)
        if full_config.model.flowmatch_shift is not None:
            backend_config.setdefault("flowmatch_shift", full_config.model.flowmatch_shift)
        if full_config.model.resolution is not None:
            backend_config.setdefault("resolution", full_config.model.resolution)

        net, model_source = create_model_backend(
            dataset_name=full_config.model.dataset_name,
            guidance_type=full_config.model.guidance_type,
            guidance_rate=full_config.model.guidance_rate,
            backend=full_config.model.backend,
            backend_config=backend_config,
            device=device,
        )
        full_config.model.backend_options = backend_config
        if is_master:
            print("[Launch] Stable Diffusion model loaded.")
        net = net.to(device)
        net.eval()

        hps_cfg = RewardHPSConfig(
            device=device,
            batch_size=full_config.reward.batch_size,
            enable_amp=full_config.reward.enable_amp,
            weights_path=full_config.reward.weights_path,
            cache_dir=full_config.reward.cache_dir,
            hps_version=full_config.reward.hps_version,
        )
        if full_config.reward.type == "multi":
            if full_config.reward.multi is None:
                raise RuntimeError("reward.multi must be configured when reward.type='multi'.")
            multi_cfg = full_config.reward.multi
            reward = RewardMultiMetric(
                RewardMultiMetricConfig(
                    device=device,
                    batch_size=full_config.reward.batch_size,
                    image_value_range=(0.0, 1.0),
                    weights=RewardMetricWeights(
                        hps=multi_cfg.weights.hps,
                        pickscore=multi_cfg.weights.pickscore,
                        imagereward=multi_cfg.weights.imagereward,
                        clip=multi_cfg.weights.clip,
                        aesthetic=multi_cfg.weights.aesthetic,
                    ),
                    hps=hps_cfg,
                    pickscore_model_name_or_path=multi_cfg.pickscore.model,
                    pickscore_processor_name_or_path=multi_cfg.pickscore.processor,
                    paths=RewardMultiMetricPaths(
                        imagereward_checkpoint=multi_cfg.imagereward.checkpoint,
                        imagereward_med_config=multi_cfg.imagereward.med_config,
                        imagereward_cache_dir=multi_cfg.imagereward.cache_dir,
                        clip_cache_dir=multi_cfg.clip.cache_dir,
                        aesthetic_clip_path=multi_cfg.aesthetic.clip_path,
                        aesthetic_predictor_path=multi_cfg.aesthetic.predictor_path,
                    ),
                )
            )
        else:
            reward = RewardHPS(hps_cfg)

        rollout_batch_size = full_config.ppo.rollout_batch_size
        if rollout_batch_size % world_size != 0:
            raise RuntimeError(
                f"rollout_batch_size ({rollout_batch_size}) must be divisible by world_size ({world_size})."
            )
        per_rank_rollout = rollout_batch_size // world_size

        runner_config = RLRunnerConfig(
            policy=policy,
            net=net,
            num_steps=predictor_table.num_steps,
            num_points=predictor_table.num_points,
            device=device,
            guidance_type=full_config.model.guidance_type,
            guidance_rate=full_config.model.guidance_rate,
            schedule_type=full_config.model.schedule_type,
            schedule_rho=full_config.model.schedule_rho,
            dataset_name=full_config.model.dataset_name,
            precision=torch.float32,
            prompt_csv=full_config.data.prompt_csv,
            rloo_k=full_config.ppo.rloo_k,
            rng_seed=full_config.run.seed + rank,
            rank=rank,
            world_size=world_size,
            verbose=False,
            model_source=model_source,
            backend=full_config.model.backend,
            backend_config=full_config.model.backend_options,
            sigma_min=full_config.model.sigma_min,
            sigma_max=full_config.model.sigma_max,
        )
        runner = EPDRolloutRunner(runner_config)

        trainer_config = PPOTrainerConfig(
            device=device,
            rollout_batch_size=per_rank_rollout,
            rloo_k=full_config.ppo.rloo_k,
            ppo_epochs=full_config.ppo.ppo_epochs,
            minibatch_size=full_config.ppo.minibatch_size,
            learning_rate=full_config.ppo.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            clip_range=full_config.ppo.clip_range,
            kl_coef=full_config.ppo.kl_coef,
            entropy_coef=full_config.ppo.entropy_coef,
            normalize_advantages=True,
            max_grad_norm=full_config.ppo.max_grad_norm,
            decode_rgb=full_config.ppo.decode_rgb,
            image_value_range=(0.0, 1.0),
        )
        trainer = PPOTrainer(policy, runner, reward, trainer_config)

        metrics_path = full_config.run.run_dir / "logs" / "metrics.jsonl"
        if is_master:
            print(f"[Launch] Writing metrics to {metrics_path}")

        start_time = time.time()
        step = 0
        metrics_file_ctx = metrics_path.open("w", encoding="utf-8") if is_master else nullcontext()
        with metrics_file_ctx as metrics_file:
            while step < full_config.ppo.steps:
                step += 1
                step_metrics = trainer.train_step()
                reduced_metrics = reduce_metrics(step_metrics, world_size, device)
                reduced_metrics["step"] = step
                if is_master and metrics_file is not None:
                    reduced_metrics["elapsed_sec"] = time.time() - start_time
                    print(json.dumps(reduced_metrics), file=metrics_file, flush=True)
                    if step % full_config.logging.log_interval == 0:
                        summary = ", ".join(
                            f"{k}={reduced_metrics[k]:.4f}"
                            for k in ("mixed_reward_mean", "kl", "policy_loss", "mixed_reward_std")
                            if k in reduced_metrics
                        )
                        print(f"[Step {step}] {summary}")
                if (
                    full_config.logging.save_interval > 0
                    and step % full_config.logging.save_interval == 0
                    and is_master
                ) or (step == full_config.ppo.steps and is_master):
                    ckpt_path = save_policy_checkpoint(full_config.run, trainer.policy, step)
                    print(f"[Step {step}] Saved policy checkpoint to {ckpt_path}")

        if distributed:
            dist.barrier()

        if is_master:
            print(
                f"[Launch] Finished {full_config.ppo.steps} PPO steps. Results saved to {full_config.run.run_dir}"
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":  # pragma: no cover
    main()


'''

Check whether the config matches the cold-start distillation table parameters
python -m training.ppo.launch --dry-run

'''
