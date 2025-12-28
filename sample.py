import os
import re
import csv
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import json
from torch import autocast
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key
from training.loss import get_solver_fn
import torchvision
import torchvision.transforms as transforms
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion) 
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)

def load_ldm_model(config, ckpt, verbose=False):
    from models.ldm.util import instantiate_from_config
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

#----------------------------------------------------------------------------

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    """Load the Stable Diffusion 1.5 backbone used across RLEPD."""

    if dataset_name not in [None, "ms_coco"]:
        raise ValueError("This trimmed RLEPD build only supports dataset_name='ms_coco'.")
    if guidance_type not in [None, "cfg"]:
        raise ValueError("Stable Diffusion fine-tuning requires classifier-free guidance (cfg).")
    if guidance_rate is None:
        raise ValueError("guidance_rate must be provided for cfg sampling.")

    model_path, _ = check_file_by_key("ms_coco")
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')

    from omegaconf import OmegaConf
    from models.networks_edm import CFGPrecond

    config = OmegaConf.load("./models/ldm/configs/stable-diffusion/v1-inference.yaml")
    net = load_ldm_model(config, model_path)
    net = CFGPrecond(
        net,
        img_resolution=64,
        img_channels=4,
        guidance_rate=guidance_rate,
        guidance_type="classifier-free",
        label_dim=True,
    ).to(device)
    net.eval()
    net.backend = "ldm"
    net.backend_config = {}
    return net, "ldm"


def _resolve_torch_dtype(value, device):
    if value is None or value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"float32", "fp32"}:
            return torch.float32
        if lowered in {"float16", "fp16"}:
            return torch.float16
        if lowered in {"bfloat16", "bf16"}:
            return torch.bfloat16
    raise ValueError(f"Unsupported torch_dtype: {value}")


def create_model_sd3(
    dataset_name=None,
    guidance_type=None,
    guidance_rate=None,
    device=None,
    backend_config=None,
):
    if guidance_rate is None:
        raise ValueError("guidance_rate must be provided for SD3 sampling.")
    cfg = dict(backend_config) if isinstance(backend_config, dict) else {}
    model_id = cfg.get("model_name_or_path") or cfg.get("model_id") or "stabilityai/stable-diffusion-3-medium-diffusers"
    torch_dtype = _resolve_torch_dtype(cfg.get("torch_dtype", "auto"), device if isinstance(device, torch.device) else torch.device(device))
    enable_offload = bool(cfg.get("enable_model_cpu_offload", False))
    revision = cfg.get("revision")
    variant = cfg.get("variant")
    use_safetensors = cfg.get("use_safetensors", True)
    token = cfg.get("token")
    resolution = int(cfg.get("resolution", 1024) or 1024)
    if resolution not in (512, 1024):
        raise ValueError(f"SD3 resolution must be 512 or 1024, got {resolution}")
    pipeline_kwargs = cfg.get("pipeline_kwargs") if isinstance(cfg.get("pipeline_kwargs"), dict) else None
    flowmatch_mu = cfg.get("flowmatch_mu")

    from models.backends import SD3DiffusersBackend

    backend = SD3DiffusersBackend(
        model_name_or_path=model_id,
        device=device,
        torch_dtype=torch_dtype,
        guidance_scale=guidance_rate,
        enable_model_cpu_offload=enable_offload,
        revision=revision,
        variant=variant,
        use_safetensors=use_safetensors,
        token=token,
        pipeline_kwargs=pipeline_kwargs,
        flowmatch_mu=flowmatch_mu,
        resolution=resolution,
    )
    backend.backend_config = dict(cfg)
    if "flowmatch_mu" not in backend.backend_config and backend.default_flowmatch_mu is not None:
        backend.backend_config["flowmatch_mu"] = backend.default_flowmatch_mu
    backend.backend_config.setdefault("flowmatch_shift", backend.flow_shift)
    backend.backend_config.setdefault("sigma_min", backend.sigma_min)
    backend.backend_config.setdefault("sigma_max", backend.sigma_max)
    backend.backend_config.setdefault("resolution", resolution)
    backend.backend_config.setdefault("latent_resolution", getattr(backend, "latent_resolution", None))
    return backend, "sd3"


def _prepare_sd3_condition(net, prompts, guidance_rate, backend_config):
    prompts_list = list(prompts)
    if guidance_rate == 1.0:
        negative_prompt = None
    else:
        base_negative = backend_config.get("negative_prompt", "")
        if isinstance(base_negative, list):
            if len(base_negative) != len(prompts_list):
                raise ValueError("Length of negative_prompt list must match batch size.")
            negative_prompt = base_negative
        else:
            negative_prompt = [str(base_negative)] * len(prompts_list)
    return net.prepare_condition(
        prompt=prompts_list,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_rate,
    )


def create_model_backend(
    dataset_name=None,
    guidance_type=None,
    guidance_rate=None,
    backend="ldm",
    backend_config=None,
    device=None,
):
    backend = (backend or "ldm").lower()
    if backend == "sd3":
        return create_model_sd3(
            dataset_name=dataset_name,
            guidance_type=guidance_type,
            guidance_rate=guidance_rate,
            device=device,
            backend_config=backend_config,
        )
    return create_model(dataset_name, guidance_type, guidance_rate, device)

#----------------------------------------------------------------------------


@click.command()
# General options
@click.option('--predictor_path',          help='Path to trained instructor', metavar='DIR',                   type=str, required=True)
@click.option('--model_path',              help='Network filepath', metavar='PATH|URL',                             type=str)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)
@click.option('--prompt-file',             help='Path to text/CSV file with prompts (one per line)', metavar='PATH', type=click.Path(exists=True, dir_okay=False))
@click.option('--use_fp16',                help='Whether to use mixed precision', metavar='BOOL',                   type=bool, default=False)
@click.option('--backend',                 help='Override backend type (defaults to predictor metadata).',          type=str)
@click.option('--backend-config',          help='JSON string overriding backend-specific options.',                 type=str)

# Options for sampling
@click.option('--return_inters',           help='Whether to save intermediate outputs', metavar='BOOL',             type=bool, default=False)

# Options for saving
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str)
@click.option('--grid',                    help='Whether to make grid',                                             type=bool, default=False)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         type=bool, default=True, is_flag=True)

#----------------------------------------------------------------------------

def main(
    predictor_path,
    max_batch_size,
    seeds,
    grid,
    outdir,
    subdirs,
    prompt_file,
    backend,
    backend_config,
    device=torch.device('cuda'),
    **solver_kwargs,
):

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Load models.
    if dist.get_rank() != 0:
        torch.distributed.barrier()     # rank 0 goes first

    # Load predictor
    if not predictor_path.endswith('pkl'):      # load by experiment number
        # find the directory with trained predictor
        predictor_path_str = '0' * (5 - len(predictor_path)) + predictor_path
        for file_name in os.listdir("./exps"):
            if file_name.split('-')[0] == predictor_path_str:
                file_list = [f for f in os.listdir(os.path.join('./exps', file_name)) if f.endswith("pkl")]
                max_index = -1
                max_file = None
                for ckpt_name in file_list:
                    file_index = int(ckpt_name.split("-")[-1].split(".")[0])
                    if file_index > max_index:
                        max_index = file_index
                        max_file = ckpt_name
                predictor_path = os.path.join('./exps', file_name, max_file)
                break
    dist.print0(f'Loading predictor from "{predictor_path}"...')
    with dnnlib.util.open_url(predictor_path, verbose=(dist.get_rank() == 0)) as f:
        predictor = pickle.load(f)['model'].to(device)
    
    # Update settings
    prompt = solver_kwargs['prompt']
    solver_kwargs = {key: value for key, value in solver_kwargs.items() if value is not None}
    solver_kwargs['predictor'] = predictor

    if predictor.sampler_stu == 'epd':
        solver_kwargs['solver'] = solver = 'epd_parallel'
    else:
        solver_kwargs['solver'] = solver = predictor.sampler_stu
    solver_kwargs['num_steps'] = predictor.num_steps
    solver_kwargs['guidance_type'] = predictor.guidance_type
    solver_kwargs['guidance_rate'] = predictor.guidance_rate
    solver_kwargs['afs'] = predictor.afs
    solver_kwargs['denoise_to_zero'] = False
    solver_kwargs['max_order'] = predictor.max_order
    solver_kwargs['predict_x0'] = predictor.predict_x0
    solver_kwargs['lower_order_final'] = predictor.lower_order_final
    solver_kwargs['schedule_type'] = predictor.schedule_type
    solver_kwargs['schedule_rho'] = predictor.schedule_rho
    solver_kwargs['prompt'] = prompt
    solver_kwargs['dataset_name'] = dataset_name = predictor.dataset_name

    user_backend = backend.strip() if backend is not None else None
    backend_config_override = {}
    if backend_config:
        try:
            parsed_override = json.loads(backend_config)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON for --backend-config: {exc}") from exc
        if not isinstance(parsed_override, dict):
            raise click.ClickException("--backend-config must decode to a JSON object.")
        backend_config_override = parsed_override

    predictor_backend = getattr(predictor, "backend", "ldm")
    predictor_backend_config = getattr(predictor, "backend_config", {})
    if isinstance(predictor_backend_config, dict):
        base_backend_config = dict(predictor_backend_config)
    else:
        try:
            base_backend_config = dict(predictor_backend_config)
        except Exception:
            base_backend_config = {}

    resolved_backend = user_backend or predictor_backend
    merged_backend_config = base_backend_config
    merged_backend_config.update(backend_config_override)
    solver_kwargs['backend'] = resolved_backend
    solver_kwargs['backend_config'] = merged_backend_config


    # Load pre-trained diffusion models.
    net, solver_kwargs['model_source'] = create_model_backend(
        dataset_name=dataset_name,
        guidance_type=solver_kwargs['guidance_type'],
        guidance_rate=solver_kwargs['guidance_rate'],
        backend=resolved_backend,
        backend_config=merged_backend_config,
        device=device,
    )
    # TODO: support mixed precision 
    # net.use_fp16 = solver_kwargs['use_fp16']

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    # Update settings
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    num_points = predictor.num_points
    nfe = 2 * (solver_kwargs['num_steps'] - 1) - 1 if solver_kwargs["afs"] else 2 * (solver_kwargs['num_steps'] - 1)
    nfe = 2 * nfe if dataset_name in ['ms_coco'] else nfe   # should double NFE due to the classifier-free-guidance
    solver_kwargs['nfe'] = nfe

    # Load the prompts
    sample_captions = None
    if prompt_file and solver_kwargs['prompt'] is None:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
        if not lines:
            raise RuntimeError(f"No prompts found in '{prompt_file}'.")
        sample_captions = lines
    elif dataset_name in ['ms_coco'] and solver_kwargs['prompt'] is None:
        # Loading MS-COCO captions for FID-30k evaluaion
        # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
        prompt_path, _ = check_file_by_key('prompts')
        sample_captions = []
        with open(prompt_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)
        if not sample_captions:
            raise RuntimeError(f"No prompts found in '{prompt_path}'.")

    # Construct solver
    sampler_fn = get_solver_fn(solver)
    # Print solver settings.
    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        if value is None:
            continue
        elif key == 'predictor':
            continue
        elif key == 'max_order' and solver in ['euler', 'dpm', 'dpm2']:
            continue
        elif key in ['predict_x0', 'lower_order_final'] and solver not in ['dpmpp']:
            continue
        elif key in ['prompt'] and dataset_name not in ['ms_coco']:
            continue
        dist.print0(f"\t{key}: {value}")

    # Loop over batches.
    if outdir is None:
        if grid:
            outdir = os.path.join(f"./samples/grids/{dataset_name}", f"{solver}_nfe{nfe}_npoints_{num_points}")
        else:
            outdir = os.path.join(f"./samples/{dataset_name}", f"{solver}_nfe{nfe}_npoints_{num_points}")
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')


    for batch_id, batch_seeds in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)


        class_labels = c = uc = None
        if solver_kwargs['prompt'] is None:
            if sample_captions is None:
                prompts = ["" for _ in range(batch_size)]
            else:
                start = int(batch_seeds[0])
                end = int(batch_seeds[-1])
                if end >= len(sample_captions):
                    raise RuntimeError(
                        f"Batch seed index {end} exceeds available prompts ({len(sample_captions)})."
                    )
                prompts = sample_captions[start:end + 1]
                if len(prompts) != batch_size:
                    raise RuntimeError(
                        f"Prompt slice length {len(prompts)} does not match batch size {batch_size}."
                    )
        else:
            prompts = [solver_kwargs['prompt'] for _ in range(batch_size)]
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        if net.label_dim:
            if solver_kwargs['model_source'] == 'adm':                                              # ADM models
                class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)
            elif solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                if solver_kwargs['guidance_rate'] != 1.0:
                    uc = net.model.get_learned_conditioning(batch_size * [""])
                c = net.model.get_learned_conditioning(prompts)
            else:
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]

        condition_payload = None
        if solver_kwargs['model_source'] == 'sd3':
            condition_payload = _prepare_sd3_condition(
                net,
                prompts,
                solver_kwargs['guidance_rate'],
                merged_backend_config,
            )

        # Generate images.
        with torch.no_grad():
            if solver_kwargs['model_source'] == 'ldm':
                with autocast("cuda"):
                    with net.model.ema_scope():
                        call_kwargs = dict(solver_kwargs)
                        call_kwargs.update(condition=c, unconditional_condition=uc)
                        if batch_id == 0:
                            call_kwargs['verbose'] = True
                        images, _ = sampler_fn(net, latents, **call_kwargs)
                        images = net.model.decode_first_stage(images)
            elif solver_kwargs['model_source'] == 'sd3':
                call_kwargs = dict(solver_kwargs)
                call_kwargs['condition'] = condition_payload
                if batch_id == 0:
                    call_kwargs['verbose'] = True
                images, _ = sampler_fn(net, latents, **call_kwargs)
                images = net.vae_decode(images)
            else:
                call_kwargs = dict(solver_kwargs)
                call_kwargs['class_labels'] = class_labels
                call_kwargs['nums_steps'] = solver_kwargs['num_steps']
                if batch_id == 0:
                    call_kwargs['verbose'] = True
                images, _ = sampler_fn(net, latents, **call_kwargs)
        
        # Save images.
        if grid:
            images = torch.clamp(images / 2 + 0.5, 0, 1)
            os.makedirs(outdir, exist_ok=True)
            nrows = int(images.shape[0] ** 0.5)
            image_grid = make_grid(images, nrows, padding=0)
            save_image(image_grid, os.path.join(outdir, "grid.png"))
        else:
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    torch.distributed.barrier()
    dist.print0('Done.')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------


'''

MASTER_ADDR=127.0.0.1 MASTER_PORT=29610 python sample.py \
    --predictor_path exps/00036-ms_coco-10-36-epd-dpm-1-discrete/network-snapshot-000005.pkl \
    --seeds 0-3 --batch 2 --seeds "0-10" \
    --outdir ./samples/origin

MASTER_ADDR=127.0.0.1 MASTER_PORT=29610 python sample.py \
    --predictor_path exps/20251030-215325-sd15_rl_base/export/network-snapshot-export-step000040.pkl \
    --seeds 0-3 --batch 2 --seeds "0-10" \
    --outdir ./samples/rl

'''
