import torch
import numpy as np
from solver_utils import *
from torch.nn.parallel import DataParallel
from concurrent.futures import ThreadPoolExecutor
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# Initialize the hook function to get the U-Net bottleneck outputs

def init_hook(net, class_labels=None):
    unet_enc_out = []
    def hook_fn(module, input, output):
        unet_enc_out.append(output.detach())
    backend_type = getattr(net, "backend", None)
    if backend_type == "sd3":
        class _DummyHook:
            def remove(self):
                return None
        return unet_enc_out, _DummyHook()
    if hasattr(net, 'guidance_type'):                                       # models from LDM and Stable Diffusion
        hook = net.model.model.diffusion_model.middle_block.register_forward_hook(hook_fn)
    elif net.img_resolution == 256:                                         # models from CM and ADM with resolution of 256
        hook = net.model.middle_block.register_forward_hook(hook_fn)
    else:                                                                   # models from EDM
        module_name = '8x8_block2' if class_labels is not None else '8x8_block3'
        hook = net.model.enc[module_name].register_forward_hook(hook_fn)
    return unet_enc_out, hook


#----------------------------------------------------------------------------
def get_epd_prediction(predictor, step_idx, net, unet_enc_out, use_afs, batch_size):
    output = predictor(batch_size, step_idx)    
    output_list = [*output]
    
    try:
        num_points = predictor.num_points
    except:
        num_points = predictor.module.num_points

    if len(output_list) == 4:
        r, scale_dir, scale_time, weight = output_list
        r = r.reshape(-1, num_points, 1, 1)
        weight = weight.reshape(-1, num_points, 1, 1)
        scale_dir = scale_dir.reshape(-1, num_points, 1, 1)
        scale_time = scale_time.reshape(-1, num_points, 1, 1)
    elif len(output_list) == 3:
        try:
            use_scale_time = predictor.module.scale_time
        except:
            use_scale_time = predictor.scale_time

        if use_scale_time:
            r, scale_time, weight = output_list
            r = r.reshape(-1, num_points, 1, 1)
            weight = weight.reshape(-1, num_points, 1, 1)
            scale_time = scale_time.reshape(-1, num_points, 1, 1)
            scale_dir = torch.ones_like(scale_time)
        else:
            r, scale_dir, weight = output_list
            r = r.reshape(-1, num_points, 1, 1)
            weight = weight.reshape(-1, num_points, 1, 1)
            scale_dir = scale_dir.reshape(-1, num_points, 1, 1)
            scale_time = torch.ones_like(scale_dir)
    else:
        r, weight = output
        r = r.reshape(-1, num_points, 1, 1)
        weight = weight.reshape(-1, num_points, 1, 1)
        scale_dir = torch.ones_like(r)
        scale_time = torch.ones_like(r)
    
    try:
        num_steps = predictor.module.num_steps
        use_fcn = predictor.module.fcn
    except:
        num_steps = predictor.num_steps
        use_fcn = predictor.fcn
        
    if step_idx == num_steps - 2 and use_fcn:
        scale_dir.fill_(1.00)

    return r, scale_dir, scale_time, weight


#----------------------------------------------------------------------------
# Get the denoised output from the pre-trained diffusion models.

def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    backend_type = getattr(net, "backend", None)
    if backend_type == "sd3":
        return net(x, t, condition=condition)
    if hasattr(net, 'guidance_type'):     # models from LDM and Stable Diffusion
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

#----------------------------------------------------------------------------
def epd_sampler(
    net, 
    latents, 
    class_labels=None,
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False,
    predictor=None, 
    step_idx=None, 
    train=False, 
    verbose=False,
    **kwargs
):
    """
    """
    assert predictor is not None

    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    # Main sampling loop.
    backend_type = getattr(net, "backend", None)
    # SD3 flowmatch timeline starts at sigma=1, so no extra scaling; other backends keep prior behavior.
    if backend_type == "sd3":
        x_next = latents
    else:
        x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]

    x_list = []
    x_list.append(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)
        
        # Euler step.
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)

        try:
            num_points = predictor.num_points
        except:
            num_points = predictor.module.num_points
        
        if step_idx is not None:
            r_s, scale_dir_s, scale_time_s, weight_s = get_epd_prediction(predictor,
                                                                step_idx, 
                                                                net, unet_enc_out, 
                                                                use_afs, batch_size=latents.shape[0])
        else:
            r_s, scale_dir_s, scale_time_s, weight_s = get_epd_prediction(predictor, 
                                                                i, 
                                                                net, unet_enc_out, 
                                                                use_afs, batch_size=latents.shape[0])
        x_next = x_cur

        if verbose:
            print_str = f'step {i}: |'
            for j in range(num_points):
                print_str += f'r{j}: {r_s[0,j,0,0]:.5f} '
                print_str += f'st{j}: {scale_time_s[0,j,0,0]:.5f} '
                print_str += f'sd{j}: {scale_dir_s[0,j,0,0]:.5f} '
                print_str += f'w{j}: {weight_s[0,j,:,:].mean().item():.5f} |'
            dist.print0(print_str)
        
        # Switch update rule for SD3 flowmatch: always use mid-point velocities (no K=1 shortcut).
        backend_type = getattr(net, "backend", None)
        if backend_type == "sd3":
            v_combined = torch.zeros_like(x_cur)
            for j in range(num_points):
                r = r_s[:, j : j + 1, :, :]
                scale_time = scale_time_s[:, j : j + 1, :, :]
                scale_dir = scale_dir_s[:, j : j + 1, :, :]
                w = weight_s[:, j : j + 1, :, :]
                t_mid = (t_next ** r) * (t_cur ** (1 - r))
                x_mid = x_cur + (t_mid - t_cur) * d_cur

                denoised_t_mid = get_denoised(
                    net,
                    x_mid,
                    scale_time * t_mid,
                    class_labels=class_labels,
                    condition=condition,
                    unconditional_condition=unconditional_condition,
                )

                d_mid = (x_mid - denoised_t_mid) / t_mid  # velocity v at mid-point
                v_combined = v_combined + w * scale_dir * d_mid
            # Euler step using combined velocity and sigma delta
            x_next = x_cur + (t_next - t_cur) * v_combined
        else:
            # Original EPD update for non-SD3 backends
            for j in range(num_points):
                r = r_s[:,j:j+1,:,:]
                scale_time = scale_time_s[:,j:j+1,:,:]
                scale_dir = scale_dir_s[:,j:j+1,:,:]
                w = weight_s[:,j:j+1,:,:]
                t_mid = (t_next ** r) * (t_cur ** (1 - r))
                x_mid = x_cur + (t_mid - t_cur) * d_cur

                denoised_t_mid = get_denoised(net, x_mid, scale_time*t_mid, 
                                              class_labels=class_labels, 
                                              condition=condition, 
                                              unconditional_condition=unconditional_condition)

                d_mid = (x_mid - denoised_t_mid) / t_mid
                x_next = x_next + w * scale_dir * (t_next - t_cur) * d_mid


        x_list.append(x_next)
      
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r_s, scale_dir_s, scale_time_s, weight_s
    return x_next, x_list

#----------------------------------------------------------------------------
def epd_parallel_sampler(
    net, 
    latents, 
    class_labels=None,
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False,
    predictor=None, 
    step_idx=None, 
    train=False, 
    verbose=False,
    **kwargs
):
    """
    """
    assert predictor is not None

    # Time step discretization.
    x_list = []
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    backend_type = getattr(net, "backend", None)
    # Main sampling loop.
    x_next = latents if backend_type == "sd3" else latents * t_steps[0]
    x_list.append(x_next)
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)
        
        # Euler step.
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)

        try:
            num_points = predictor.num_points
        except:
            num_points = predictor.module.num_points
        
        if step_idx is not None:
            r_s, scale_dir_s, scale_time_s, weight_s = get_epd_prediction(predictor,
                                                                step_idx, 
                                                                net, unet_enc_out, 
                                                                use_afs, batch_size=latents.shape[0])
        else:
            r_s, scale_dir_s, scale_time_s, weight_s = get_epd_prediction(predictor, 
                                                                i, 
                                                                net, unet_enc_out, 
                                                                use_afs, batch_size=latents.shape[0])
        x_next = x_cur

        if verbose:
            print_str = f'step {i}: |'
            for j in range(num_points):
                print_str += f'r{j}: {r_s[0,j,0,0]:.5f} '
                print_str += f'st{j}: {scale_time_s[0,j,0,0]:.5f} '
                print_str += f'sd{j}: {scale_dir_s[0,j,0,0]:.5f} '
                print_str += f'w{j}: {weight_s[0,j,:,:].mean().item():.5f} |'
            dist.print0(print_str)

        # SD3 uses flowmatch schedule starting at sigma=1: always use mid-point velocities (no K=1 shortcut).
        t_mid = (t_next ** r_s) * (t_cur ** (1 - r_s))
        x_mid = x_cur.unsqueeze(1) + (t_mid - t_cur).unsqueeze(-1) * d_cur.unsqueeze(1)

        B, num_points, C, H, W = x_mid.shape
        x_mid_flat = x_mid.view(B * num_points, C, H, W)

        t_param = (scale_time_s * t_mid).view(B * num_points, 1, 1, 1)

        def _repeat_sd3_condition(cond_obj, times):
            if cond_obj is None:
                return None
            from models.backends.sd3_diffusers_backend import SD3Conditioning

            return SD3Conditioning(
                prompt_embeds=cond_obj.prompt_embeds.repeat_interleave(times, dim=0),
                pooled_prompt_embeds=cond_obj.pooled_prompt_embeds.repeat_interleave(times, dim=0),
                negative_prompt_embeds=cond_obj.negative_prompt_embeds.repeat_interleave(times, dim=0)
                if cond_obj.negative_prompt_embeds is not None
                else None,
                negative_pooled_prompt_embeds=cond_obj.negative_pooled_prompt_embeds.repeat_interleave(times, dim=0)
                if cond_obj.negative_pooled_prompt_embeds is not None
                else None,
                guidance_scale=cond_obj.guidance_scale,
                num_images_per_prompt=cond_obj.num_images_per_prompt,
            )

        if backend_type == "sd3":
            cond_flat = _repeat_sd3_condition(condition, num_points)
            uc_flat = _repeat_sd3_condition(unconditional_condition, num_points)
        else:
            cond_flat = None
            uc_flat = None
            if condition is not None:
                cond_flat = condition.repeat_interleave(num_points, dim=0)
            if unconditional_condition is not None:
                uc_flat = unconditional_condition.repeat_interleave(num_points, dim=0)

        denoised_t_mid_flat = get_denoised(
            net,
            x_mid_flat,
            t_param,
            class_labels=class_labels,
            condition=cond_flat,
            unconditional_condition=uc_flat,
        )
        denoised_t_mid = denoised_t_mid_flat.view(B, num_points, C, H, W)
        d_mid = (x_mid - denoised_t_mid) / t_mid.unsqueeze(-1)

        if backend_type == "sd3":
            # Combine multi-point velocities then take a single Euler step on sigma grid.
            weights = (weight_s * scale_dir_s).unsqueeze(-1)
            v_combined = (weights * d_mid).sum(dim=1)
            x_next = x_cur + (t_next - t_cur) * v_combined
        else:
            factor = (weight_s * scale_dir_s).unsqueeze(-1) * (t_next - t_cur).unsqueeze(1)
            update = factor * d_mid  
            update_sum = update.sum(dim=1)       
            x_next = x_cur + update_sum

        x_list.append(x_next)
      
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r_s, scale_dir_s, scale_time_s, weight_s
    return x_next, x_list

#----------------------------------------------------------------------------
def dpm_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='time_uniform', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    inner_steps=2,  # New parameter for the number of inner steps
    r=0.5,
    **kwargs
):
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    backend_type = getattr(net, "backend", None)
    # Main sampling loop.
    x_next = latents if backend_type == "sd3" else latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]

    x_list = []
    x_list.append(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Compute the inner step size
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device, schedule_type='polynomial', schedule_rho=7)
        for i, (t_c, t_n) in enumerate(zip(t_s[:-1],t_s[1:])):
            # Euler step.
            use_afs = (afs and i == 0)
            if use_afs:
                d_cur = x_cur / ((1 + t_c**2).sqrt())
            else:
                denoised = get_denoised(net, x_cur, t_c, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
                d_cur = (x_cur - denoised) / t_c
            t_mid = (t_n ** r) * (t_c ** (1 - r))
            x_next = x_cur + (t_mid - t_c) * d_cur

            # Apply 2nd order correction.
            denoised = get_denoised(net, x_next, t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            
            d_mid = (x_next - denoised) / t_mid
            x_cur = x_cur + (t_n - t_c) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)
        x_next = x_cur
        x_list.append(x_next)

        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


#----------------------------------------------------------------------------
def dpm_sampler_2(
    net,
    latents,
    class_labels=None,
    condition=None,
    unconditional_condition=None,
    num_steps=None,
    sigma_min=0.002,
    sigma_max=80,
    schedule_type='time_uniform',
    schedule_rho=7,
    afs=False,
    denoise_to_zero=False,
    return_inters=False,
    inner_steps=None,
    r=0.5,
    t_steps=None,
    **kwargs,
):
    """
    DPM-Solver-2 implementation aligned with project sampler interface.
    """
    del inner_steps  # Unused but kept for signature compatibility.
    backend_type = getattr(net, "backend", None)

    if t_steps is None:
        if num_steps is None:
            raise ValueError("num_steps must be specified when t_steps is not provided.")
        t_steps = get_schedule(
            num_steps,
            sigma_min,
            sigma_max,
            device=latents.device,
            schedule_type=schedule_type,
            schedule_rho=schedule_rho,
            net=net,
        )
    else:
        t_steps = torch.as_tensor(t_steps, device=latents.device, dtype=latents.dtype)

    x_next = latents if backend_type == "sd3" else latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]

    x_list = [x_next]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        use_afs = afs and i == 0
        if use_afs:
            d_cur = x_cur / ((1 + t_cur ** 2).sqrt())
        else:
            denoised = get_denoised(
                net,
                x_cur,
                t_cur,
                class_labels=class_labels,
                condition=condition,
                unconditional_condition=unconditional_condition,
            )
            d_cur = (x_cur - denoised) / t_cur

        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_mid = x_cur + (t_mid - t_cur) * d_cur

        denoised_mid = get_denoised(
            net,
            x_mid,
            t_mid,
            class_labels=class_labels,
            condition=condition,
            unconditional_condition=unconditional_condition,
        )
        d_prime = (x_mid - denoised_mid) / t_mid

        x_next = x_cur + (t_next - t_cur) * ((1 / (2 * r)) * d_prime + (1 - 1 / (2 * r)) * d_cur)

        x_list.append(x_next)

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(
            net,
            x_next,
            t_next,
            class_labels=class_labels,
            condition=condition,
            unconditional_condition=unconditional_condition,
        )
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


#----------------------------------------------------------------------------

def heun_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='time_uniform', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    inner_steps=3,  # Kept for signature compatibility; unused in standard Heun.
    r=0.5,          # Kept for signature compatibility; unused in standard Heun.
    **kwargs
):
    del inner_steps, r
    # Time step discretization.
    t_steps = get_schedule(
        num_steps,
        sigma_min,
        sigma_max,
        device=latents.device,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        net=net,
    )
    backend_type = getattr(net, "backend", None)
    # Main sampling loop.
    x_next = latents if backend_type == "sd3" else latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]

    x_list = [x_next]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        use_afs = afs and i == 0
        if use_afs:
            d_cur = x_cur / ((1 + t_cur ** 2).sqrt())
        else:
            denoised = get_denoised(
                net,
                x_cur,
                t_cur,
                class_labels=class_labels,
                condition=condition,
                unconditional_condition=unconditional_condition,
            )
            d_cur = (x_cur - denoised) / t_cur

        # Heun (explicit trapezoidal) step: one Euler proposal + one correction.
        x_euler = x_cur + (t_next - t_cur) * d_cur
        denoised_next = get_denoised(
            net,
            x_euler,
            t_next,
            class_labels=class_labels,
            condition=condition,
            unconditional_condition=unconditional_condition,
        )
        d_prime = (x_euler - denoised_next) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        x_list.append(x_next)

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


#----------------------------------------------------------------------------
def ipndm_sampler(
    net,
    latents,
    class_labels=None,
    condition=None,
    unconditional_condition=None,
    num_steps=None,
    sigma_min=0.002,
    sigma_max=80,
    schedule_type='polynomial',
    schedule_rho=7,
    afs=False,
    denoise_to_zero=False,
    return_inters=False,
    predictor=None,
    train=False,
    step_idx=None,
    max_order=4,
    buffer_model=None,
    verbose=False,
    **kwargs
):
    """
    Implements the Improved Pseudo-Numerical Diffusion Method (IPNDM) sampler,
    with support for multi-point prediction, updated based on the new epd_sampler.
    """
    assert max_order >= 1 and max_order <= 4
    if buffer_model is None:
        buffer_model = []

    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    backend_type = getattr(net, "backend", None)

    # Main sampling loop.
    x_next = latents if backend_type == "sd3" else latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    if not train:
        buffer_model = []

    x_list = []
    x_list.append(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)

        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        hook.remove()
        t_cur_r = t_cur.reshape(-1, 1, 1, 1)
        t_next_r = t_next.reshape(-1, 1, 1, 1)

        if predictor is not None:
            try:
                num_points = predictor.num_points
            except AttributeError:
                num_points = predictor.module.num_points

            current_step_index = step_idx if step_idx is not None else i
            r_s, scale_dir_s, scale_time_s, weight_s = get_epd_prediction(
                predictor,
                current_step_index,
                net,
                unet_enc_out,
                use_afs,
                batch_size=latents.shape[0]
            )
            
            x_next = x_cur
            
            if verbose:
                print_str = f'step {i}: |'
                for j in range(num_points):
                    print_str += f'r{j}: {r_s[0,j,0,0]:.5f} '
                    print_str += f'st{j}: {scale_time_s[0,j,0,0]:.5f} '
                    print_str += f'sd{j}: {scale_dir_s[0,j,0,0]:.5f} '
                    print_str += f'w{j}: {weight_s[0,j,:,:].mean().item():.5f} |'
                # Assuming dist.print0 is a custom print function.
                # If not, replace with a standard print().
                # dist.print0(print_str)
                print(print_str)

            for j in range(num_points):
                r = r_s[:, j:j+1, :, :]
                scale_time = scale_time_s[:, j:j+1, :, :]
                scale_dir = scale_dir_s[:, j:j+1, :, :]
                w = weight_s[:, j:j+1, :, :]

                t_mid = (t_next_r ** r) * (t_cur_r ** (1 - r))
                
                # Predictor step (Adams-Bashforth) to find x_mid
                order = min(max_order, len(buffer_model) + 1)
                if order == 1:    # First Euler step.
                    x_mid = x_cur + (t_mid - t_cur_r) * d_cur
                elif order == 2:  # Use one history point.
                    x_mid = x_cur + (t_mid - t_cur_r) * (3 * d_cur - buffer_model[-1]) / 2
                elif order == 3:  # Use two history points.
                    x_mid = x_cur + (t_mid - t_cur_r) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
                elif order == 4:  # Use three history points.
                    x_mid = x_cur + (t_mid - t_cur_r) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24

                denoised_t_mid = get_denoised(net, x_mid, scale_time * t_mid,
                                              class_labels=class_labels,
                                              condition=condition,
                                              unconditional_condition=unconditional_condition)
                d_mid = (x_mid - denoised_t_mid) / t_mid
                
                # Accumulate weighted updates
                x_next = x_next + w * scale_dir * (t_next_r - t_cur_r) * d_mid

        else: # Standard IPNDM without predictor
            order = min(max_order, len(buffer_model) + 1)
            if order == 1:      # First Euler step.
                x_next = x_cur + (t_next_r - t_cur_r) * d_cur
            elif order == 2:    # Use one history point.
                x_next = x_cur + (t_next_r - t_cur_r) * (3 * d_cur - buffer_model[-1]) / 2
            elif order == 3:    # Use two history points.
                x_next = x_cur + (t_next_r - t_cur_r) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
            elif order == 4:    # Use three history points.
                x_next = x_cur + (t_next_r - t_cur_r) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        
        x_list.append(x_next)
        
        # Update history buffer
        if len(buffer_model) < max_order -1:
             buffer_model.append(d_cur.detach())
        elif max_order > 1:
            buffer_model.pop(0)
            buffer_model.append(d_cur.detach())

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, buffer_model, [], r_s, scale_dir_s, scale_time_s, weight_s
    
    return x_next, x_list


#----------------------------------------------------------------------------

def edm_sampler(
    net,
    latents,
    class_labels=None,
    condition=None,
    unconditional_condition=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    schedule_type=None,
    schedule_rho=7,
    afs=False,
    denoise_to_zero=False,
    return_inters=False,
    S_churn=0.0,
    S_min=0.0,
    S_max=float("inf"),
    S_noise=1.0,
    guidance_rate=1.0,
    **kwargs,
):
    device = latents.device
    dtype = latents.dtype
    backend_type = getattr(net, "backend", None)
    round_sigma = getattr(net, "round_sigma", lambda x: x)

    if backend_type == "sd3" and schedule_type == "flowmatch":
        # Use SD3 flowmatch schedule from backend for compatibility with SD3 training.
        t_steps = net.make_flowmatch_schedule(num_steps, device=device)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        x_next = latents.to(dtype)
    else:
        sigma_min = max(sigma_min, getattr(net, "sigma_min", sigma_min))
        sigma_max = min(sigma_max, getattr(net, "sigma_max", sigma_max))

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        rho = schedule_rho
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # For SD3, keep initial scaling consistent with flowmatch branch (no sigma multiply).
        x_next = latents.to(dtype if backend_type == "sd3" else torch.float64)
        if backend_type != "sd3":
            x_next = x_next * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        t_cur_val = float(t_cur)
        t_next_val = float(t_next)

        if backend_type == "sd3" and schedule_type == "flowmatch":
            gamma = 0.0
            t_hat = t_cur
            t_hat_val = t_cur_val
            noise = torch.zeros_like(x_next)
        elif backend_type == "sd3":
            gamma = 0.0
            t_hat = t_cur
            t_hat_val = t_cur_val
            noise = torch.zeros_like(x_next)
        else:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if (S_min <= t_cur <= S_max) else 0
            t_hat = round_sigma(t_cur + gamma * t_cur)
            t_hat_val = float(t_hat)
            noise = S_noise * torch.randn_like(x_next)

        x_hat = x_next + (t_hat_val**2 - t_cur_val**2) ** 0.5 * noise

        t_hat_tensor = torch.full(
            (latents.shape[0], 1, 1, 1), t_hat_val, device=device, dtype=dtype
        )
        denoised = get_denoised(
            net,
            x_hat.to(dtype),
            t_hat_tensor,
            class_labels=class_labels,
            condition=condition,
            unconditional_condition=unconditional_condition,
        ).to(torch.float64 if backend_type != "sd3" else dtype)

        d_cur = (x_hat - denoised) / t_hat_val
        x_next = x_hat + (t_next_val - t_hat_val) * d_cur

        if i < num_steps - 1:
            t_next_tensor = torch.full(
                (latents.shape[0], 1, 1, 1), t_next_val, device=device, dtype=dtype
            )
            denoised_next = get_denoised(
                net,
                x_next.to(dtype),
                t_next_tensor,
                class_labels=class_labels,
                condition=condition,
                unconditional_condition=unconditional_condition,
            ).to(torch.float64 if backend_type != "sd3" else dtype)
            d_prime = (x_next - denoised_next) / t_next_val
            x_next = x_hat + (t_next_val - t_hat_val) * (0.5 * d_cur + 0.5 * d_prime)

    result = x_next.to(dtype)

    if return_inters:
        return result, []
    return result, []


#----------------------------------------------------------------------------

@torch.no_grad()
def ddim_sampler(
    net,
    latents,
    class_labels=None,
    condition=None,
    unconditional_condition=None,
    num_steps=50,
    sigma_min=None,
    sigma_max=None,
    schedule_type=None,
    schedule_rho=None,
    afs=False,
    denoise_to_zero=False,
    return_inters=False,
    ddim_eta=0.0,
    guidance_rate=1.0,
    **kwargs,
):
    if num_steps is None:
        raise ValueError("num_steps must be specified for the DDIM sampler.")

    if abs(float(ddim_eta)) > 0.0:
        dist.print0("[DDIM] Ignore non-zero ddim_eta, now we only use ODE.")

    t_steps = get_schedule(
        num_steps,
        sigma_min,
        sigma_max,
        device=latents.device,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        net=net,
    ).to(latents.dtype)

    x_next = latents * t_steps[0]
    trajectory = [x_next.unsqueeze(0)] if return_inters else None

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        use_afs = afs and i == 0
        if use_afs:
            d_cur = x_cur / (1 + t_cur**2).sqrt()
        else:
            denoised = get_denoised(
                net,
                x_cur,
                t_cur,
                class_labels=class_labels,
                condition=condition,
                unconditional_condition=unconditional_condition,
            )
            d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        if return_inters:
            trajectory.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        t_final = t_steps[-1]
        x_next = get_denoised(
            net,
            x_next,
            t_final,
            class_labels=class_labels,
            condition=condition,
            unconditional_condition=unconditional_condition,
        )
        if return_inters:
            trajectory.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(trajectory, dim=0).to(latents.device)
    return x_next, trajectory


#----------------------------------------------------------------------------
