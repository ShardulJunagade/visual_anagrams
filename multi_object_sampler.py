# multi_object_sampler.py

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from diffusers.utils.torch_utils import randn_tensor

@torch.no_grad()
def sample_multi_object_anagram_stage_1(
    model,
    prompt_embeds,
    negative_prompt_embeds,
    views,
    masks,
    num_inference_steps=100,
    guidance_scale=7.0,
    generator=None,
    num_objects=1,
):
    # --- Setup ---
    device = torch.device('cuda')
    height = width = model.unet.config.sample_size
    num_views = len(views)

    # Reshape prompts for easier access: (num_views, num_objects, seq_len, embed_dim)
    prompt_embeds = prompt_embeds.view(num_views, num_objects, -1, prompt_embeds.shape[-1])
    negative_prompt_embeds = negative_prompt_embeds.view(num_views, num_objects, -1, negative_prompt_embeds.shape[-1])

    # --- Initial Noise ---
    noisy_images = randn_tensor((1, model.unet.config.in_channels, height, width), generator=generator, device=device, dtype=prompt_embeds.dtype)
    
    # --- Timesteps ---
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    for i, t in enumerate(tqdm(timesteps)):
        inverted_view_noise_preds = []
        inverted_view_variance_preds = []

        # 1. --- OUTER LOOP (Visual Anagrams Logic) ---
        for view_idx, view_fn in enumerate(views):
            
            viewed_noisy_image = view_fn.view(noisy_images[0])
            model_input_for_view = viewed_noisy_image.repeat(num_objects, 1, 1, 1)

            view_prompt_embeds = prompt_embeds[view_idx]
            view_negative_prompt_embeds = negative_prompt_embeds[view_idx]
            
            cfg_prompt_embeds = torch.cat([view_negative_prompt_embeds, view_prompt_embeds])
            
            model_input = torch.cat([model_input_for_view] * 2)
            model_input = model.scheduler.scale_model_input(model_input, t)

            # --- Predict Noise ---
            unet_pred = model.unet(
                model_input,
                t,
                encoder_hidden_states=cfg_prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # --- Perform CFG ---
            noise_pred_uncond, noise_pred_text = unet_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # --- Split noise and variance ---
            # FIX: Use chunk(2, dim=1) to correctly split the 6-channel output into two 3-channel tensors.
            noise_pred, predicted_variance = noise_pred.chunk(2, dim=1)

            # 2. --- INNER LOOP (MultiDiffusion Logic) ---
            transformed_masks = view_fn.view(masks)
            combined_noise_for_view = torch.sum(noise_pred * transformed_masks, dim=0)
            combined_variance_for_view = torch.mean(predicted_variance, dim=0)

            # 3. --- Invert the transformation (Visual Anagrams Logic) ---
            inverted_noise = view_fn.inverse_view(combined_noise_for_view)
            inverted_variance = view_fn.inverse_view(combined_variance_for_view)
            inverted_view_noise_preds.append(inverted_noise)
            inverted_view_variance_preds.append(inverted_variance)

        # --- Average all view predictions (Visual Anagrams Logic) ---
        final_noise_pred = torch.stack(inverted_view_noise_preds).mean(dim=0)
        final_variance_pred = torch.stack(inverted_view_variance_preds).mean(dim=0)
        
        # --- Denoise Step ---
        noisy_images = model.scheduler.step(
            model_output=final_noise_pred, 
            timestep=t, 
            sample=noisy_images, 
            variance_noise=final_variance_pred,
            generator=generator, 
            return_dict=False
        )[0]

    return noisy_images