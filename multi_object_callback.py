# multi_object_callback.py

import torch

def create_multi_object_callback(model, prompt_embeds, negative_prompt_embeds, views, masks, guidance_scale, num_objects):
    """
    Creates a callback function that implements the MultiDiffusion and Visual Anagrams logic
    at each step of the denoising process.
    """
    # Reshape prompts for easier access: (num_views, num_objects, seq_len, embed_dim)
    prompt_embeds = prompt_embeds.view(len(views), num_objects, -1, prompt_embeds.shape[-1])
    negative_prompt_embeds = negative_prompt_embeds.view(len(views), num_objects, -1, negative_prompt_embeds.shape[-1])

    # This is the function that will be called by the pipeline at each step
    def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
        with torch.no_grad():
            inverted_view_noise_preds = []

            # 1. --- OUTER LOOP (Visual Anagrams Logic) ---
            for view_idx, view_fn in enumerate(views):
                # Get the noisy latent for the current view
                viewed_noisy_image = view_fn.view(latents)
                
                # Repeat the viewed latent for each object to get batched input
                model_input_for_view = viewed_noisy_image.repeat(num_objects, 1, 1, 1)

                # Get prompts for this specific view
                view_prompt_embeds = prompt_embeds[view_idx]
                view_negative_prompt_embeds = negative_prompt_embeds[view_idx]
                
                # For CFG, concat negative and positive prompts
                cfg_prompt_embeds = torch.cat([view_negative_prompt_embeds, view_prompt_embeds])
                
                # For CFG, duplicate model input
                model_input = torch.cat([model_input_for_view] * 2)
                model_input = model.scheduler.scale_model_input(model_input, timestep)

                # --- Predict Noise (UNet output is 6 channels for IF) ---
                unet_pred = model.unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=cfg_prompt_embeds,
                )[0]

                # --- Perform CFG ---
                noise_pred_uncond, noise_pred_text = unet_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # --- INNER LOOP (MultiDiffusion Logic) ---
                # `noise_pred` is (num_objects, 6, H, W). We combine it using masks.
                transformed_masks = view_fn.view(masks)
                # We need to expand masks to have 6 channels to multiply with the noise_pred
                expanded_masks = transformed_masks.repeat(1, model.unet.config.out_channels, 1, 1)
                combined_pred_for_view = torch.sum(noise_pred * expanded_masks, dim=0, keepdim=True)
                
                # --- Invert the transformation (Visual Anagrams Logic) ---
                # Bring the combined 6-channel prediction back to the original coordinate space
                inverted_pred = view_fn.inverse_view(combined_pred_for_view)
                inverted_view_noise_preds.append(inverted_pred)

            # --- Average all view predictions ---
            # This is our "consensus" 6-channel prediction
            final_noise_pred = torch.stack(inverted_view_noise_preds).mean(dim=0)
            
            # --- Hijack the pipeline's internal state ---
            # We modify the pipeline's internal noise prediction (`model.noise_pred`)
            # with our averaged prediction before it takes the denoising step.
            model.noise_pred = final_noise_pred

        return latents

    return callback_fn