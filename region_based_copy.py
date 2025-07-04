from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height //= 8
    panorama_width //= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    views = []
    for i in range(int(num_blocks_height * num_blocks_width)):
        h_start = (i // num_blocks_width) * stride
        h_end = h_start + window_size
        w_start = (i % num_blocks_width) * stride
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

class IdentityTransform:
    def __call__(self, x): return x
    def inverse(self, x): return x

class Rotate90Transform:
    def __call__(self, x): return torch.rot90(x, k=1, dims=(-2, -1))
    def inverse(self, x): return torch.rot90(x, k=3, dims=(-2, -1))

class Rotate180Transform:
    def __call__(self, x): return torch.rot90(x, k=2, dims=(-2, -1))
    def inverse(self, x): return torch.rot90(x, k=2, dims=(-2, -1))

class Rotate270Transform:
    def __call__(self, x): return torch.rot90(x, k=3, dims=(-2, -1))
    def inverse(self, x): return torch.rot90(x, k=1, dims=(-2, -1))

class MultiDiffusionWithAnagrams(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()
        print(f'[INFO] loading stable diffusion...')
        if hf_key: model_key = hf_key
        elif sd_version == '2.1': model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0': model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5': model_key = "runwayml/stable-diffusion-v1-5"
        else: model_key = sd_version

        self.device = device
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        tok = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(tok.input_ids.to(self.device))[0]
        uncond_tok = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_tok.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        return posterior.sample() * 0.18215

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def generate(self, masks, prompts, anagram_prompts, transforms, negative_prompts='', height=512, width=2048, num_inference_steps=50, guidance_scale=7.5, bootstrapping=20):
        text_embeds = [self.get_text_embeds(p, n) for p, n in zip(prompts, negative_prompts)]
        anagram_embeds = [self.get_text_embeds(p, n) for p, n in zip(anagram_prompts, negative_prompts)]
        latent = torch.randn((1, self.unet.config.in_channels, height // 8, width // 8), device=self.device)
        views = get_views(height, width)
        count, value = torch.zeros_like(latent), torch.zeros_like(latent)
        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for t in tqdm(self.scheduler.timesteps):
                count.zero_(); value.zero_()
                for h0, h1, w0, w1 in views:
                    patch = latent[:, :, h0:h1, w0:w1]
                    mask_patch = masks[:, :, h0:h1, w0:w1]
                    region_idx = mask_patch.reshape(mask_patch.shape[0], -1).sum(dim=1).argmax().item()

                    transform = transforms[region_idx]
                    embed = text_embeds[region_idx]
                    ana_embed = anagram_embeds[region_idx]

                    patch_rep = patch.repeat(2, 1, 1, 1)
                    noise_pred = self.unet(patch_rep, t, encoder_hidden_states=embed)['sample']
                    transformed = transform(patch_rep)
                    noise_ana = self.unet(transformed, t, encoder_hidden_states=ana_embed)['sample']
                    noise_ana_inv = transform.inverse(noise_ana)
                    combined_noise = 0.5 * (noise_pred + noise_ana_inv)
                    uncond, cond = combined_noise.chunk(2)
                    guided = uncond + guidance_scale * (cond - uncond)
                    denoised = self.scheduler.step(guided, t, patch_rep)['prev_sample'][0:1]
                    value[:, :, h0:h1, w0:w1] += denoised * mask_patch[region_idx:region_idx+1]
                    count[:, :, h0:h1, w0:w1] += mask_patch[region_idx:region_idx+1]
                latent = torch.where(count > 0, value / count, value)

        img = self.decode_latents(latent)
        return T.ToPILImage()(img[0].cpu())

def preprocess_mask(path, h, w, device):
    mask = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    mask = torch.from_numpy((mask > 0.5).astype(np.float32))[None, None].to(device)
    return torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_prompt', type=str, default='')
    parser.add_argument('--bg_negative', type=str, default='')
    parser.add_argument('--fg_prompts', nargs='*', type=str, required=True)
    parser.add_argument('--fg_negative', nargs='*', type=str, default=None)
    parser.add_argument('--fg_masks', nargs='*', type=str, required=True)
    parser.add_argument('--anagram_prompts', nargs='*', type=str, required=True)
    parser.add_argument('--transforms', nargs='*', type=str, required=True, choices=['identity', 'rot90', 'rot180', 'rot270'])
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'])
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=768)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--bootstrapping', type=int, default=20)
    parser.add_argument('--outfile', type=str, default='out.png')
    opt = parser.parse_args()

    seed_everything(opt.seed)
    device = torch.device('cuda')

    sd = MultiDiffusionWithAnagrams(device, opt.sd_version)
    fg_masks = torch.cat([preprocess_mask(mp, opt.H // 8, opt.W // 8, device) for mp in opt.fg_masks])
    bg_mask = 1 - fg_masks.sum(dim=0, keepdim=True).clamp(max=1)
    masks = torch.cat([bg_mask, fg_masks], dim=0)

    opt.fg_negative = opt.fg_negative or [''] * len(opt.fg_prompts)
    prompts = [opt.bg_prompt] + opt.fg_prompts
    neg_prompts = [opt.bg_negative] + opt.fg_negative
    anagram_prompts = [opt.bg_prompt] + opt.anagram_prompts

    transform_map = {
        'identity': IdentityTransform(),
        'rot90': Rotate90Transform(),
        'rot180': Rotate180Transform(),
        'rot270': Rotate270Transform()
    }
    transforms = [IdentityTransform()] + [transform_map[t] for t in opt.transforms]

    img = sd.generate(masks, prompts, anagram_prompts, transforms, neg_prompts, opt.H, opt.W, opt.steps, guidance_scale=7.5, bootstrapping=opt.bootstrapping)
    os.makedirs(os.path.dirname(opt.outfile) or '.', exist_ok=True)
    img.save(opt.outfile)
