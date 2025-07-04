# generate_multi_object_anagram.py

import argparse
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms.functional as TF
# --- Use the correct, specific pipeline ---
from diffusers import IFPipeline

from visual_anagrams.views import get_views
from visual_anagrams.utils import save_illusion, save_metadata
# --- Import our new callback creator ---
from multi_object_callback import create_multi_object_callback

def preprocess_mask(mask_path, size=64):
    mask = Image.open(mask_path).convert("L")
    mask = TF.to_tensor(mask)
    mask = TF.resize(mask, (size, size), interpolation=TF.InterpolationMode.NEAREST)
    mask = (mask > 0.5).float()
    return mask

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, type=str)
parser.add_argument("--save_dir", type=str, default='results_multi_object', help='Location to save samples')
parser.add_argument("--prompts", required=True, type=str, nargs='+', help='Prompts in order: [view1_obj1, view1_obj2, ..., view2_obj1, view2_obj2, ...]')
parser.add_argument("--masks", required=True, type=str, nargs='+', help='Paths to mask files for each object.')
parser.add_argument("--views", required=True, type=str, nargs='+', help='Name of views to use.')
parser.add_argument("--style", default='', type=str, help='Optional style string to prepend to prompts')
parser.add_argument("--num_inference_steps", type=int, default=150)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--guidance_scale", type=float, default=9.0)
parser.add_argument("--device", type=str, default='cuda')
args = parser.parse_args()

# --- Validation ---
num_views = len(args.views)
num_objects = len(args.masks)
if len(args.prompts) != num_views * num_objects:
    raise ValueError(f"Number of prompts must be num_views * num_objects. "
                     f"Expected {num_views * num_objects}, but got {len(args.prompts)}.")

# --- Setup Directories ---
save_dir = Path(args.save_dir) / args.name
save_dir.mkdir(exist_ok=True, parents=True)

# --- Load Models ---
# Use the specific IFPipeline to ensure all components are loaded correctly
stage_1 = IFPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0",
                variant="fp16",
                torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# --- Prepare Inputs ---
views = get_views(args.views)

# Prepare prompts and embeddings
prompts = [f'{args.style} {p}'.strip() for p in args.prompts]
prompt_embeds, negative_prompt_embeds = stage_1.encode_prompt(prompts)

# Prepare masks (for stage 1 at 64x64)
masks = torch.stack([preprocess_mask(m) for m in args.masks]).to(args.device, dtype=torch.float16)

# Normalize masks
mask_sum = torch.sum(masks, dim=0, keepdim=True)
masks = torch.where(mask_sum > 0, masks / (mask_sum + 1e-8), masks)

# --- Generation Loop ---
for i in range(args.num_samples):
    print(f"Generating sample {i+1}/{args.num_samples}")
    
    generator = torch.manual_seed(args.seed + i)
    sample_dir = save_dir / f'{args.seed + i:04}'
    sample_dir.mkdir(exist_ok=True, parents=True)

    # --- Create the callback function with all our parameters ---
    callback = create_multi_object_callback(
        stage_1,
        prompt_embeds,
        negative_prompt_embeds,
        views,
        masks,
        args.guidance_scale,
        num_objects
    )

    # --- Run the pipeline with the callback ---
    # The pipeline handles the entire denoising loop.
    # Our logic is injected at each step via the callback.
    # NOTE: We pass a single dummy prompt because the callback handles the real ones.
    image = stage_1(
        prompt="dummy",  # This is ignored but required by the function signature
        generator=generator,
        num_inference_steps=args.num_inference_steps,
        callback=callback,
        guidance_scale=args.guidance_scale # Pass guidance scale here too
    ).images[0]
    
    # Save the final illusion and its transformed views
    # We need to convert the PIL Image back to a tensor for save_illusion
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    save_illusion(image_tensor, views, sample_dir)
    print(f"Saved illusion to {sample_dir}")