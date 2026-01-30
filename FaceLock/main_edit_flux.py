import argparse
from PIL import Image
import torch
from diffusers import Flux2KleinPipeline
from utils_flux import edit_prompts
import os
from codecarbon import EmissionsTracker
import torch.nn.functional as F
from functools import wraps
original_sdpa = F.scaled_dot_product_attention

# 2. Define a wrapper that filters out the bad argument
@wraps(original_sdpa)
def safe_sdpa(*args, **kwargs):
    # This is the line that stops the crash
    kwargs.pop('enable_gqa', None)
    return original_sdpa(*args, **kwargs)

# 3. Replace the function in the torch namespace
F.scaled_dot_product_attention = safe_sdpa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--src_dir", required=True, help="path to the directory of the src images")
    parser.add_argument("--edit_dir", required=True, help="path to the directory of the edited images")
    # edit configuration
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()

    guidance_scale = args.guidance_scale
    image_guidance_scale = args.image_guidance_scale
    num_inference_steps = args.num_inference_steps
    if args.seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed
    model = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16).to("cuda")
    src_dir = args.src_dir
    edit_dir = args.edit_dir
    image_files = sorted(os.listdir(src_dir))
    
    tracker = EmissionsTracker(project_name="githubparams_edit",log_level="info")
    tracker.start()

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(src_dir, image_file)
        src_image = Image.open(image_path).convert("RGB")
        for idx, prompt in edit_prompts.items():
            save_dir = os.path.join(edit_dir, f"seed{seed}", f"prompt{idx}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.manual_seed(seed)
            edit_image = model(
                prompt=prompt,
                image=src_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=3.5
            ).images[0]
            edit_image.save(os.path.join(save_dir, image_file))
        if (i + 1) % 100 == 0:
            print(f"Edited [{i + 1}/{len(image_files)}]")

    emissions = tracker.stop()
    print(f"Emissions: {emissions} kg CO2")
