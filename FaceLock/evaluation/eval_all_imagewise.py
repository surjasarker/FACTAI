# Compute all metrics (CLIP-I, CLIP-S, Facial, LPIPS, PSNR, SSIM) image-wise
# Instead of averaging per-prompt, this averages across prompts for each image
import argparse
import pandas as pd
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    edit_prompts,
    get_image_embeddings,
    get_text_embeddings,
    load_model_by_repo_id,
    compute_score,
    pil_to_input
)

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize CLIP models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize LPIPS
lpips_fn = lpips.LPIPS(net='vgg').to(device)

# Initialize PSNR and SSIM
psnr_fn = PeakSignalNoiseRatio(data_range=1.0)
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)

# Initialize facial recognition model
repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
aligner_id = 'minchul/cvlface_DFA_mobilenet'
fr_model = None
aligner = None

def init_facial_models():
    global fr_model, aligner
    if fr_model is None:
        fr_model = load_model_by_repo_id(
            repo_id=repo_id,
            save_path=f'{os.environ["HF_HOME"]}/{repo_id}',
            HF_TOKEN=os.environ.get('HUGGINGFACE_HUB_TOKEN')
        ).to(device)
        aligner = load_model_by_repo_id(
            repo_id=aligner_id,
            save_path=f'{os.environ["HF_HOME"]}/{aligner_id}',
            HF_TOKEN=os.environ.get('HUGGINGFACE_HUB_TOKEN')
        ).to(device)


def compute_metrics_for_image(
    src_image_path,
    edit_image_paths,
    clean_edit_image_paths,
    prompts,
    compute_facial=True
):
    """
    Compute all metrics for a single source image across all prompts.

    Args:
        src_image_path: Path to the source image
        edit_image_paths: List of paths to edited images (one per prompt)
        clean_edit_image_paths: List of paths to clean edited images (one per prompt), can be None
        prompts: List of prompts corresponding to each edit
        compute_facial: Whether to compute facial similarity

    Returns:
        Dictionary with mean metrics across all prompts for this image
    """
    src_image = Image.open(src_image_path).convert("RGB")
    src_embeddings = get_image_embeddings(src_image, clip_processor, clip_model, device)

    clip_i_scores = []
    clip_s_scores = []
    facial_scores = []
    lpips_scores = []
    psnr_scores = []
    ssim_scores = []

    for idx, (edit_path, prompt) in enumerate(zip(edit_image_paths, prompts)):
        edit_image = Image.open(edit_path).convert("RGB")

        # CLIP-I: Image similarity between source and edit
        edit_embeddings = get_image_embeddings(edit_image, clip_processor, clip_model, device)
        clip_i = F.cosine_similarity(edit_embeddings, src_embeddings, dim=-1).mean().item()
        clip_i_scores.append(clip_i)

        # CLIP-S: Directional similarity with text
        text_embeddings = get_text_embeddings(prompt, clip_processor, clip_model, device)
        delta_embeddings = edit_embeddings - src_embeddings
        clip_s = F.cosine_similarity(delta_embeddings, text_embeddings).item()
        clip_s_scores.append(clip_s)

        # Facial similarity
        if compute_facial:
            try:
                facial_sim = compute_score(
                    pil_to_input(src_image).to(device),
                    pil_to_input(edit_image).to(device),
                    aligner,
                    fr_model
                ).item()
                facial_scores.append(facial_sim)
            except Exception as e:
                facial_scores.append(np.nan)

        # LPIPS, PSNR, SSIM (comparing defended edit to clean edit)
        if clean_edit_image_paths is not None:
            clean_edit_path = clean_edit_image_paths[idx]
            clean_edit_image = Image.open(clean_edit_path).convert("RGB")

            # LPIPS
            clean_tensor_lpips = lpips.im2tensor(lpips.load_image(clean_edit_path)).to(device)
            edit_tensor_lpips = lpips.im2tensor(lpips.load_image(edit_path)).to(device)
            lpips_val = lpips_fn(clean_tensor_lpips, edit_tensor_lpips).item()
            lpips_scores.append(lpips_val)

            # PSNR
            edit_tensor = T.ToTensor()(edit_image)
            clean_tensor = T.ToTensor()(clean_edit_image)
            psnr_val = psnr_fn(edit_tensor, clean_tensor).item()
            psnr_scores.append(psnr_val)

            # SSIM
            edit_tensor_ssim = T.ToTensor()(edit_image).unsqueeze(0)
            clean_tensor_ssim = T.ToTensor()(clean_edit_image).unsqueeze(0)
            ssim_val = ssim_fn(edit_tensor_ssim, clean_tensor_ssim).item()
            ssim_scores.append(ssim_val)

    result = {
        'clip_i': np.mean(clip_i_scores),
        'clip_s': np.mean(clip_s_scores),
    }

    if compute_facial and facial_scores:
        result['facial'] = np.nanmean(facial_scores)

    if clean_edit_image_paths is not None:
        result['lpips'] = np.mean(lpips_scores)
        result['psnr'] = np.mean(psnr_scores)
        result['ssim'] = np.mean(ssim_scores)

    return result


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, type=str,
                        help="path to the directory containing the source images")
    parser.add_argument("--clean_edit_dir", default=None,
                        help="path to the directory containing edits on the unprotected images")
    parser.add_argument("--defend_edit_dirs", required=True, nargs="+",
                        help="path to the directory containing different attack budget subdirectories")
    parser.add_argument("--seed", required=True, type=int,
                        help="the seed to evaluate on")
    parser.add_argument("--output", default="imagewise_metrics.csv",
                        help="output CSV filename")
    parser.add_argument("--no_facial", action="store_true",
                        help="skip facial similarity computation")
    args = parser.parse_args()

    src_dir = args.src_dir
    src_image_files = sorted(os.listdir(src_dir))
    num_images = len(src_image_files)
    prompt_num = len(edit_prompts)
    seed = args.seed

    for x in args.defend_edit_dirs:
        assert os.path.exists(x), f"Directory {x} does not exist"

    # Initialize facial models if needed
    if not args.no_facial:
        print("Initializing facial recognition models...")
        init_facial_models()

    results = []

    # Process clean edits if provided
    if args.clean_edit_dir is not None:
        print("Processing clean edits...")
        clean_seed_dir = os.path.join(args.clean_edit_dir, f"seed{seed}")

        for k in tqdm(range(num_images), desc="Clean"):
            src_image_path = os.path.join(src_dir, src_image_files[k])
            image_name = src_image_files[k]

            edit_paths = []
            prompts = []
            for i in range(prompt_num):
                prompt_dir = os.path.join(clean_seed_dir, f"prompt{i}")
                edit_files = sorted(os.listdir(prompt_dir))
                edit_paths.append(os.path.join(prompt_dir, edit_files[k]))
                prompts.append(edit_prompts[i])

            metrics = compute_metrics_for_image(
                src_image_path,
                edit_paths,
                None,  # No clean comparison for clean edits
                prompts,
                compute_facial=not args.no_facial
            )

            metrics['method'] = 'clean'
            metrics['image'] = image_name
            results.append(metrics)

    # Process defended edits
    for edit_dir in args.defend_edit_dirs:
        eps_dirs = sorted(os.listdir(edit_dir))
        for eps_dir in eps_dirs:
            cur_method = os.path.join(edit_dir, eps_dir)
            print(f"Processing {cur_method}...")
            seed_dir = os.path.join(cur_method, f"seed{seed}")
            clean_seed_dir = os.path.join(args.clean_edit_dir, f"seed{seed}") if args.clean_edit_dir else None

            for k in tqdm(range(num_images), desc=eps_dir):
                src_image_path = os.path.join(src_dir, src_image_files[k])
                image_name = src_image_files[k]

                edit_paths = []
                clean_edit_paths = [] if args.clean_edit_dir else None
                prompts = []

                for i in range(prompt_num):
                    prompt_dir = os.path.join(seed_dir, f"prompt{i}")
                    edit_files = sorted(os.listdir(prompt_dir))
                    edit_paths.append(os.path.join(prompt_dir, edit_files[k]))
                    prompts.append(edit_prompts[i])

                    if args.clean_edit_dir:
                        clean_prompt_dir = os.path.join(clean_seed_dir, f"prompt{i}")
                        clean_edit_files = sorted(os.listdir(clean_prompt_dir))
                        clean_edit_paths.append(os.path.join(clean_prompt_dir, clean_edit_files[k]))

                metrics = compute_metrics_for_image(
                    src_image_path,
                    edit_paths,
                    clean_edit_paths,
                    prompts,
                    compute_facial=not args.no_facial
                )

                metrics['method'] = cur_method
                metrics['image'] = image_name
                results.append(metrics)

    # Create DataFrame and save
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['method', 'image', 'clip_i', 'clip_s']
    if 'facial' in df.columns:
        cols.append('facial')
    if 'lpips' in df.columns:
        cols.extend(['lpips', 'psnr', 'ssim'])
    df = df[cols]

    print("\n" + "="*80)
    print("Image-wise Results:")
    print("="*80)
    print(df)

    # Print summary statistics per method
    print("\n" + "="*80)
    print("Summary Statistics (Mean per Method):")
    print("="*80)
    summary = df.groupby('method').mean(numeric_only=True)
    print(summary)

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Also save summary
    summary_file = args.output.replace('.csv', '_summary.csv')
    summary.to_csv(summary_file)
    print(f"Summary saved to {summary_file}")
