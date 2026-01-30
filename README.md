# Revisiting FaceLock: A Reproducibility Study

This is the github page for the paper, [Revisiting \textsc{FaceLock}: A Reproducibility Study on "Edit Away and My Face Will not Stay: Personal Biometric Defense against Malicious Generative Editing"].

This paper attempts to reproduce and extend the results from the paper [Edit Away and My Face Will not Stay: Personal Biometric Defense against Malicious Generative Editing](https://arxiv.org/abs/2411.16832).
A large part of the code used in this repo was taken directly from their github repository: https://github.com/taco-group/FaceLock

## Abstract

## Environments

We provide two environments, one for [the instructpix2pix model](environment.yml) and one for [the flux model](environment_flux.yml). They have different environments because the flux model requires updated versions of certain packages which dont support instructpix2pix

```bash
conda env create -f pix2pix_env.yml
conda activate facelock_pix2pix
```

```bash
conda env create -f flux_env.yml
conda activate facelock_flux
```

For the explanation of how to run the code, first we will go over the preprocessing steps which are universial over both models and then we will show how to edit and defend images per model.

## Preprocessing steps
Before running the defences and edits we need to preprocess the data to ensure the data that is being inputted is the correct size(dont forget to activate one of the environments first).
```bash
cd preprocessing
python downsample_preprocess.py \
    --input_folder path/to/input_images \
    --output_folder path/to/output_images
```
Please note that the images are also converted to png as a preprocessing step but this is done automatically in defense.py and defense_flux.py so there is no need to do this before defending.

## Image Defending and Editing with InstructPix2Pix
The code for the Defending and Editing in InstructPix2Pix is a slightly altered version of the code from: - Wang et al., "Edit away and my
face will not stay: Personal biometric defense against malicious generative editing", Computer Vision and Pattern Recognition (2024)., https://github.com/taco-group/FaceLock

Next we demonstrate the code for handling image defending and editing across multiple images for the instructPix2Pix model.

```bash
cd FaceLock
python main_defend.py --image_dir=${input image dir} --output_dir=${output image dir} --defend_method=${selected defense method} [--attack_budget=0.03 --step_size=0.01 --num_iters=100 --help]
```

Arguments explanation:

- `image_dir` the path to the directory of the source images to be protected
- `output_dir` the path to the directory containing the protected images generated
- other arguments are similar to the single image defending version, use `help` to see more details


```bash
cd FaceLock
python main_edit.py --src_dir=${input image dir} --edit_dir=${output image dir} [--num_inference_steps=100 --image_guidance_scale=1.5 --guidance_scale=7.5 --help]
```

Arguments explanation:

- `src_dir` the path to the directory of the source images to be edited
- `edit_dir` the path to the directory containing the edited images generated
- other arguments are similar to the single image editing version, use `help` to see more details

## Image Defending and Editing with FLUX.2-klein-9B
The code for the Defending and Editing with Flux.2-klein-9B is a modified version of the InstructPix2Pix code which was taken directly from: - Wang et al., "Edit away and my
face will not stay: Personal biometric defense against malicious generative editing", Computer Vision and Pattern Recognition (2024)., https://github.com/taco-group/FaceLock


Next we demonstrate the code for handling image defending and editing across multiple images for the FLUX.2-klein-9B model.


```bash
cd FaceLock
python main_defend_flux.py --image_dir=${input image dir} --output_dir=${output image dir} --defend_method=${selected defense method} [--attack_budget=0.03 --step_size=0.01 --num_iters=100 --help]
```

Arguments explanation:

- `image_dir` the path to the directory of the source images to be protected
- `output_dir` the path to the directory containing the protected images generated
- other arguments are similar to the single image defending version, use `help` to see more details

  
```bash
cd FaceLock
python main_edit_flux.py --src_dir=${input image dir} --edit_dir=${output image dir} [--num_inference_steps=100 --image_guidance_scale=1.5 --guidance_scale=7.5 --help]
```

Arguments explanation:

- `src_dir` the path to the directory of the source images to be edited
- `edit_dir` the path to the directory containing the edited images generated
- other arguments are similar to the single image editing version, use `help` to see more details


## Purification
Next we demonstrate how to apply the purification methods we tested in the paper to data.

for the JPEG75 purification:
```bash
cd purification
python jpeg75_purification.py \
    --input_dir path/to/input_images \
    --output_dir path/to/output_images
```
Arguments explanation:

- `input_dir` the path to the directory of the defended images to be purified
- `output_dir` the path to the directory where the purified images should be stored



for the blur purification:
```bash
cd purification
python blur_purification.py \
    --input_dir path/to/input_images \
    --output_dir path/to/output_images
```
Arguments explanation:

- `input_dir` the path to the directory of the defended images to be purified
- `output_dir` the path to the directory where the purified images should be stored



for the rotate purification:
```bash
cd purification
python rotate_purification.py \
    --input_dir path/to/input_images \
    --output_dir path/to/output_images
```
Arguments explanation:

- `input_dir` the path to the directory of the defended images to be purified
- `output_dir` the path to the directory where the purified images should be stored


## Evaluation
The code for the Evaluation is directly taken from: - Wang et al., "Edit away and my
face will not stay: Personal biometric defense against malicious generative editing", Computer Vision and Pattern Recognition (2024)., https://github.com/taco-group/FaceLock

They provide the evaluation code for computing the `PSNR, SSIM, LPIPS, CLIP-S, CLIP-I, FR` metrics mentioned in the paper.

```bash
cd Facelock/evaluation
# PSNR metric
python eval_psnr.py --clean_edit_dir=${path to the clean edits} --defend_edit_dirs ${sequence of path to the protected edits} --seed=${the seed used to edit and evaluate on}
# SSIM metric
python eval_ssim.py --clean_edit_dir=${path to the clean edits} --defend_edit_dirs ${sequence of path to the protected edits} --seed=${the seed used to edit and evaluate on}
# LPIPS metric
python eval_lpips.py --clean_edit_dir=${path to the clean edits} --defend_edit_dirs ${sequence of path to the protected edits} --seed=${the seed used to edit and evaluate on}
# CLIP-S metric
python eval_clip_s.py --src_dir=${path to the source images} --defend_edit_dirs ${sequence of path to the protected edits} --seed=${the seed used to edit and evaluate on} [--clean_edit_dir=${path to the clean edits}]
# CLIP-I metric
python eval_clip_i.py --src_dir=${path to the source images} --defend_edit_dirs ${sequence of path to the protected edits} --seed=${the seed used to edit and evaluate on} [--clean_edit_dir=${path to the clean edits}]
# FR metric
python eval_facial.py --src_dir=${path to the source images} --defend_edit_dirs ${sequence of path to the protected edits} --seed=${the seed used to edit and evaluate on} [--clean_edit_dir=${path to the clean edits}]
```

