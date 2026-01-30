import os
import shutil

# paths
subsample_list = "/gpfs/home6/scur0103/celeba-hq/subsampled_set/"
src_root = "/gpfs/home6/scur0103/edits-celeba-hq/facelock/downsampled_val_imgs/seed42/"
dst_root = "/gpfs/home6/scur0103/edits-celeba-hq/facelock/subsampled_set_paper_budget/seed42/"

# read the 110 filenames long
subsample_files = set(os.listdir(subsample_list))

# create destination root
os.makedirs(dst_root, exist_ok=True)

# iterate through each edit folder
for edit_folder in os.listdir(src_root):
    src_dir = os.path.join(src_root, edit_folder)
    dst_dir = os.path.join(dst_root, edit_folder)

    if not os.path.isdir(src_dir):
        continue

    # create matching subfolder
    os.makedirs(dst_dir, exist_ok=True)

    copied = 0
    for img_name in subsample_files:
        src_img = os.path.join(src_dir, img_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_dir)
            copied += 1

    print(f"{edit_folder}: copied {copied} images")
