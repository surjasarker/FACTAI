import os
import shutil

src_list = "/gpfs/home6/scur0103/celeba-hq/subsampled_set/"
src_search = "/gpfs/home6/scur0103/defended-celeba-hq/facelock/downsampled_val_imgs/budget_0.02/"
dst = "/gpfs/home6/scur0103/defended-celeba-hq/facelock/subsampled_set_paper_budget/budget_0.02/"

os.makedirs(dst, exist_ok=True)

# list of 110 filenames
files = set(os.listdir(src_list))

count = 0
for f in files:
    src_path = os.path.join(src_search, f)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst)
        count += 1

print(f"Copied {count} files.")
