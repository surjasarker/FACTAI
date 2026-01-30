import os
import shutil

src = "/gpfs/home6/scur0103/clean_edit/downsampled_val_imgs/seed42"
dst = "/gpfs/home6/scur0103/clean_edit_subsample/downsampled_val_imgs/seed42"
list_dir = "/gpfs/home6/scur0103/celeba-hq/subsampled_set"

# get filenames to retain (110 expected)
keep_files = set(os.listdir(list_dir))

# ensure destination exists
os.makedirs(dst, exist_ok=True)

# loop through 25 subfolders
for sub in os.listdir(src):
    sub_src = os.path.join(src, sub)
    sub_dst = os.path.join(dst, sub)

    if not os.path.isdir(sub_src):
        continue

    os.makedirs(sub_dst, exist_ok=True)

    # loop through files, copy if in keep list
    for fname in os.listdir(sub_src):
        if fname in keep_files:
            shutil.copy2(os.path.join(sub_src, fname),
                         os.path.join(sub_dst, fname))

print("Done.")
