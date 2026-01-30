import os
import shutil

# BASE PATHS
base_set = "/gpfs/home6/scur0103/celeba-hq/subsampled_set/"
edited_base = "/gpfs/home6/scur0103/edits-celeba-hq/"
defended_base = "/gpfs/home6/scur0103/defended-celeba-hq/"
missed_base = "/gpfs/home6/scur0103/defended-celeba-hq_missed/"

methods = ["facelock", "encoder", "photoguard"]

# Folder naming pieces that stay consistent
edited_suffix = "subsampled_0.10step0.02steps100/seed42/"
defended_suffix = "subsampled_0.10step0.02/budget_0.1/"

NUM_SUBFOLDERS = 25  # depends on your dataset; adjust if needed

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# List base filenames
all_files = set(os.listdir(base_set))
print(f"Found {len(all_files)} base files.")

for method in methods:
    edited_root = os.path.join(edited_base, method, edited_suffix)
    defended_root = os.path.join(defended_base, method, defended_suffix)
    missed_root = os.path.join(missed_base, method, defended_suffix)

    ensure_dir(missed_root)

    # Check 25 subfolders in edited directory
    subfolders = sorted([os.path.join(edited_root, d) for d in os.listdir(edited_root)
                         if os.path.isdir(os.path.join(edited_root, d))])

    if len(subfolders) != NUM_SUBFOLDERS:
        print(f"WARNING: {method} has {len(subfolders)} subfolders, expected {NUM_SUBFOLDERS}.")

    print(f"Processing {method}: {len(subfolders)} edit subfolders.")

    for fname in all_files:
        present_everywhere = True

        for subfolder in subfolders:
            if not os.path.exists(os.path.join(subfolder, fname)):
                present_everywhere = False
                break

        if not present_everywhere:
            defended_path = os.path.join(defended_root, fname)
            if os.path.exists(defended_path):
                shutil.copy(defended_path, missed_root)
            else:
                print(f"[MISSING DEFENDED] {method}: {fname}")

    print(f"Finished {method}.\n")
