import os

src = "/gpfs/home6/scur0103/celeba-hq/subsampled_set/"
out_file = "subsampled_filenames_list.txt"

files = sorted(os.listdir(src))

content = "[" + ", ".join(files) + "]"

with open(out_file, "w") as f:
    f.write(content)

print(f"Wrote {len(files)} filenames to {out_file}")
