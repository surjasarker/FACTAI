import os
from pathlib import Path
from PIL import Image
import random 

src = Path("/gpfs/home6/scur0103/defended-celeba-hq/facelock/subsampled_0.02step0.003png/budget_0.02/")
dst = Path("/gpfs/home6/scur0103/defended-celeba-hq/facelock/subsampled_0.02step0.003png_rotate/budget_0.02/")

dst.mkdir(parents=True, exist_ok=True)

# Process all files that look like images
for f in src.iterdir():
    if not f.is_file():
        continue
    try:
        im = Image.open(f)
        # Ensure RGB
        if im.mode != "RGB":
            im = im.convert("RGB")
        out_path = dst / f.name
        angle = random.randint(-10, 10)
        im = im.rotate(angle, expand=True)
        im.save(out_path, optimize=True)
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Skipping {f}: {e}")