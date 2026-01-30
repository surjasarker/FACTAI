import os
import random
import shutil

source_dir = r"celeba-hq/downsampled-male"
dest_dir = r"subsampled_set"

os.makedirs(dest_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

images = [
    f for f in os.listdir(source_dir)
    if f.lower().endswith(image_extensions)
]

if len(images) < 50:
    raise ValueError(f"Not enough images")

sampled_images = random.sample(images, 51)

for img in sampled_images:
    src_path = os.path.join(source_dir, img)
    dst_path = os.path.join(dest_dir, img)
    shutil.copy2(src_path, dst_path)

print("Successfully subsampled.")
