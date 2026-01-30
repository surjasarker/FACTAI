from pathlib import Path
from PIL import Image
import argparse


def main(input_dir, output_dir):
    src = Path(input_dir)
    dst = Path(output_dir)

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue

        try:
            im = Image.open(f)
            if im.mode != "RGB":
                im = im.convert("RGB")

            out_path = dst / (f.stem + ".jpg")
            im.save(out_path, "JPEG", quality=75, optimize=True)
            print(f"Saved {out_path}")

        except Exception as e:
            print(f"Skipping {f}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to JPEG 75")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input image directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output image directory"
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
