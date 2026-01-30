import os
from pathlib import Path
import cv2 as cv
import argparse


def main(input_dir, output_dir):
    src = Path(input_dir)
    dst = Path(output_dir)

    dst.mkdir(parents=True, exist_ok=True)

    for f in src.iterdir():
        if not f.is_file():
            continue

        try:
            im = cv.imread(str(f))
            if im is None:
                continue

            out_path = dst / f.name
            im_blur = cv.GaussianBlur(im, (5, 5), 1.5)
            cv.imwrite(str(out_path), im_blur)
            print(f"Saved to {out_path}")

        except Exception as e:
            print(f"{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply blur to images")
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
