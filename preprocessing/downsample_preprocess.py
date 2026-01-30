import os
import cv2
import argparse

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                resized_img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_AREA
                )

                cv2.imwrite(os.path.join(output_folder, filename), resized_img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="resize to 512x512")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to input image folder"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to output image folder"
    )

    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
