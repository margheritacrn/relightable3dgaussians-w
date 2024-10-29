"""
The script converts rgb images in jpg/png format to exr files
"""
from argparse import ArgumentParser, Namespace
import os, sys
import cv2
import numpy as np


def to_exr(data_path: str, resize_dim: list):
    output_path = data_path[:-4] + "/exr"
    images = images = os.listdir(data_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for image_fn in images:
        # load the sRGB image
        image = cv2.imread(data_path + "/" + image_fn)
        # check if the image was loaded successfully
        if image is None:
            raise ValueError("Image not found or could not be loaded.")
        
        if resize_dim is not None:
            image = cv2.resize(image, tuple(resize_dim), cv2.INTER_AREA)

        # convert image from uint8 to float32
        image_float = image.astype(np.float32) / 255.0

        # save image in .EXR format
        cv2.imwrite(output_path + "/"+ image_fn[:-4] + ".exr", image_float)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="args for exr conversion")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--resize_dim', nargs='+', type=int)
    args = parser.parse_args(sys.argv[1:])
    assert args.data_path[-4:] == "/rgb", "data must be stored in a folder named rgb"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    to_exr(args.data_path, args.resize_dim)
    # All done
    print("\nConvrsion to .exr files complete.")