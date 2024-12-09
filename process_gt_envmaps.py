"""The script generates SH coefficients of the GT environment maps provided in NeRF-OSR dataset. The environment maps are rotated around x-axis of 90 degree"""
import os
import cv2
from envmap import EnvironmentMap, rotation_matrix
from PIL import Image
import utils.sh_additional_utils as utility
import utils.sh_additional_utils as sh_utility
from argparse import ArgumentParser
import numpy as np
import imageio.v3 as im


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" 


def rotate_envmap(envmap: EnvironmentMap, angles:list, return_sh: bool = False, lmax: int = 2, resize_height: int = None, save_jpg_path: str = None):
    """The function rotates the input environement map with a rotation
    of the given angles.
    Args:
        envma(Environmentmap): environment map object,
        angles(list): [azimuth, elevation, roll],
        return_sh (bool): if True returns rotated sh coeffs, else rotated envmap,
        lmax(int): degree of sh coefficients,
        resize_height(int): envmap resized to (resize_height, resize_height // 2),
        save_jpg_path(str): path where rotated envmap jpg is saved
    Returns:
        rotated sh coeffs of degree lmax or envmap.
    """
    if resize_height is not None:
        envmap = envmap.resize(targetSize=resize_height)
    dcm = rotation_matrix(azimuth=angles[0], # z, yaw
                          elevation=angles[1], # y, pitch: - 90 deg
                          roll=angles[2]) # x, roll
    envmap_rot = envmap.rotate(dcm)
    envmap_rot_data = envmap_rot.data.copy()

    if save_jpg_path is not None:
        envmap_jpg_rot = (envmap_rot  - envmap_rot.min()) / (envmap_rot.max() - envmap_rot.min()) * 255
        envmap_jpg_rot = envmap_jpg_rot.astype(np.uint8)
        envmap_jpg_rot = Image.fromarray(envmap_jpg_rot)
        envmap_jpg_rot.save(save_jpg_path)

    if return_sh:
        envmap_rot_sh_coeffs = sh_utility.get_coefficients_from_image(envmap_rot_data, lmax)
        return envmap_rot, envmap_rot_sh_coeffs,
    else:
        return envmap_rot


def get_sh_coeffs_gt_envmaps(nerfosr_path: str, lmax: int=4):
    for scene in os.listdir(nerfosr_path):
        print(f"Scene: {scene}")
        gtenvmapsdir_path = os.path.join(nerfosr_path, scene + "/test/ENV_MAP_CC")
        for lighting_cond in os.listdir(gtenvmapsdir_path):
            lighting_cond_path = os.path.join(gtenvmapsdir_path, lighting_cond)
            for gtenvmap_filename in os.listdir(lighting_cond_path):
                if "rotated" not in gtenvmap_filename and gtenvmap_filename[-4:] == ".jpg": 
                    print(f"Processing {gtenvmap_filename}")
                    # Rotate
                    gtenvmap_jpg_path = os.path.join(lighting_cond_path, gtenvmap_filename)
                    gt_envmap = EnvironmentMap(gtenvmap_jpg_path, 'latlong')
                    # Rotate envmap
                    _, gtenvmap_sh_coeffs = rotate_envmap(gt_envmap, angles=[0,0,-np.pi/2], return_sh=True, lmax=4, resize_height=180,
                                                        save_jpg_path=gtenvmap_jpg_path[:-4]+"_rotated.jpg") # angles z,y,x
                    np.save(gtenvmap_jpg_path[:-4]+'rotatedSH.npy', gtenvmap_sh_coeffs)
                    # SH reconstruction 
                    rendered_sh = sh_utility.sh_render(gtenvmap_sh_coeffs, width = 360)
                    rendered_sh = (rendered_sh - rendered_sh.min()) / (rendered_sh.max() - rendered_sh.min()) * 255
                    rendered_sh = rendered_sh.astype(np.uint8)
                    reconstructed_envmap = Image.fromarray(rendered_sh)
                    reconstructed_envmap.save(gtenvmap_jpg_path[:-4]+"rotatedSHrec.jpg")


def main(nerfosr_path: str, lmax: int):
    print("Processing NeRF-OSR GT envmaps")
    get_sh_coeffs_gt_envmaps(nerfosr_path, lmax)
    print("\nEnd")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate SH coefficients for GT envmaps- NeRF-OSR dataset")
    parser.add_argument("--nerfosr", "-osr", type=str)
    parser.add_argument("--lmax", type=int, default=4)
    args, _ = parser.parse_known_args()
    main(args.nerfosr, args.lmax)
