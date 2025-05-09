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
import matplotlib.pyplot as plt
import torch
import spaudiopy
from matplotlib import pyplot as plt
from utils.sh_additional_utils import get_coefficients_from_image


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" 


def find_folder(base_path, folder_name):
    for root, dirs, _ in os.walk(base_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None



def scale_saturated_pixels_and_extract_sh(rendered_sh_path, scale_high, lmax=4, threshold=0.999):
    
    rendered_sh = plt.imread(rendered_sh_path)
    rendered_sh = torch.from_numpy(rendered_sh).float() / 255
    rendered_sh[rendered_sh > threshold] *= scale_high
    coeffs = get_coefficients_from_image(rendered_sh.numpy(), lmax)
    return coeffs


def rotate_envmap(envmap: EnvironmentMap, angles:list, return_sh: bool = False, lmax: int = 2, resize_height: int = None, save_jpg_path: str = None):
    """The function rotates the input environement map with a rotation
    of the given angles.
    Args:
        envmap(Environmentmap): environment map object,
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
        envmap_jpg_rot = (envmap_rot_data  - envmap_rot_data.min()) / (envmap_rot_data.max() - envmap_rot_data.min()) * 255
        envmap_jpg_rot = envmap_jpg_rot.astype(np.uint8)
        envmap_jpg_rot = Image.fromarray(envmap_jpg_rot)
        envmap_jpg_rot.save(save_jpg_path)

    if return_sh:
        envmap_rot_sh_coeffs = sh_utility.get_coefficients_from_image(envmap_rot_data, lmax)
        return envmap_rot, envmap_rot_sh_coeffs,
    else:
        return envmap_rot


def process_gt_envmaps(nerfosr_path: str, lmax: int=4, rotate=False):
    for scene in os.listdir(nerfosr_path):
        print(f"Scene: {scene}")
        gtenvmapsdir_path = find_folder(os.path.join(nerfosr_path, scene), "ENV_MAP_CC")
        scale_high = 10
        if gtenvmapsdir_path is None:
            gtenvmapsdir_path = os.path.join(nerfosr_path, scene + "/test/ENV_MAP_CC")
        for lighting_cond in os.listdir(gtenvmapsdir_path):
            lighting_cond_path = os.path.join(gtenvmapsdir_path, lighting_cond)
            for gtenvmap_filename in os.listdir(lighting_cond_path):
                print(f"Processing {gtenvmap_filename}")
                # Rotate
                gtenvmap_jpg_path = os.path.join(lighting_cond_path, gtenvmap_filename)
                gt_envmap_sh = scale_saturated_pixels_and_extract_sh(gtenvmap_jpg_path, scale_high=scale_high, lmax=lmax)
                if rotate: #rotate around x axis
                    gt_envmap_sh = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, 0, 0, -np.pi/2, 'real')
                    np.savetxt(gtenvmap_jpg_path[:-4]+f'rotated_SH{lmax}.txt', gt_envmap_sh.T)
                else:
                    np.savetxt(gtenvmap_jpg_path[:-4]+f'SH{lmax}.txt', gt_envmap_sh.T)
                # Envmap SH reconstruction 
                rendered_sh = sh_utility.sh_render(gt_envmap_sh.T, width = 360)
                rendered_sh = torch.tensor(rendered_sh** (1/ 2.2))
                rendered_sh =  np.array(rendered_sh * 255).clip(0,255).astype(np.uint8)
                rendered_sh = (rendered_sh - rendered_sh.min()) / (rendered_sh.max() - rendered_sh.min())
                if rotate:
                    plt.imsave(gtenvmap_jpg_path[:-4]+f"rotatedSH{lmax}rec.jpg", rendered_sh)
                else:
                    plt.imsave(gtenvmap_jpg_path[:-4]+f"SH{lmax}rec.jpg")


def main(nerfosr_path: str, lmax: int, rotate: bool):
    print("Processing NeRF-OSR GT envmaps")
    process_gt_envmaps(nerfosr_path, lmax, rotate)
    print("\nEnd")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate SH coefficients for GT envmaps- NeRF-OSR dataset")
    parser.add_argument("--nerfosr", "-osr", type=str)
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--rotate", action="store_true")
    args, _ = parser.parse_known_args()
    print(args)
    main(args.nerfosr, args.lmax, args.rotate)
