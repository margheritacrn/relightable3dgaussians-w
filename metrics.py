#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, img2mse, mse2psnr
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from functools import reduce
import torchvision
from os import makedirs


def readImages(renders_dir, gt_dir, occluders_masks_path=None, sky_masks_path=None):
    renders = []
    gts = []
    image_names = []
    resized_masks = 0
    masks = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt = Image.open(gt_dir / fname)
        gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

        if sky_masks_path is not None:
            sky_mask_path = os.path.join(sky_masks_path, fname[:-4] + "_mask.png")
            sky_mask = Image.open(sky_mask_path).convert("L")
            sky_mask = tf.to_tensor(sky_mask).cuda()
        else:
            sky_mask = None
        if occluders_masks_path is not None:
            occluders_mask_path = os.path.join(occluders_masks_path, fname[:-4]+".png")
            occluders_mask = Image.open(occluders_mask_path).convert("L")
            if occluders_mask.height != gt.shape[2]:
                resized_mask = occluders_mask.resize((occluders_mask.width, gt.shape[2]), Image.NEAREST)
                occluders_mask = tf.to_tensor(resized_mask).cuda()
                occluders_mask = (occluders_mask > 0.5).float()
                resized_masks += 1
            else:
                occluders_mask = tf.to_tensor(occluders_mask).cuda()
            occluders_mask = occluders_mask.expand_as(gt)
        else:
            occluders_mask = None

        if sky_mask is not None and occluders_mask is not None:
            mask_final = torch.logical_or(sky_mask, occluders_mask).int()
        elif sky_mask is not None:
            mask_final = sky_mask
        elif occluders_mask is not None:
            mask_final = occluders_mask
        else:
            mask_final = None
  
        masks.append(mask_final)
        gts.append(gt)
        renders.append(render)
        image_names.append(fname)

    print(f"Reading images completed- resized masks: {resized_masks}")
    return renders, gts, image_names, masks


def evaluate_full(model_path, test_dir = None, occluders_masks_path=None, sky_masks_path=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_path:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}
        
        if test_dir is None:
            test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names, masks = readImages(renders_dir, gt_dir, occluders_masks_path, sky_masks_path)

            ssims = []
            mses = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    _,C,H,W = renders[idx].shape
                    ssim_ = ssim(renders[idx], gts[idx], mask=masks[idx])
                    mse = img2mse(renders[idx], gts[idx], mask=masks[idx])
                    psnr = mse2psnr(mse)
                    ssims.append(ssim_)
                    mses.append(mse)
                    psnrs.append(psnr)
                    lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  MSE : {:>12.7f}".format(torch.tensor(mses).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "MSE": torch.tensor(mses).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "MSE": {name: mse for mse, name in zip(torch.tensor(mses).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results_full.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_full.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


def evaluate_half(model_path, occluders_masks_path=None, sky_masks_path=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_path:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names, masks = readImages(renders_dir, gt_dir, occluders_masks_path, sky_masks_path)

            ssims = []
            mses = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    _,C,H,W = renders[idx].shape
                    ssim_ = ssim(renders[idx][:,:,:,W//2:], gts[idx][:,:,:,W//2:], mask = masks[idx][:,:,:,W//2:])
                    mse = img2mse(renders[idx][:,:,:,W//2:], gts[idx][:,:,:,W//2:], mask = masks[idx][:,:,:,W//2:])
                    psnr = mse2psnr(mse)
                    lpips_ = lpips(renders[idx][:,:,:,W//2:], gts[idx][:,:,:,W//2:], net_type="vgg")
                    ssims.append(ssim_)
                    mses.append(mse)
                    psnrs.append(psnr)
                    lpipss.append(lpips_)
                    
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  MSE : {:>12.7f}".format(torch.tensor(mses).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "MSE": torch.tensor(mses).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "MSE": {name: mse for mse, name in zip(torch.tensor(mses).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results_half.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_half.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_path', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--masks_path', type=str)
    parser.add_argument('--sky_masks_path', type=str)
    args = parser.parse_args()
    evaluate_half(args.model_path, args.masks_path, args.sky_masks_path)
