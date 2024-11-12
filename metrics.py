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
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np


def readImages(renders_dir, gt_dir, masks_path=None):
    renders = []
    gts = []
    image_names = []
    discarded_samples = 0
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt = Image.open(gt_dir / fname)
        gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        if masks_path is not None:
            mask_path = os.path.join(masks_path, fname)
            mask = Image.open(mask_path).convert("L")
            if mask.height != gt.shape[2]:
                resized_mask = mask.resize((mask.width, gt.shape[2]), Image.NEAREST)
                mask = tf.to_tensor(resized_mask).cuda()
            else:
                mask = tf.to_tensor(mask).cuda()
            # Ensure mask is binary
            mask = (mask > 0.5).float()
            if mask.shape[1] != gt.shape[2]:
                discarded_samples += 1
                continue
            mask = mask.expand_as(gt)
            renders.append(gt*mask)
            gts.append(render*mask)
        else:
            renders.append(gt)
            gts.append(render)
        image_names.append(fname)
    print(f"Reading images completed- discarder samples: {discarded_samples}")
    return renders, gts, image_names


def evaluate_half(model_paths, mask_path=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
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
            renders, gts, image_names = readImages(renders_dir, gt_dir, mask_path)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    _,C,H,W = renders[idx].shape
                    ssims.append(ssim(renders[idx][:,:,:,W//2:], gts[idx][:,:,:,W//2:]))
                    psnrs.append(psnr(renders[idx][:,:,:,W//2:], gts[idx][:,:,:,W//2:]))
                    lpipss.append(lpips(renders[idx][:,:,:,W//2:], gts[idx][:,:,:,W//2:], net_type="vgg"))
                    
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
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
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--mask_path', type=str)
    args = parser.parse_args()
    evaluate_half(args.model_paths, args.mask_path)
