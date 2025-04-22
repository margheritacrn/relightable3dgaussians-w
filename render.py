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

import torch
from scene import Scene
import os
import sys
from tqdm import tqdm
from os import makedirs
import torch.nn.functional as F
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import apply_depth_colormap
from utils.general_utils import get_minimum_axis
from scene.NVDIFFREC.util import save_image_raw
import numpy as np
from omegaconf import DictConfig
from scene.relit3DGW_model import Relightable3DGW
import hydra
from eval_with_gt_envmaps import process_environment_map_image
import glob
import spaudiopy


def render_test_with_gt_envmaps(source_path,model_path, iteration, views, model, pipeline, background, sky_sh_degree, specular):
    render_path = os.path.join(model_path, "test", "iteration_{}".format(iteration), "renders_with_gt_envmaps")
    gt_path = os.path.join(model_path, "test", "iteration_{}".format(iteration), "gt")


    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    
    # Envmaps params
    scale = 10
    if "st" in source_path:
        scale = 30
    threshold = 0.99

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        view_id = torch.tensor([view.uid], device = 'cuda')
        if "_DSC" in view.image_name:
                lighting_condition = view.image_name.split("_DSC")[0]
        else:
                lighting_condition = view.image_name.split("_IMG")[0]
        envmap_folder_path = os.path.join(source_path, "test", "ENV_MAP_CC", lighting_condition)
        envmap_img_path =  glob.glob(os.path.join(envmap_folder_path, '*.jpg'))
        if len(envmap_img_path) == 0:
            continue
        envmap_img_path = [fname for fname in envmap_img_path if "SH" not in fname and "rotated" not in fname][0]
        envlight_sh = process_environment_map_image(envmap_img_path, scale, threshold)
        # rotate around x axis
        envlight_sh = spaudiopy.sph.rotate_sh(envlight_sh.T, 0, -np.pi/2, 0, 'real')
        envlight_sh = torch.tensor(envlight_sh.T, dtype=torch.float32, device="cuda")
        gt = view.original_image.cuda()
        model.envlight.set_base(envlight_sh)
        sky_sh = torch.zeros((9,3), dtype=torch.float32, device="cuda")

        render_pkg = render(view, model.gaussians, model.envlight, sky_sh, sky_sh_degree, pipeline, background, debug=True, fix_sky=True, specular=specular)
        render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)


        for k in render_pkg.keys():
            if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm" or k == "normal_ref" or k == "alpha":
                continue
            save_path = os.path.join(model_path, "test", "iteration_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "diffuse_color" or k=="specular_color":
                render_pkg[k] = torch.clamp(render_pkg[k], 0.0, 1.0)
            if k == "albedo":
                render_pkg[k] = torch.clamp(render_pkg[k], 0.0, 1.0)
            if k == "alpha":
                render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][...,None], min=0., max=1.).permute(2,0,1)
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
            elif "normal" in k:
                render_pkg[k] = 0.5 + (0.5*render_pkg[k])
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, view.image_name + ".png"))

        torch.cuda.synchronize()

        gt = gt[0:3, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, view.image_name + ".png"))


def render_set(model_path, name, iteration, views, model, pipeline, background, sky_sh_degree, fix_sky, specular):
    render_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "renders")
    gt_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "gts")
    lighting_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "rendered_envlights")
    sky_map_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "rendered_sky_maps")

    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    makedirs(lighting_path, exist_ok=True)
    makedirs(sky_map_path, exist_ok=True)


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        view_id = torch.tensor([view.uid], device = 'cuda')
        gt = view.original_image.cuda()
        if name == 'test':
            embedding_gt = model.embeddings_test(view_id)
        else:
            embedding_gt = model.embeddings(view_id)
        envlight_sh, sky_sh = model.mlp(embedding_gt)
        model.envlight.set_base(envlight_sh)

        render_pkg = render(view, model.gaussians, model.envlight, sky_sh, sky_sh_degree, pipeline, background, debug=True, fix_sky=fix_sky, specular=specular)
        render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

        torch.cuda.synchronize()

        gt = gt[0:3, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, view.image_name + ".png"))
        for k in render_pkg.keys():
            if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm" or k == "normal_ref" or k == "alpha":
                continue
            save_path = os.path.join(model_path, name, "iteration_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "diffuse_color" or k=="specular_color":
                render_pkg[k] = torch.clamp(render_pkg[k], 0.0, 1.0)
            if k == "albedo":
                render_pkg[k] = torch.clamp(render_pkg[k], 0.0, 1.0)
            if k == "alpha":
                render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][...,None], min=0., max=1.).permute(2,0,1)
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
            elif "normal" in k:
                render_pkg[k] = 0.5 + (0.5*render_pkg[k])
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, view.image_name + ".png"))

    print(f"{name}- rendering illuminations")
    if name == "test":
        model.render_envlights_sh_all(save_path=lighting_path, eval = True, save_sh_coeffs=True)
    else:
        model.render_envlights_sh_all(save_path=lighting_path, eval = False, save_sh_coeffs=True)
        model.render_sky_sh_all(save_path=sky_map_path, eval = False, save_sh_coeffs=True)


def render_sets(cfg, skip_train : bool, skip_test : bool, render_with_gt_envmaps: bool=False):
    with torch.no_grad():
        model = Relightable3DGW(cfg)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        fix_sky = cfg.fix_sky
        specular = cfg.specular

        if not skip_train:
             render_set(cfg.dataset.model_path, "train", model.load_iteration, model.scene.getTrainCameras(), model, cfg.pipe, background, cfg.sky_sh_degree, fix_sky, specular)

        if not skip_test:
            if not render_with_gt_envmaps:
                # NOTE: to be updated
                model.optimize_embeddings_test()
                render_set(cfg.dataset.model_path, "test", model.load_iteration, model.scene.getTestCameras(), model, cfg.pipe, background, cfg.sky_sh_degree, fix_sky, specular)
            else:
                render_test_with_gt_envmaps(cfg.dataset.source_path,cfg.dataset.model_path,model.load_iteration, model.scene.getTestCameras(), model, cfg.pipe, background, cfg.sky_sh_degree, specular)


@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    cfg.dataset.eval = True
    render_sets(cfg, cfg.skip_train, cfg.skip_test, cfg.render_with_gt_envmaps)
    # All done
    print("\nRendering complete.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_with_gt_envmaps", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"load_iteration={str(args.iteration)}",
        f"+skip_train={args.skip_train}",
        f"+skip_test={args.skip_test}",
        f"+render_with_gt_envmaps={args.render_with_gt_envmaps}"
    ]

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()
