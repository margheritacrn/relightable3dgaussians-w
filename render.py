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


def render_lightings(model_path, iteration, model):
    print(f"Rendering illuminations")
    lighting_path = os.path.join(model_path, "rendered_envlights_sh/"+"iteration_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)
    model.render_envlights_sh_all(save_path=lighting_path)


def render_set(model_path, name, iteration, views, model, pipeline, background):
    render_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "gt")
    lighting_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "rendered_envlights")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(lighting_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        view_id = torch.tensor([view.uid], device = 'cuda')
        gt = view.original_image.cuda()
        if name == 'test':
            embedding_gt = model.embeddings_test(view_id)
        else:
            embedding_gt = model.embeddings(view_id)
        envlight_sh = model.envlight_sh_mlp(embedding_gt)
        model.envlight.set_base(envlight_sh)
        render_pkg = render(view, model.gaussians, model.envlight, pipeline, background, debug=True)
        render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

        torch.cuda.synchronize()

        gt = gt[0:3, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
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
        model.render_envlights_sh_all(save_path=lighting_path, test = True, save_sh_coeffs=True)
    else:
        model.render_envlights_sh_all(save_path=lighting_path, test = False, save_sh_coeffs=True)


def render_sets(cfg, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        model = Relightable3DGW(cfg)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(cfg.dataset.model_path, "train", model.load_iteration, model.scene.getTrainCameras(), model, cfg.pipe, background)

        if not skip_test:
             model.optimize_embeddings_test()
             render_set(cfg.dataset.model_path, "test", model.load_iteration, model.scene.getTestCameras(), model, cfg.pipe, background)

        # render_lightings(model.config.dataset.model_path, model.load_iteration, model)


def render_sets_training(model: Relightable3DGW, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        bg_color = [1,1,1] if model.config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(model.config.dataset.model_path, "train", model.load_iteration, model.scene.getTrainCameras(), model, model.config.pipe, background)

        if not skip_test:
             # Optimize embeddings for test set
             model.optimize_embeddings_test()
             render_set(model.config.dataset.model_path, "test", model.load_iteration, model.scene.getTestCameras(), model, model.config.pipe, background)

        render_lightings(model.config.dataset.model_path, "rendered_envlights_sh", model) # only for train sofar


@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print("Rendering " + cfg.dataset.model_path)
    render_sets(cfg, cfg.skip_train, cfg.skip_test)
    # All done
    print("\nRendering complete.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"load_iteration={str(args.iteration)}",
        f"+skip_train={args.skip_train}",
        f"+skip_test={args.skip_test}"
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()
