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
#TODO: handle loading model at input iteration


def render_lightings(model_path, name, iteration, model):
    lighting_path = os.path.join(model_path, name, "iteration_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)
    model.render_envlights_sh_all(path = lighting_path)


def render_set(model_path, name, iteration, views, model, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        view_id = torch.tensor([view.uid], device = 'cuda')
        gt = view.original_image.cuda()
        embedding_gt = model.embeddings(view_id)
        envlight_sh = model.envlight_sh_mlp(embedding_gt)
        model.envlight.set_base(envlight_sh)
        render_pkg = render(view, model.gaussians, model.envlight, pipeline, background, debug=True)

        torch.cuda.synchronize()

        gt = gt[0:3, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        for k in render_pkg.keys():
            if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
                continue
            save_path = os.path.join(model_path, name, "iteration_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "alpha":
                render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][...,None], min=0., max=1.).permute(2,0,1)
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
            elif "normal" in k:
                render_pkg[k] = 0.5 + (0.5*render_pkg[k])
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(cfg, skip_train : bool, skip_test : bool, iteration: int):
    with torch.no_grad():
        model = Relightable3DGW(cfg, load_iteration = iteration)
        #gaussians = GaussianModel(cfg.dataset.sh_degree)
        #scene = Scene(cfg.dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(cfg.dataset.model_path, "train", model.scene.loaded_iter, model.scene.getTrainCameras(), model, cfg.pipe, background)

        if not skip_test:
             render_set(cfg.dataset.model_path, "test", model.scene.loaded_iter, model.scene.getTestCameras(), model, cfg.pipe, background)

        render_lightings(cfg.dataset.model_path, "rendered_envlights_sh", model)


@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print("Rendering " + cfg.cfg.dataset.model_path)
    render_sets(cfg, cfg.skip_train, cfg.skip_test, cfg.iteration)
    # All done
    print("\nRendering complete.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"+iteration={str(args.iteration)}",
        f"+skip_train={args.skip_train}",
        f"+skip_train={args.skip_test}"
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()
    # All done
    print("\nRendering complete.")
