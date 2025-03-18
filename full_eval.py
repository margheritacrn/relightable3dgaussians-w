# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser

nerfosr_scenes = ["lk2", "lwp", "st"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument('--nerfosr', "-osr", required=True, type=str)
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--lambda_sky_gauss", type=float, default=0.05)
parser.add_argument("--reg_normal_from_iter", type=int, default=15_000)
parser.add_argument("--reg_sky_gauss_depth_from_iter", type=int, default=0)
parser.add_argument("--lambda_envlight", type=float, default=100)
parser.add_argument("--init_embeddings", action="store_true")
parser.add_argument("--init_sh_mlp", action="store_true")
args, _ = parser.parse_known_args()

if not args.skip_training:
    common_args = " --quiet --save_iterations 15_000 20_000 30_000 40_000 --test_iterations 3_000 5_000 10_000 15_000 20_000 30_000 40_000 "
    optimizer_args = f" --lambda_sky_gauss {str(args.lambda_sky_gauss)} --lambda_envlight {str(args.lambda_envlight)} --reg_sky_gauss_depth_from_iter {str(args.reg_sky_gauss_depth_from_iter)} --reg_normal_from_iter {str(args.reg_normal_from_iter)} "
    if args.init_embeddings:
        optimizer_args += " --init_embeddings "
    if args.init_sh_mlp:
        optimizer_args += " --init_sh_mlp "
    for scene in nerfosr_scenes:
        source = args.nerfosr + "/" + scene
        os.system("python train.py --source_path " + source + " --model_path " + args.output_path + "/" + scene + optimizer_args + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in nerfosr_scenes:
        all_sources.append(args.nerfosr + "/" + scene)

    common_args = " --quiet "
    for scene, source in zip(nerfosr_scenes, all_sources):
        # os.system("python render.py --iteration 30000 --source_path " + source + " --model_path " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 40_000 --render_with_gt_envmaps --source_path " + source + " --model_path " + args.output_path + "/" + scene + common_args)


# Eval with GT envmaps
for scene in nerfosr_scenes:
    scene_path = args.output_path + "/" + scene
    source = args.nerfosr + "/" + scene
    test_config_path = "./example_test_configs/" + scene
    iterations = ["40_000"]
    for iter in iterations:
        os.system("python eval_with_gt_envmaps.py" + " --model_path " + scene_path + " --source_path " + source + " --test_config " + test_config_path + " --iteration " + iter)