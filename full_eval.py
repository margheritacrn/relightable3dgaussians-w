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

nerfosr_scenes = ["lk2", "schloss", "lwp", "st"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument('--nerfosr', "-osr", required=True, type=str)
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--num_sky_points", type=int, default=0)
args, _ = parser.parse_known_args()

if not args.skip_training:
    common_args = " --quiet --eval --save_iterations 30_000 50_000 "
    for scene in nerfosr_scenes:
        source = args.nerfosr + "/" + scene
        os.system("python train.py --source_path " + source + " --model_path " + args.output_path + "/" + scene + " --num_sky_points " + str(args.num_sky_points) + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in nerfosr_scenes:
        all_sources.append(args.nerfosr + "/" + scene)

    common_args = " --quiet "
    for scene, source in zip(nerfosr_scenes, all_sources):
        os.system("python render.py --iteration 30000 --source_path " + source + " --model_path " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 50000 --source_path " + source + " --model_path " + args.output_path + "/" + scene + common_args)

# reconstruction eval
if not args.skip_metrics:
    scenes_string = ""
    for scene in nerfosr_scenes:
        scene_path = args.output_path + "/" + scene
        source = args.nerfosr + "/" + scene
        masks_path = source + "/masks"
        sky_masks_path = source + "/sky_masks" 
        os.system("python metrics.py -m " + scene_path + " --masks_path " + masks_path + " --sky_masks_path " + sky_masks_path)

# eval with gt_envmaps