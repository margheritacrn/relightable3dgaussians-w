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

nerfosr_scenes = ["schloss", "lwp"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

nerfosr_scenes = []


if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--nerfosr', "-osr", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --save_iterations 30_000 50_000 "
    for scene in nerfosr_scenes:
        source = args.nerfosr + "/" + scene
        os.system("python train.py --source_path " + source + " --model_path " + args.output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in nerfosr_scenes:
        all_sources.append(args.nerfosr + "/" + scene)

    common_args = " --quiet --eval "
    for scene, source in zip(nerfosr_scenes, all_sources):
        os.system("python render.py --iteration 30000 --source_path " + source + " --model_path " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 50000 --source_path " + source + " --model_path " + args.output_path + "/" + scene + common_args)