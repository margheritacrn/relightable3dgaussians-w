"The script evaluates the method"
import cv2
import numpy as np
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
from pathlib import Path
from metrics import evaluate_full


def get_gt_envmap_sh(eval_fname, source_path):
    if "IMG" in eval_fname:
        lighting_cond = eval_fname[:eval_fname.find("_IMG")]
    else:
        lighting_cond = eval_fname[:eval_fname.find("_DSC")]
    envmap_path = os.path.join(source_path, "test/ENV_MAP_CC/"+lighting_cond)
    envmap_sh_fname = list(Path(envmap_path).glob("*.npy"))
    assert len(envmap_sh_fname) == 1, "the directory should contain only one envmap file"
    gt_envmap_sh = np.load(envmap_sh_fname[0])
    gt_envmap_sh = torch.tensor(gt_envmap_sh, dtype=torch.float32, device="cuda")
    return gt_envmap_sh


def render_with_gt_envmaps(cfg):
    with torch.no_grad():
        model = Relightable3DGW(cfg)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = {"train": model.scene.getTrainCameras(), "test":model.scene.getTestCameras()}
        eval_nerfosr_fnames = os.listdir(os.path.join(cfg.dataset.source_path, "test/mask"))
        eval_nerfosr_fnames = [fname.split('.')[0] for fname in eval_nerfosr_fnames]
        eval_nerfosr_cameras = [test_cam for test_cam in views["test"] if test_cam.image_name in eval_nerfosr_fnames]
        render_paths = {"train": os.path.join(cfg.dataset.model_path, "eval_nerfosr", "train", "iteration_{}".format(model.load_iteration), "renders"),
                        "test": os.path.join(cfg.dataset.model_path, "eval_nerfosr", "test", "iteration_{}".format(model.load_iteration), "renders")}
        gts_paths = {"train":os.path.join(cfg.dataset.model_path, "eval_nerfosr", "train", "iteration_{}".format(model.load_iteration), "gt"),
                     "test": os.path.join(cfg.dataset.model_path, "eval_nerfosr", "test", "iteration_{}".format(model.load_iteration), "gt")}

        for set in ["test"]:
            print(f"Rendering {set}")
            makedirs(render_paths[set], exist_ok=True)
            makedirs(gts_paths[set], exist_ok=True)

            for view in tqdm(views[set], desc="Rendering progress"):# in eval_nerfosr_cameras/test_cameras:
                if set == "train":
                    rint = torch.randint(0, 5, (1,))
                    gt_envmap_sh = get_gt_envmap_sh(eval_nerfosr_fnames[rint], cfg.dataset.source_path)
                else:
                    gt_envmap_sh = get_gt_envmap_sh(view.image_name, cfg.dataset.source_path)
                model.envlight.set_base(gt_envmap_sh)
                gt = view.original_image.cuda()

                render_pkg = render(view, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
                render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

                torch.cuda.synchronize()

                gt = gt[0:3, :, :]
                torchvision.utils.save_image(render_pkg["render"], os.path.join(render_paths[set], view.image_name + ".png"))
                torchvision.utils.save_image(gt, os.path.join(gts_paths[set], view.image_name + ".png"))


@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print("Rendering with GT light" + cfg.dataset.model_path)
    render_with_gt_envmaps(cfg)
    print("Masked evaluation")
    # evaluate_full(model_path=[cfg.dataset.model_path],
                  # test_dir=Path(os.path.join(cfg.dataset.model_path, "eval_nerfosr")),
                  # masks_path=os.path.join(cfg.dataset.source_path, "test/mask"))
    # All done
    print("\nEnd")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"load_iteration={str(args.iteration)}"
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()