import cv2
import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
import numpy as np
import sys
import importlib
from skimage.metrics import structural_similarity as ssim_skimage
from utils.loss_utils import mse2psnr, img2mae, img2mse, img2mse_image
from utils.sh_additional_utils import get_coefficients_from_image
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tf
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
import numpy as np
from omegaconf import DictConfig
from scene.relit3DGW_model import Relightable3DGW
import hydra
import matplotlib.pyplot as plt
import spaudiopy
import utils.sh_additional_utils as sh_utility
from PIL import Image
import moviepy.video.io.ImageSequenceClip


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def process_environment_map_exr(envmap_path: str, l_max: int=4):
    envmap =  cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)
    coeffs = get_coefficients_from_image(envmap, l_max= l_max)
    return coeffs


def render_single_and_relight(cfg, trained_illuminations, external_envmap_path=None):
    with torch.no_grad():
        model = Relightable3DGW(cfg)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        tmp = cfg.test_image_name.split(".")[0]
        test_camera = [c for c in model.scene.getTestCameras() if c.image_name == tmp][0]

        relits_path = os.path.join(cfg.dataset.model_path, "novel_view_relighting", "iteration_{}".format(model.load_iteration), test_camera.image_name)
        makedirs(relits_path, exist_ok=True)

        # render
        if external_envmap_path is not None:
            assert external_envmap_path[-4:] == ".exr", ".exr format required"
            envmap_sh = process_environment_map_exr(external_envmap_path)
            envmap_sh_torch = torch.tensor(envmap_sh, dtype=torch.float32, device="cuda")
            model.envlight.set_base(envmap_sh_torch)
            render_pkg = render(test_camera, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
            render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torchvision.utils.save_image(render_pkg["render"], os.path.join(relits_path, "relit_external_" + os.path.basename(external_envmap_path)[:-4] + ".png"))

        if trained_illuminations:
            trained_illuminations_path = os.path.join(cfg.dataset.model_path, "envlights_sh", "iteration_{}".format(model.load_iteration))
            trained_illuminations_sh = [np.load(os.path.join(trained_illuminations_path, envlight_sh)) for envlight_sh in os.listdir(trained_illuminations_path)]
            for num, trained_illumination_sh in enumerate(tqdm(trained_illuminations_sh)):
                trained_illumination_sh_torch = torch.tensor(trained_illumination_sh, dtype=torch.float32, device="cuda")
                model.envlight.set_base(trained_illumination_sh_torch)
                render_pkg = render(test_camera, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
                render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
                torchvision.utils.save_image(render_pkg["render"], os.path.join(relits_path, str(num) + ".png"))

            # generate video
            image_files = [os.path.join(relits_path,img) for img in os.listdir(relits_path) if (img.endswith(".png") and "external" not in img)]
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=5)
            clip.write_videofile(os.path.join(relits_path, 'relit_novel_view.mp4'))


@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print("Rendering " + cfg.test_image_name)
    render_single_and_relight(cfg, cfg.trained_illuminations, cfg.external_envmap_path)
    print("\nEnd")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--source_path", "-s", type=str)
    parser.add_argument("--model_path", "-m", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test_image_name", default="", type=str)
    parser.add_argument("--trained_illuminations", action="store_true")
    parser.add_argument("--external_envmap_path", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"dataset.eval={args.eval}",
        f"load_iteration={str(args.iteration)}",
        f"+test_image_name={args.test_image_name}",
        f"+trained_illuminations={args.trained_illuminations}",
        f"+external_envmap_path={args.external_envmap_path}"
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()


            