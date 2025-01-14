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
import imageio.v3 as im


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def process_environment_map_exr(envmap_path: str, l_max: int=4):
    envmap = im.imread(envmap_path, plugin='EXR-FI')[:, :, :3]
    coeffs = get_coefficients_from_image(envmap, l_max= l_max)
    return coeffs


def process_environment_map_jpg(img_path, scale_high=10, threshold=0.999):
    
    img = plt.imread(img_path)
    img = torch.from_numpy(img).float() / 255
    img[img > threshold] *= scale_high
    coeffs = get_coefficients_from_image(img.numpy(), 4)
    return coeffs


def save_reconstructed_envmap_sh(envmap_sh, save_name):
    render_sh_envmap = sh_utility.sh_render(envmap_sh, width = 360)
    render_sh_envmap = (render_sh_envmap - render_sh_envmap.min()) / (render_sh_envmap.max() - render_sh_envmap.min()) * 255
    render_sh_envmap = render_sh_envmap.astype(np.uint8)
    render_sh_envmap = Image.fromarray(render_sh_envmap)
    render_sh_envmap.save(save_name)


def render_single_and_relight(cfg, envmap_path=None, N_rots_envmap: int = 100):
    with torch.no_grad():
        model = Relightable3DGW(cfg)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        tmp = cfg.test_image_name.split(".")[0]
        test_camera = [c for c in model.scene.getTestCameras() if c.image_name == tmp][0]

        relits_path = os.path.join(cfg.dataset.model_path, "novel_view_relighting", "iteration_{}".format(model.load_iteration), test_camera.image_name)
        makedirs(relits_path, exist_ok=True)

        # render
        if envmap_path != "":
            relits_ext_envmap_path = os.path.join(relits_path, "external_envmap_" + os.path.basename(envmap_path)[:-4])
            makedirs(relits_ext_envmap_path, exist_ok=True)
            if envmap_path[-4:] == ".exr":
                envmap_sh = process_environment_map_exr(envmap_path)
            elif envmap_path[-4:] == ".jpg":
                envmap_sh = process_environment_map_jpg(envmap_path)
            else:
                envmap_sh = np.load(os.path.join(cfg.dataset.model_path, "train", "iteration_50000", "rendered_envlights", envmap_path))
            
            save_reconstructed_envmap_sh(envmap_sh, os.path.join(relits_ext_envmap_path, "external_reconstructed" + os.path.basename(envmap_path)[:-4] + ".png"))
            envmap_sh_rot_around_x = spaudiopy.sph.rotate_sh(envmap_sh.T, 0, 0,-np.pi/2, 'real')
            envmaps_sh_rot_around_x_torch = torch.tensor(envmap_sh_rot_around_x.T, dtype=torch.float32, device="cuda")
            envmap_sh_torch = torch.tensor(envmap_sh, dtype=torch.float32, device="cuda")
            # render with no rotation around x-axis
            model.envlight.set_base(envmap_sh_torch)
            render_pkg = render(test_camera, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
            render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torchvision.utils.save_image(render_pkg["render"], os.path.join(relits_ext_envmap_path, "relit_external_" + os.path.basename(envmap_path)[:-4] + ".png"))
            # render with rotation around x-axis 
            model.envlight.set_base(envmaps_sh_rot_around_x_torch)
            render_pkg = render(test_camera, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
            render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torchvision.utils.save_image(render_pkg["render"], os.path.join(relits_ext_envmap_path, "relit_external_rotated_x_" + os.path.basename(envmap_path)[:-4] + ".png"))
            # save
            save_reconstructed_envmap_sh(envmap_sh_rot_around_x.T, os.path.join(relits_ext_envmap_path, "external_rotated_x_reconstructed" + os.path.basename(envmap_path)[:-4] + ".png"))
            # render with N rotations of the environment map around the y axis sampled in [-0.3 * np.pi, 1.1*np.pi]:
            sun_angles = [torch.tensor([angle,0, 0]) for angle in torch.linspace(0, 2* np.pi, N_rots_envmap)]
            for num, angle in enumerate(tqdm(sun_angles)):
                # rotate envmap and render
                envmap_sh_rot = spaudiopy.sph.rotate_sh(envmap_sh.T, 0, 0,-np.pi/2, 'real')
                envmap_sh_rot = spaudiopy.sph.rotate_sh(envmap_sh_rot, angle[2], angle[0], angle[1], 'real')

                if num == 0 or num == 50: 
                    save_reconstructed_envmap_sh(envmap_sh_rot.T, os.path.join(relits_ext_envmap_path, f"rotated_{num}_reconstructed" + ".png"))

                envmap_sh_rot_torch = torch.tensor(envmap_sh_rot.T, dtype=torch.float32, device="cuda")
                model.envlight.set_base(envmap_sh_rot_torch)
                render_pkg = render(test_camera, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
                render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
                torchvision.utils.save_image(render_pkg["render"], os.path.join(relits_ext_envmap_path, str(num) + ".png"))

            # generate video
            image_files = [os.path.join(relits_ext_envmap_path ,str(num) + ".png") for num in range(0, N_rots_envmap) ]
            # image_files = [os.path.join(relits_ext_envmap_path ,img) for img in os.listdir(relits_ext_envmap_path) if (img.endswith(".png") and "external" not in img and "reconstructed" not in img)]
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=30)
            clip.write_videofile(os.path.join(relits_ext_envmap_path , 'relit_novel_view_external_envmap_rots.mp4'))



@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print("Rendering " + cfg.test_image_name)
    render_single_and_relight(cfg, cfg.envmap_path)
    print("\nEnd")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--source_path", "-s", type=str)
    parser.add_argument("--model_path", "-m", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test_image_name", default="", type=str)
    parser.add_argument("--envmap_path", default="", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"dataset.eval={args.eval}",
        f"load_iteration={str(args.iteration)}",
        f"+test_image_name={args.test_image_name}",
        f"+envmap_path={args.envmap_path}"
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()


            