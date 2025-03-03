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
import glob
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


TINY_NUMBER = 1e-6


def process_environment_map_image(img_path, scale_high, threshold):
    
    img = plt.imread(img_path)
    img = torch.from_numpy(img).float() / 255
    img[img > threshold] *= scale_high
    coeffs = get_coefficients_from_image(img.numpy(), 4)
    return coeffs


def render_and_evaluate_tuning_scenes(cfg, save_renders=False):
    with torch.no_grad():
        model = Relightable3DGW(cfg)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        test_cameras = model.scene.getTestCameras()
        test_envmaps_path = os.path.join(cfg.dataset.source_path, "test/ENV_MAP_CC") 
        # Environment maps processing
        scale = 10
        threshold = 0.999
        sun_angle_range = [0, 2*np.pi]
        init_rot_x = -np.pi/2
        init_rot_y = 0
        init_rot_z = 0
        psnrs = []

        if save_renders:
            renders_path = os.path.join(cfg.dataset.model_path, "eval_gt_envmap", "test", "iteration_{}".format(model.load_iteration), "renders")
            gts_path = os.path.join(cfg.dataset.model_path, "eval_gt_envmap", "test", "iteration_{}".format(model.load_iteration), "gt")
            makedirs(renders_path, exist_ok=True)
            makedirs(gts_path, exist_ok=True)
        
        for view in tqdm(test_cameras):
            if "_DSC" in view.image_name:
                lighting_condition = view.image_name.split("_DSC")[0]
            else:
                lighting_condition = view.image_name.split("_IMG")[0]
            
            # Get processed envmap
            envmap_path = os.path.join(test_envmaps_path, lighting_condition)
            if len(glob.glob(os.path.join(envmap_path, "*.jpg"))) == 0:
                continue
            envmap_img_path = glob.glob(os.path.join(envmap_path, "*.jpg"))[0]
            gt_envmap_sh = process_environment_map_image(envmap_img_path, scale, threshold) # (25,3)
            # Get gt image
            gt_image = view.original_image.cuda()
            # Get mask
            occluders_mask_path = os.path.join(cfg.dataset.source_path, "masks", view.image_name + ".png")
            occluders_mask = cv2.imread(occluders_mask_path, cv2.IMREAD_GRAYSCALE)
            occluders_mask = cv2.resize(occluders_mask, (gt_image.shape[2], gt_image.shape[1]))
            sky_mask_path = os.path.join(cfg.dataset.source_path, "sky_masks", view.image_name + "_mask.png")
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)
            sky_mask = cv2.resize(sky_mask, (gt_image.shape[2], gt_image.shape[1]))
            mask = cv2.bitwise_and((sky_mask).astype(np.uint8), (occluders_mask).astype(np.uint8))
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = torch.from_numpy(mask//255).cuda()

            best_psnr = 0
            best_angle = None
            all_psnrs = []

            n = 101
            sun_angles_prepare_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], n)  
            sun_angles = [torch.tensor([angle,0, 0]) for angle in sun_angles_prepare_list] #rotate only around y

            for angle in tqdm(sun_angles):
                # rotate envmap and render
                gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, init_rot_z, init_rot_y, init_rot_x, 'real')
                gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh_rot, angle[2], angle[0], angle[1], 'real')
                
                gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot.T, dtype=torch.float32, device="cuda")
                model.envlight.set_base(gt_envmap_sh_rot)
                model.skylight.set_base(torch.zeros((9,3), dtype=torch.float32, device="cuda"))
                render_pkg = render(view, model.gaussians, model.envlight, model.skylight, cfg.pipe, background, debug=False, fix_sky=True)
                render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

                # compute metrics
                current_psnr = mse2psnr(img2mse(render_pkg["render"], gt_image, mask=mask))
                all_psnrs.append(current_psnr.cpu())

                if current_psnr > best_psnr:
                    best_angle = angle
                    best_psnr = current_psnr
            
            # Render with gtenvmap rotated according to best angles        
            gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, init_rot_z, init_rot_y, init_rot_x, 'real')
            gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh_rot, best_angle[2], best_angle[0], best_angle[1], 'real')
            gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot, dtype=torch.float32, device="cuda")
            model.envlight.set_base(gt_envmap_sh_rot.T)
            model.skylight.set_base(torch.zeros((9,3), dtype=torch.float32, device="cuda"))
            render_pkg = render(view, model.gaussians, model.envlight, model.skylight, cfg.pipe, background, debug=False, fix_sky=True)
            rendering_masked = torch.clamp(render_pkg["render"]*mask, 0.0, 1.0)
            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            psnrs.append(mse2psnr(img2mse(rendering_masked, gt_image, mask=mask)))

            if save_renders:
                torch.cuda.synchronize()
                gt_image = gt_image[0:3, :, :]
                torchvision.utils.save_image(rendering_masked, os.path.join(renders_path, view.image_name + "_masked.png"))
                torchvision.utils.save_image(rendering, os.path.join(renders_path, view.image_name + ".png"))
                torchvision.utils.save_image(gt_image*mask, os.path.join(gts_path, view.image_name + ".png"))
            
        return torch.tensor(psnrs).mean()


def render_and_evaluate_test_scenes(cfg, eval_all=False):
    with torch.no_grad():
        model = Relightable3DGW(cfg)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Read test config
        sys.path.append(cfg.test_config)
        config_test = importlib.import_module("test_config").config
        config_test_names = [key.split(".")[0] for key in config_test.keys()]

        if eval_all:
            test_cameras = model.scene.getTestCameras()
        else:
            test_cameras = [c for c in model.scene.getTestCameras() if c.image_name in config_test_names]

        out_dir_name = "eval_gt_envmap"
        if eval_all:
            out_dir_name = "eval_gt_envmap_all"
        renders_path = os.path.join(cfg.dataset.model_path, out_dir_name, "test", "iteration_{}".format(model.load_iteration), "renders")
        gts_path = os.path.join(cfg.dataset.model_path, out_dir_name, "test", "iteration_{}".format(model.load_iteration), "gt")
        makedirs(renders_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        ssims, psnrs, mses, maes = [], [], [], []
        img_names, used_angles =[], []

        for view in tqdm(test_cameras):
            print(view.image_name)

            if eval_all:
                if "_DSC" in view.image_name:
                    lighting_condition = view.image_name.split("_DSC")[0]
                else:
                    lighting_condition = view.image_name.split("_IMG")[0]
                source_img = [config_test_name for config_test_name in config_test_names if lighting_condition in config_test_name]
                image_config = config_test[source_img[0]]
            else:
                image_config = config_test[view.image_name]
            mask_path = image_config["mask_path"]
            envmap_img_path = image_config["env_map_path"]
            init_rot_x = image_config["initial_env_map_rotation"]["x"]
            init_rot_y = image_config["initial_env_map_rotation"]["y"]
            init_rot_z = image_config["initial_env_map_rotation"]["z"]
            threshold = image_config["env_map_scaling"]["threshold"]
            scale = image_config["env_map_scaling"]["scale"]
            sun_angle_range = image_config["sun_angles"]

            gt_envmap_sh = process_environment_map_image(envmap_img_path, scale, threshold) # (25,3)

            # Get gt
            gt_image = view.original_image.cuda()

            # Get eval mask
            if eval_all and view.image_name not in config_test.keys():
                mask_path = os.path.join(cfg.dataset.source_path, "test", "cityscapes_mask", "binary_masks", view.image_name + ".png")
                """
                occluders_mask_path = os.path.join(cfg.dataset.source_path, "masks", view.image_name + ".png")
                occluders_mask = cv2.imread(occluders_mask_path, cv2.IMREAD_GRAYSCALE)
                occluders_mask = cv2.resize(occluders_mask, (gt_image.shape[2], gt_image.shape[1]))
                sky_mask_path = os.path.join(cfg.dataset.source_path, "sky_masks", view.image_name + "_mask.png")
                sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)
                sky_mask = cv2.resize(sky_mask, (gt_image.shape[2], gt_image.shape[1]))
                mask = cv2.bitwise_and((sky_mask).astype(np.uint8), (occluders_mask).astype(np.uint8))
                """
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (gt_image.shape[2], gt_image.shape[1]))
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = torch.from_numpy(mask//255).cuda()

            best_psnr = 0
            best_angle = None
            all_psnrs = []

            n = 51
            sun_angles_prepare_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], n)  
            sun_angles = [torch.tensor([angle,0, 0]) for angle in sun_angles_prepare_list] #rotate only around y

            for angle in tqdm(sun_angles):
                # rotate envmap and render
                gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, init_rot_z, init_rot_y, init_rot_x, 'real')
                gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh_rot, angle[2], angle[0], angle[1], 'real')
                
                gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot.T, dtype=torch.float32, device="cuda")
                model.envlight.set_base(gt_envmap_sh_rot)
                model.skylight.set_base(torch.zeros((9,3), dtype=torch.float32, device="cuda"))
                render_pkg = render(view, model.gaussians, model.envlight, model.skylight, cfg.pipe, background, debug=False, fix_sky=True)
                render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

                # compute metrics
                current_psnr = mse2psnr(img2mse(render_pkg["render"], gt_image, mask=mask))
                all_psnrs.append(current_psnr.cpu())

                if current_psnr > best_psnr:
                    best_angle = angle
                    best_psnr = current_psnr

            print(f"Lowest PSNR: {np.array(all_psnrs).min()}\n")
            print(f"Best angle {best_angle}")
            print(f"Best PSNR: {best_psnr}\n")

            # Render with gtenvmap rotated according to best angles        
            gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, init_rot_z, init_rot_y, init_rot_x, 'real')
            gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh_rot, best_angle[2], best_angle[0], best_angle[1], 'real')

            # Save best_envmap:
            np.save(os.path.join(renders_path, "best_envmap" + view.image_name+".npy"), gt_envmap_sh_rot)

            render_best_angle_envmap = sh_utility.sh_render(gt_envmap_sh_rot.T, width = 360)
            render_best_angle_envmap = (render_best_angle_envmap - render_best_angle_envmap.min()) / (render_best_angle_envmap.max() - render_best_angle_envmap.min()) * 255
            render_best_angle_envmap = render_best_angle_envmap.astype(np.uint8)
            render_best_angle_envmap = Image.fromarray(render_best_angle_envmap)
            render_best_angle_envmap.save(os.path.join(renders_path, "best_angle_rot_envmap"+view.image_name+".jpg"))
                
            gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot, dtype=torch.float32, device="cuda")
            model.envlight.set_base(gt_envmap_sh_rot.T)
            model.skylight.set_base(torch.zeros((9,3), dtype=torch.float32, device="cuda"))

            render_pkg = render(view, model.gaussians, model.envlight, model.skylight, cfg.pipe, background, debug=False, fix_sky=True)
            rendering_masked = torch.clamp(render_pkg["render"]*mask, 0.0, 1.0)
            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torch.cuda.synchronize()
            gt_image = gt_image[0:3, :, :]
            torchvision.utils.save_image(rendering_masked, os.path.join(renders_path, view.image_name + "_masked.png"))
            torchvision.utils.save_image(rendering, os.path.join(renders_path, view.image_name + ".png"))
            torchvision.utils.save_image(gt_image*mask, os.path.join(gts_path, view.image_name + ".png"))
            
            used_angles.append(best_angle)
            img_names.append(view.image_name)
            
            # Compute metrics
            psnrs.append(mse2psnr(img2mse(rendering, gt_image, mask=mask)))
            maes.append(img2mae(rendering, gt_image, mask=mask))
            mses.append(img2mse(rendering, gt_image, mask=mask))

            rendered_np= rendering.cpu().detach().numpy().transpose(1, 2, 0)
            gt_image_np = gt_image.cpu().detach().numpy().transpose(1, 2, 0)
            
            _, full = ssim_skimage(rendered_np, gt_image_np, win_size=5, channel_axis=2, full=True, data_range=1.0)
            mssim_over_mask = (torch.tensor(full).cuda()*mask.unsqueeze(-1)).sum() / (3*mask.sum())
            ssims.append(mssim_over_mask)
    psnrs_dict = {img_name: psnr.item() for img_name, psnr in zip(img_names, psnrs)}
    mses_dict = {img_name: mse.item() for img_name, mse in zip(img_names, mses)}
    maes_dict = {img_name: mae.item() for img_name, mae in zip(img_names, maes)}
    ssims_dict = {img_name: ssim.item() for img_name, ssim in zip(img_names, ssims)}
    # Print metrics
    print("  PSNR: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  MSE: {:>12.7f}".format(torch.tensor(mses).mean(), ".5"))
    print("  MAE: {:>12.7f}".format(torch.tensor(maes).mean(), ".5"))
    print("  SSIM: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    # Save metrics 
    with open(os.path.join(renders_path, "metrics.txt"), 'w') as f:
        print("  PSNR: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"), file=f)
        print("  MSE: {:>12.7f}".format(torch.tensor(mses).mean(), ".5"), file=f)
        print("  MAE: {:>12.7f}".format(torch.tensor(maes).mean(), ".5"), file=f)
        print("  SSIM: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"), file=f)
        print(f"  best PSNRs: {psnrs_dict}", file=f)
        print(f"  best MSEs: {mses_dict}", file=f)
        print(f"  best MAEs: {maes_dict}", file=f)
        print(f"  best SSIMs: {ssims_dict}", file=f)

@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print("Rendering with GT illumination" + cfg.dataset.model_path)
    cfg.dataset.eval = True
    render_and_evaluate_test_scenes(cfg, cfg.eval_all)

    print("\nEnd")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--source_path", "-s", type=str)
    parser.add_argument("--model_path", "-m", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_config", default="", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--eval_all", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"load_iteration={str(args.iteration)}",
        f"+test_config={args.test_config}",
        f"+eval_all={args.eval_all}"
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()


            