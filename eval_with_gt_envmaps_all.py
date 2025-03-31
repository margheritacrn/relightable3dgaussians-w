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
from utils.loss_utils import mse2psnr, img2mae, img2mse, img2mse_image, l1_loss
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


@torch.no_grad()
def render_and_evaluate_test_scenes(cfg, eval_all=False):
    model = Relightable3DGW(cfg)

    bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Read test config
    sys.path.append(cfg.dataset.test_config_path)
    config_test = importlib.import_module("test_config").config
    config_test_names = [key.split(".")[0] for key in config_test.keys()]

    lighting_conditions = [config_test_name.split("_IMG")[0] if "IMG" in config_test_name else config_test_name.split("_DSC")[0] for config_test_name in config_test_names]
    test_views = [view for view in model.scene.getTestCameras()]
    lights_to_test_views = {lighting: [test_view for test_view in test_views if lighting in test_view.image_name] for lighting in lighting_conditions}

    out_dir_name = "eval_gt_envmap_all"
    renders_path = os.path.join(cfg.dataset.model_path, out_dir_name, "test", "iteration_{}".format(model.load_iteration), "renders")
    renders_unmasked_path = os.path.join(cfg.dataset.model_path, out_dir_name, "test", "iteration_{}".format(model.load_iteration), "renders_unmasked")
    gts_path = os.path.join(cfg.dataset.model_path, out_dir_name, "test", "iteration_{}".format(model.load_iteration), "gt")
    makedirs(renders_path, exist_ok=True)
    makedirs(renders_unmasked_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    ssims, psnrs, mses, maes, rec_losses = [], [], [], [], []
    img_names = []
    sky_sh = torch.zeros((9,3), dtype=torch.float32, device="cuda")

    for lighting_cond in tqdm(lights_to_test_views.keys()):
        test_views = lights_to_test_views[lighting_cond]
        source_test_imname = next((config_test_name) for config_test_name in config_test_names if lighting_cond in config_test_name)
        config = config_test[source_test_imname]
        envmap_img_path = config["env_map_path"]
        init_rot_x = config["initial_env_map_rotation"]["x"]
        init_rot_y = config["initial_env_map_rotation"]["y"]
        init_rot_z = config["initial_env_map_rotation"]["z"]
        threshold = config["env_map_scaling"]["threshold"]
        scale = config["env_map_scaling"]["scale"]
        sun_angle_range = config["sun_angles"]
        # Preprocess environment map
        gt_envmap_sh = process_environment_map_image(envmap_img_path, scale, threshold) # (25,3)

        n = 51
        sun_angles_prepare_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], n)
        sun_angles = [torch.tensor([angle,0, 0]) for angle in sun_angles_prepare_list] #rotate only around y

        best_psnr = 0
        best_angle = None

        for angle in tqdm(sun_angles):
            views_psnrs = []
            all_psnrs = []
            for view in test_views:

                gt_image = view.original_image.cuda()

                eval_mask_path = os.path.join(cfg.dataset.source_path, "test", "cityscapes_mask", "binary_masks", view.image_name + ".png")
                eval_mask = cv2.imread(eval_mask_path, cv2.IMREAD_GRAYSCALE)
                eval_mask = cv2.resize(eval_mask, (gt_image.shape[2], gt_image.shape[1]))
                kernel = np.ones((5, 5), np.uint8)
                eval_mask = cv2.erode(eval_mask, kernel, iterations=1)
                eval_mask = torch.from_numpy(eval_mask//255).cuda()

                # Rotate envmap and render
                gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, init_rot_z, init_rot_y, init_rot_x, 'real')
                gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh_rot, angle[2], angle[0], angle[1], 'real')
                gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot.T, dtype=torch.float32, device="cuda")
                model.envlight.set_base(gt_envmap_sh_rot)
                render_pkg = render(view, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree, cfg.pipe, background, debug=False, fix_sky=True, specular=cfg.specular)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                # Compute PSNR
                view_psnr = mse2psnr(img2mse(rendering, gt_image, mask=eval_mask))
                views_psnrs.append(view_psnr.cpu())
    
            current_psnr = torch.tensor(views_psnrs).mean()
            all_psnrs.append(current_psnr.cpu())

            if current_psnr > best_psnr:
                best_angle = angle
                best_psnr = current_psnr

        print(f"{lighting_cond}- Lowest avg PSNR: {np.array(all_psnrs).min()}\n")
        print(f"{lighting_cond}- Best angle {best_angle}")
        print(f"{lighting_cond}- Best avg PSNR: {best_psnr}\n")

        # Get best orientation of the gt envmap
        gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, init_rot_z, init_rot_y, init_rot_x, 'real')
        gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh_rot, angle[2], angle[0], angle[1], 'real')

        # Save best orientation of the gt envmap:
        np.save(os.path.join(renders_path, "best_envmap" + lighting_cond+".npy"), gt_envmap_sh_rot)
        render_best_angle_envmap = sh_utility.sh_render(gt_envmap_sh_rot.T, width = 360)
        render_best_angle_envmap = torch.tensor(render_best_angle_envmap**(1/ 2.2))
        render_best_angle_envmap =  np.array(render_best_angle_envmap*255).clip(0,255).astype(np.uint8)
        render_best_angle_envmap = Image.fromarray(render_best_angle_envmap)
        render_best_angle_envmap.save(os.path.join(renders_path, "best_angle_rot_envmap"+lighting_cond+".jpg"))

        gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot.T, dtype=torch.float32, device="cuda")

        # Render with best angle
        for view in tqdm(test_views):

                gt_image = view.original_image.cuda()

                sky_mask = view.sky_mask.cuda()

                eval_mask_path = os.path.join(cfg.dataset.source_path, "test", "cityscapes_mask", "binary_masks", view.image_name + ".png")
                eval_mask = cv2.imread(eval_mask_path, cv2.IMREAD_GRAYSCALE)
                eval_mask = cv2.resize(eval_mask, (gt_image.shape[2], gt_image.shape[1]))
                kernel = np.ones((5, 5), np.uint8)
                eval_mask = cv2.erode(eval_mask, kernel, iterations=1)
                eval_mask = torch.from_numpy(eval_mask//255).cuda()

                # Render
                model.envlight.set_base(gt_envmap_sh_rot)
                render_pkg = render(view, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree, cfg.pipe, background, debug=False, fix_sky=True, specular=cfg.specular)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                rendering_masked = torch.clamp(render_pkg["render"]*eval_mask, 0.0, 1.0)
                torch.cuda.synchronize()
                gt_image = gt_image[0:3, :, :]
                torchvision.utils.save_image(rendering_masked, os.path.join(renders_path, view.image_name + "_masked.png"))
                torchvision.utils.save_image(rendering, os.path.join(renders_unmasked_path, view.image_name + ".png"))
                torchvision.utils.save_image(rendering*sky_mask + torch.ones_like(rendering)*(1 - sky_mask),
                                                os.path.join(renders_unmasked_path, view.image_name + "_masked_sky.png"))
                torchvision.utils.save_image(gt_image*eval_mask, os.path.join(gts_path, view.image_name + ".png"))

                img_names.append(view.image_name)
        
                # Compute metrics
                psnrs.append(mse2psnr(img2mse(rendering, gt_image, mask=eval_mask)))
                maes.append(img2mae(rendering, gt_image, mask=eval_mask))
                mses.append(img2mse(rendering, gt_image, mask=eval_mask))

                rendered_np= rendering.cpu().detach().numpy().transpose(1, 2, 0)
                gt_image_np = gt_image.cpu().detach().numpy().transpose(1, 2, 0)
                
                _, full = ssim_skimage(rendered_np, gt_image_np, win_size=5, channel_axis=2, full=True, data_range=1.0)
                mssim_over_mask = (torch.tensor(full).cuda()*eval_mask.unsqueeze(-1)).sum() / (3*eval_mask.sum())
                ssims.append(mssim_over_mask)
                Ll1 = l1_loss(rendering, gt_image, mask=eval_mask.expand_as(gt_image))
                Ssim = (1.0 - mssim_over_mask)
                rec_loss = Ll1 * (1-cfg.optimizer.lambda_dssim) + cfg.optimizer.lambda_dssim * Ssim
                rec_losses.append(rec_loss)


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
        print("  REC LOSS: {:>12.7f}".format(torch.tensor(rec_losses).mean(), ".5"), file=f)
        print(f"  best PSNRs: {psnrs_dict}", file=f)
        print(f"  best MSEs: {mses_dict}", file=f)
        print(f"  best MAEs: {maes_dict}", file=f)
        print(f"  best SSIMs: {ssims_dict}", file=f)


@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print("Rendering and evaluating with GT illumination" + cfg.dataset.model_path)
    cfg.dataset.eval = True
    render_and_evaluate_test_scenes(cfg,  cfg.eval_all)

    print("\nEnd")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation with GT environment maps script parameters")
    parser.add_argument("--source_path", "-s", type=str)
    parser.add_argument("--model_path", "-m", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_config_path", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--eval_all", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"dataset.test_config_path={args.test_config_path}",
        f"load_iteration={str(args.iteration)}",
        f"+eval_all={args.eval_all}"
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()


            