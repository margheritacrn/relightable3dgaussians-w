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
from envmap import EnvironmentMap
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
from process_gt_envmaps import rotate_envmap

TINY_NUMBER = 1e-6


def render_and_evaluate(cfg):
    with torch.no_grad():
        model = Relightable3DGW(cfg)

        bg_color = [1,1,1] if cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Read test config
        sys.path.append(cfg.test_config)
        config_test = importlib.import_module("test_config").config
        config_test_names = [key.split(".")[0] for key in config_test.keys()]

        test_cameras = [c for c in model.scene.getTestCameras() if c.image_name in config_test_names]
        renders_path = os.path.join(cfg.dataset.model_path, "eval_lumigauss", "test", "iteration_{}".format(model.load_iteration), "renders")
        gts_path = os.path.join(cfg.dataset.model_path, "eval_lumigauss", "test", "iteration_{}".format(model.load_iteration), "gt")
        makedirs(renders_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        ssims, psnrs, mses, maes = [], [], [], []
        img_names, used_angles =[], []

        for view in tqdm(test_cameras):
            print(view.image_name)

            image_config = config_test[view.image_name]
            mask_path = image_config["mask_path"]
            envmap_img_path = image_config["env_map_path"]
            init_rot_x = image_config["initial_env_map_rotation"]["x"]
            init_rot_y = image_config["initial_env_map_rotation"]["y"]
            init_rot_z = image_config["initial_env_map_rotation"]["z"]
            threshold = image_config["env_map_scaling"]["threshold"]
            scale = image_config["env_map_scaling"]["scale"]
            sun_angle_range = image_config["sun_angles"]

            gt_envmap = EnvironmentMap(envmap_img_path, 'latlong')
            gt_envmap.data[gt_envmap.data > threshold] *= scale

            # Get gt
            gt_image = view.original_image.cuda()

            # Get eval mask
            mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask=cv2.resize(mask, (gt_image.shape[2], gt_image.shape[1]))
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask=torch.from_numpy(mask//255).cuda()

            best_psnr = 0
            best_angle = None
        
            n = 2 #51
            sun_angles_prepare_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], n)  
            sun_angles = [torch.tensor([angle,0, 0]) for angle in sun_angles_prepare_list] #rotate only around y

            for angle in tqdm(sun_angles):
                # rotate envmap and render
                gt_envmap_rot = rotate_envmap(gt_envmap.copy(), angles=[init_rot_z, init_rot_y, init_rot_x])
                gt_envmap_rot, gt_rot_envmap_sh = rotate_envmap(gt_envmap_rot, angles=[angle[2], angle[0], angle[1]],
                                                  return_sh=True, lmax=4)
                
                gt_rot_envmap_sh = torch.tensor(gt_rot_envmap_sh, dtype=torch.float32, device="cuda")
                model.envlight.set_base(gt_rot_envmap_sh)

                render_pkg = render(view, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
                render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

                # compute metrics
                current_psnr = mse2psnr(img2mse(render_pkg["render"], gt_image, mask=mask))

                if current_psnr > best_psnr:
                    best_angle = angle
                    best_psnr = current_psnr

            # Render with gtenvmap rotated according to best angles        
            gt_envmap_rot = rotate_envmap(gt_envmap.copy(), angles=[init_rot_z, init_rot_y, init_rot_x]) 
            gt_envmap_rot = rotate_envmap(gt_envmap_rot, angles=[best_angle[2], best_angle[0], best_angle[1]],
                                        return_sh=True, lmax=4)
            gt_rot_envmap_sh = torch.tensor(gt_rot_envmap_sh, dtype=torch.float32, device="cuda")
            model.envlight.set_base(gt_rot_envmap_sh)

            render_pkg = render(view, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
            render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torch.cuda.synchronize()
            gt_image = gt_image[0:3, :, :]
            torchvision.utils.save_image(render_pkg["render"], os.path.join(renders_path, view.image_name + ".png"))
            torchvision.utils.save_image(gt_image, os.path.join(gts_path, view.image_name + ".png"))
            
            used_angles.append(best_angle)
            img_names.append(view.image_name)
            
            # Compute metrics
            psnrs.append(mse2psnr(img2mse(render_pkg["render"], gt_image, mask=mask)))
            maes.append(img2mae(render_pkg["render"], gt_image, mask=mask))
            mses.append(img2mse(render_pkg["render"], gt_image, mask=mask))

            rendered_np= render_pkg["render"].cpu().detach().numpy().transpose(1, 2, 0)
            gt_image_np = gt_image.cpu().detach().numpy().transpose(1, 2, 0)
            
            _, full = ssim_skimage(rendered_np, gt_image_np, win_size=5, channel_axis=2, full=True, data_range=1.0)
            mssim_over_mask = (torch.tensor(full).cuda()*mask.unsqueeze(-1)).sum() / (3*mask.sum())
            ssims.append(mssim_over_mask)

    # Save metrics 
    with open(os.path.join(renders_path, "metrics.txt"), 'w') as f:
        print("  PSNR: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"), file=f)
        print("  MSE: {:>12.7f}".format(torch.tensor(mses).mean(), ".5"), file=f)
        print("  MAE: {:>12.7f}".format(torch.tensor(maes).mean(), ".5"), file=f)
        print("  SSIM: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"), file=f)


@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print("Rendering with GT light" + cfg.dataset.model_path)
    render_and_evaluate(cfg)
    print("\nEnd")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_config", default="", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"load_iteration={str(args.iteration)}",
        f"+test_config={args.test_config}"
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    sys.argv = [sys.argv[0]] + cl_args
    main()


            