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
import os
import torch.nn.functional as F
import torch
from torchvision import transforms
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, predicted_normal_loss, predicted_depth_loss, depth_loss_gaussians, envlight_loss, min_scale_loss
from gaussian_renderer import render, network_gui
import sys
from utils.general_utils import safe_state, grad_thr_exp_scheduling
from utils.image_utils import apply_depth_colormap
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.relit3DGW_model import Relightable3DGW
from scene.net_models import SHMlp, EmbeddingNet
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import DictConfig
import hydra


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(cfg, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(cfg.dataset)
    model = Relightable3DGW(cfg)
    if cfg.init_sh_mlp:
        model.initialize_sh_mlp()
    model.training_set_up()

    bg_color = [1, 1, 1] if cfg.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(cfg.optimizer.iterations), desc="Training progress")
    for iteration in range(1, cfg.optimizer.iterations + 1): 
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = model.scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device = 'cuda')
        gt_image = viewpoint_cam.original_image.cuda()
        sky_mask = viewpoint_cam.sky_mask.expand_as(gt_image).cuda()
        occluders_mask = viewpoint_cam.occluders_mask.expand_as(gt_image).cuda()

        # Get SH coefficients of environment lighting for current training image
        embedding_gt_image = model.embeddings(viewpoint_cam_id)
        envlight_sh = model.envlight_sh_mlp(embedding_gt_image)
        # Get environment lighting object for the current training image
        model.envlight.set_base(envlight_sh)
        # Repeat for sky light
        skylight_sh = model.skylight_sh_mlp(embedding_gt_image)
        model.skylight.set_base(skylight_sh)

        if cfg.fix_sky:
            render_pkg = render(viewpoint_cam, model.gaussians, model.envlight, cfg.pipe, background, debug=False, fix_sky=True)
        else:
            render_pkg = render(viewpoint_cam, model.gaussians, model.envlight, model.skylight, cfg.pipe, background, debug=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        sky_color = render_pkg["sky_color"]

        # Loss
        if cfg.fix_sky:
            rec_loss = Ll1*(1-cfg.optimizer.lambda_dssim) + cfg.optimizer.lambda_dssim *(1.0 - ssim(image, gt_image, mask=occluders_mask*sky_mask))
        else:
            Ll1 = l1_loss(image, gt_image, mask=occluders_mask*sky_mask) + l1_loss(sky_color, gt_image, mask=1-sky_mask)
            Ssim = (1.0 - ssim(image, gt_image, mask=occluders_mask*sky_mask)) + (1.0 - ssim(sky_color, gt_image, mask=1-sky_mask))
            rec_loss = Ll1*(1-cfg.optimizer.lambda_dssim) + cfg.optimizer.lambda_dssim *Ssim
        loss = rec_loss
        logs =  {"Reconstruction loss": f"{rec_loss:.{7}f}"}


        # Normal regularization
        if iteration > cfg.optimizer.reg_normal_from_iter and cfg.optimizer.lambda_normal > 0:
            rendered_normal = render_pkg["normal"]*occluders_mask*sky_mask
            rendered_surf_normal = render_pkg["normal_ref"]*occluders_mask*sky_mask
            normals_prod_mask = ~((rendered_normal == 0).all(dim=0) & (rendered_surf_normal == 0).all(dim=0))
            normals_prod = rendered_normal*rendered_surf_normal#.detach()
            normal_consistency_loss =  (1 - (normals_prod[:,normals_prod_mask]).sum(dim=0))[None]
            normal_consistency_loss = cfg.optimizer.lambda_normal*(normal_consistency_loss).mean()
            loss += normal_consistency_loss
            logs.update({"Normal loss": f"{normal_consistency_loss:.{5}f}"})

        # Envlight regularization
        if cfg.optimizer.lambda_envlight > 0:
            viewing_dirs = (model.gaussians.get_xyz - viewpoint_cam.camera_center.repeat(model.gaussians.get_opacity.shape[0], 1))
            viewing_dirs_norm = viewing_dirs/viewing_dirs.norm(dim=1, keepdim=True)
            with torch.no_grad():
                normals = model.gaussians.get_normal(dir_pp_normalized=viewing_dirs_norm)
                normals = normals[~model.gaussians.get_is_sky.squeeze()]
            envl_loss = cfg.optimizer.lambda_envlight*envlight_loss(envlight_sh.squeeze(), model.envlight.get_shdegree, normals)
            skyl_loss = cfg.optimizer.lambda_envlight*envlight_loss(skylight_sh.squeeze(), model.skylight.get_shdegree, normals)
            loss += envl_loss + skyl_loss
            logs.update({"Envlight loss": f"{envl_loss:.{5}f}"})

        # Planar regularization
        if cfg.optimizer.lambda_scale > 0:
            scale_loss = cfg.optimizer.lambda_scale*min_scale_loss(radii, model.gaussians)
            loss += scale_loss

        # Depth regularization
        if iteration > cfg.optimizer.reg_sky_gauss_depth_from_iter and cfg.optimizer.lambda_sky_gauss > 0:
            sky_gaussians_mask = model.gaussians.get_is_sky.squeeze()
            gaussians_depth = model.gaussians.get_depth(viewpoint_cam)
            sky_gaussians_depth = gaussians_depth[(sky_gaussians_mask) & (visibility_filter)]
            avg_depth_sky_gauss = torch.mean(sky_gaussians_depth)
            avg_depth_non_sky_gauss = torch.mean(gaussians_depth[(~sky_gaussians_mask) & (visibility_filter)]).detach()
            depth_loss_sky_gauss = cfg.optimizer.lambda_sky_gauss*depth_loss_gaussians(avg_depth_sky_gauss, avg_depth_non_sky_gauss)
            loss += depth_loss_sky_gauss
            logs.update({"Depth loss": f"{depth_loss_sky_gauss:.{5}f}"})



        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix(logs)
                progress_bar.update(100)
            if iteration == cfg.optimizer.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            model.gaussians.max_radii2D[visibility_filter] = torch.max(model.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            losses_extra = {}
            losses_extra['psnr'] = psnr(image*occluders_mask, gt_image*occluders_mask).mean()
            training_report(tb_writer, iteration, Ll1, loss, losses_extra, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations,
                            model, render, {"pipe": cfg.pipe, "background": background, "debug": True, "fix_sky": cfg.fix_sky})
            if iteration in saving_iterations or iteration == cfg.optimizer.iterations:
                print(f" ITER: {iteration} saving model")
                model.save(iteration)

            # Densification
            if iteration < cfg.optimizer.densify_until_iter:
                model.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration == cfg.optimizer.densify_from_iter:
                    grad_threshold = cfg.optimizer.densify_grad_threshold

                if iteration > cfg.optimizer.densify_from_iter and iteration % cfg.optimizer.densification_interval == 0:
                    size_threshold = 20 if iteration > cfg.optimizer.opacity_reset_interval else None
                    model.gaussians.densify_and_prune(grad_threshold, 0.005, model.scene.cameras_extent, size_threshold, viewing_dirs_norm)
                    grad_threshold = grad_thr_exp_scheduling(iteration, cfg.optimizer.densify_until_iter, cfg.optimizer.densify_grad_threshold)

                if iteration % cfg.optimizer.opacity_reset_interval == 0 or (iteration == cfg.optimizer.densify_from_iter):
                    model.gaussians.reset_opacity()

            # Optimizer step
            if iteration < cfg.optimizer.iterations:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none = True)
                model.update_learning_rate(iteration)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if not args.logger:
        return tb_writer
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, losses_extra, l1_loss, elapsed, testing_iterations, model: Relightable3DGW, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for k in losses_extra.keys():
            tb_writer.add_scalar(f'train_loss_patches/{k}_loss', losses_extra[k].item(), iteration)

    # Report samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = [{'name': 'train', 'cameras' : [model.scene.getTrainCameras()[idx % len(model.scene.getTrainCameras())] for idx in range(5, 30, 5)]}]
        with torch.no_grad():
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    images = []
                    gts = []
                    for idx, viewpoint in enumerate(config['cameras']):
                        gt_image = viewpoint.original_image.cuda()
                        viewpoint_cam_id = torch.tensor([viewpoint.uid], device = 'cuda')
                        embedding_gt_image = model.embeddings(viewpoint_cam_id)
                        envlight_sh = model.envlight_sh_mlp(embedding_gt_image)
                        skylight_sh = model.skylight_sh_mlp(embedding_gt_image)
                        model.envlight.set_base(envlight_sh)
                        model.skylight.set_base(skylight_sh)
                        render_pkg = renderFunc(viewpoint, model.gaussians, model.envlight, model.skylight, renderArgs["pipe"],
                                                renderArgs["background"], debug=renderArgs["debug"], fix_sky=renderArgs["fix_sky"])
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        images.append(image)
                        gt_image = torch.clamp(gt_image, 0.0, 1.0)
                        gts.append(gt_image)
                        reconstructed_envlight = model.envlight.render_sh().cuda().permute(2,0,1)
                        reconstructed_skylight = model.skylight.render_sh().cuda().permute(2,0,1)
                        if tb_writer and (idx < 10):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/reconstructed_envlight".format(viewpoint.image_name), reconstructed_envlight[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/reconstructed_skylight".format(viewpoint.image_name), reconstructed_skylight[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            for k in render_pkg.keys():
                                if render_pkg[k].dim()<3 or k=="render" :
                                    continue
                                if "diffuse" in k:
                                    image_k = render_pkg[k]
                                if k == "depth":
                                    image_k = apply_depth_colormap(-render_pkg[k][0][...,None])
                                    image_k = image_k.permute(2,0,1)
                                elif k == "alpha":
                                    image_k = apply_depth_colormap(render_pkg[k][0][...,None], min=0., max=1.)
                                    image_k = image_k.permute(2,0,1)
                                else:
                                    if "normal" in k:
                                        render_pkg[k] = 0.5 + (0.5*render_pkg[k]) # (-1, 1) -> (0, 1)
                                    image_k = torch.clamp(render_pkg[k], 0.0, 1.0)
                                tb_writer.add_images(config['name'] + "_view_{}/{}".format(viewpoint.image_name, k), image_k[None], global_step=iteration)

                    l1_losses = [l1_loss(image, gt) for image, gt in zip(images,gts)]
                    l1_train = torch.tensor(l1_losses).mean() 
                    psnrs = [psnr(image, gt) for image, gt in zip(images,gts)]
                    psnr_train = torch.mean(torch.stack(psnrs))

                    print("\n[ITER {}] Evaluating train {}: L1 {} PSNR {}".format(iteration, config['name'], l1_train, psnr_train))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_train, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_train, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", model.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/roughness_histogram", model.gaussians.get_roughness, iteration)
            tb_writer.add_histogram("scene/metalness_histogram", model.gaussians.get_metalness, iteration)
            tb_writer.add_scalar('total_points', model.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="configs", config_name="relightable3DG-W")
def main(cfg: DictConfig):
    print(cfg)
    print("Optimizing " + cfg.dataset.model_path)
    training(cfg, cfg.test_iterations, cfg.save_iterations)
    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--lambda_sky_gauss", type=float, default=0.05)
    parser.add_argument("--reg_normal_from_iter", type=float, default=15_000)
    parser.add_argument("--reg_sky_gauss_depth_from_iter", type=float, default=0)
    parser.add_argument("--lambda_sky_col", type=float, default=0.5)
    parser.add_argument("--fix_sky", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"fix_sky={str(args.fix_sky)}",
        f"dataset.eval={str(args.eval)}",
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"optimizer.lambda_sky_gauss={args.lambda_sky_gauss}",
        f"optimizer.reg_normal_from_iter={args.reg_normal_from_iter}",
        f"optimizer.reg_sky_gauss_depth_from_iter={args.reg_sky_gauss_depth_from_iter}",
        f"optimizer.lambda_sky_col={args.lambda_sky_col}",
        f"+test_iterations={args.test_iterations}",
        f"+save_iterations={args.save_iterations}",
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    sys.argv = [sys.argv[0]] + cl_args
    main()

