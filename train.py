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
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, depth_loss_gaussians, envlight_loss, min_scale_loss, envl_sh_loss
from gaussian_renderer import render
import sys
from utils.general_utils import safe_state, grad_thr_exp_scheduling
from utils.image_utils import apply_depth_colormap
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from scene.relit3DGW_model import Relightable3DGW
from omegaconf import DictConfig
import hydra
from utils.sh_utils import render_sh_map
from eval_with_gt_envmaps import evaluate_test_report


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

    eval = cfg.dataset.eval

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
        envlight_sh, sky_sh = model.mlp(embedding_gt_image)
        envlight_sh_rand_noise = torch.randn_like(envlight_sh)*0.025
        # Get environment lighting object for the current training image
        model.envlight.set_base(envlight_sh + envlight_sh_rand_noise)

        render_pkg = render(viewpoint_cam, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree, cfg.pipe, background, debug=False, fix_sky=cfg.fix_sky, specular=cfg.specular)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        diff_col, spec_col = render_pkg["diffuse_color"], render_pkg["specular_color"]

        # Loss
        Ll1 = l1_loss(image, gt_image, mask=occluders_mask)
        Ssim = (1.0 - ssim(image, gt_image, mask=occluders_mask))
        rec_loss = Ll1 * (1-cfg.optimizer.lambda_dssim) + cfg.optimizer.lambda_dssim * Ssim
        loss = rec_loss
        logs =  {"Reconstruction loss": f"{rec_loss:.{7}f}"}

        # Sky Gaussians regularization
        loss_sky_brdf = l1_loss(diff_col, torch.zeros_like(diff_col), mask=1-sky_mask) + l1_loss(spec_col, torch.zeros_like(spec_col), mask=1-sky_mask)
        loss += cfg.optimizer.lamba_sky_brdf * loss_sky_brdf

        # Normal regularization
        if iteration > cfg.optimizer.reg_normal_from_iter and cfg.optimizer.lambda_normal > 0:
            rendered_normal = render_pkg["normal"]*occluders_mask*sky_mask
            rendered_surf_normal = render_pkg["normal_ref"]*occluders_mask*sky_mask
            normal_consistency_loss =  (1 - (rendered_normal*rendered_surf_normal).sum(dim=0))[None]
            normal_consistency_loss = cfg.optimizer.lambda_normal * (normal_consistency_loss).mean()
            loss += normal_consistency_loss
            logs.update({"Normal loss": f"{normal_consistency_loss:.{5}f}"})

        # Envlight regularization
        if cfg.optimizer.lambda_envlight > 0:         
            envl_loss = envl_sh_loss(envlight_sh, cfg.envlight_sh_degree)
            loss += envl_loss
            logs.update({"Envlight loss": f"{envl_loss:.{5}f}"})

        # Planar regularization
        if cfg.optimizer.lambda_scale > 0:
            scale_loss = cfg.optimizer.lambda_scale * min_scale_loss(radii, model.gaussians)
            loss += scale_loss

        # Depth regularization
        if iteration > cfg.optimizer.reg_sky_gauss_depth_from_iter and cfg.optimizer.lambda_sky_gauss > 0:
            sky_gaussians_mask = model.gaussians.get_is_sky.squeeze()
            gaussians_depth = model.gaussians.get_depth(viewpoint_cam)
            sky_gaussians_depth = gaussians_depth[(sky_gaussians_mask) & (visibility_filter)]
            avg_depth_sky_gauss = torch.mean(sky_gaussians_depth)
            avg_depth_non_sky_gauss = torch.mean(gaussians_depth[(~sky_gaussians_mask) & (visibility_filter)]).detach()
            depth_loss_sky_gauss = cfg.optimizer.lambda_sky_gauss * depth_loss_gaussians(avg_depth_sky_gauss, avg_depth_non_sky_gauss)
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
                            model, eval, render, {"sky_sh_degree": cfg.sky_sh_degree, "pipe": cfg.pipe, "background": background, "debug": True, "fix_sky": cfg.fix_sky, "specular": cfg.specular})
            if iteration in saving_iterations or iteration == cfg.optimizer.iterations:
                print(f" ITER: {iteration} saving model")
                model.save(iteration)

            # Densification
            if iteration < cfg.optimizer.densify_until_iter:
                model.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration == cfg.optimizer.densify_from_iter:
                    grad_threshold = cfg.optimizer.densify_grad_threshold

                if iteration > cfg.optimizer.densify_from_iter and iteration % cfg.optimizer.densification_interval == 0:
                    dir_pp = (model.gaussians.get_xyz - viewpoint_cam.camera_center.repeat(model.gaussians.get_opacity.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    size_threshold = 20 if iteration > cfg.optimizer.opacity_reset_interval else None
                    model.gaussians.densify_and_prune(grad_threshold, 0.005, model.scene.cameras_extent, size_threshold, dir_pp_normalized)
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


def training_report(tb_writer, iteration, Ll1, loss, losses_extra, l1_loss, elapsed, testing_iterations, model: Relightable3DGW, eval, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for k in losses_extra.keys():
            tb_writer.add_scalar(f'train_loss_patches/{k}_loss', losses_extra[k].item(), iteration)

    # Report samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        train_config = [{'name': 'train', 'cameras' : [model.scene.getTrainCameras()[idx % len(model.scene.getTrainCameras())] for idx in range(5, 30, 5)]}]
        with torch.no_grad():
            for config in train_config:
                if config['cameras'] and len(config['cameras']) > 0:
                    images = []
                    gts = []
                    for idx, viewpoint in enumerate(config['cameras']):
                        gt_image = viewpoint.original_image.cuda()
                        viewpoint_cam_id = torch.tensor([viewpoint.uid], device = 'cuda')
                        embedding_gt_image = model.embeddings(viewpoint_cam_id)
                        envlight_sh, sky_sh = model.mlp(embedding_gt_image)
                        model.envlight.set_base(envlight_sh)
                        render_pkg = renderFunc(viewpoint, model.gaussians, model.envlight, sky_sh, renderArgs["sky_sh_degree"], renderArgs["pipe"],
                                                renderArgs["background"], debug=renderArgs["debug"], fix_sky=renderArgs["fix_sky"], specular=renderArgs["specular"])
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        images.append(image)
                        gt_image = torch.clamp(gt_image, 0.0, 1.0)
                        gts.append(gt_image)
                        reconstructed_envlight = model.envlight.render_sh().cuda().permute(2,0,1)
                        reconstructed_sky_map = render_sh_map(sky_sh.squeeze()).cuda().permute(2,0,1)
                        specular_light = model.envlight.get_specular_light_sh(torch.mean(model.gaussians.get_roughness).unsqueeze(0))
                        reconstructed_spec_light = render_sh_map(specular_light.squeeze(), gamma_correct=True).cuda().permute(2,0,1)
                        if tb_writer and (idx < 10):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/reconstructed_envlight".format(viewpoint.image_name), reconstructed_envlight[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/reconstructed_sky_map".format(viewpoint.image_name), reconstructed_sky_map[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/reconstructed_spec_light".format(viewpoint.image_name), reconstructed_spec_light[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            for k in render_pkg.keys():
                                if (render_pkg[k].dim()<3 or k=="render") and (k != "render_sun_dir"):
                                    continue
                                if "diffuse" in k:
                                    image_k = render_pkg[k]
                                elif k == "depth":
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
                    print("\n[ITER {}] Evaluating train : L1 {} PSNR {}".format(iteration, l1_train, psnr_train))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_train, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_train, iteration)

                    if eval:
                        psnr_test =  evaluate_test_report(model, renderArgs["background"], iteration, tb_writer)
                        print("\n[ITER {}] Evaluating test : PSNR {}".format(iteration, psnr_test))

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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 7_000, 30_000])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test_config_path", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--lambda_sky_gauss", type=float, default=0.05)
    parser.add_argument("--reg_normal_from_iter", type=float, default=15_000)
    parser.add_argument("--reg_sky_gauss_depth_from_iter", type=float, default=0)
    parser.add_argument("--lambda_envlight", type=float, default=100)
    parser.add_argument("--init_embeddings", action="store_true")
    parser.add_argument("--init_sh_mlp", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"init_embeddings={str(args.init_embeddings)}",
        f"init_sh_mlp={str(args.init_sh_mlp)}",
        f"dataset.eval={str(args.eval)}",
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"dataset.test_config_path={args.test_config_path}",
        f"optimizer.lambda_sky_gauss={args.lambda_sky_gauss}",
        f"optimizer.reg_normal_from_iter={args.reg_normal_from_iter}",
        f"optimizer.reg_sky_gauss_depth_from_iter={args.reg_sky_gauss_depth_from_iter}",
        f"optimizer.lambda_envlight={args.lambda_envlight}",
        f"+test_iterations={args.test_iterations}",
        f"+save_iterations={args.save_iterations}",
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    sys.argv = [sys.argv[0]] + cl_args
    main()

