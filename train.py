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
from utils.loss_utils import l1_loss, ssim, predicted_normal_loss, delta_normal_loss, zero_one_loss, envlight_loss, envlight_prior_loss, min_scale_loss
from gaussian_renderer import render, network_gui
import sys
from utils.general_utils import safe_state
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
from render import render_sets_training
import hydra
#TODO: add training for envlight--> pretraining of AE and then train along with the Gaussians the MLP returning SH coefficients
#NOTE: I deactivated temporarily network gui (reactivate later for training with debug)
#TODO: add regularization term for environment light SH coefficients: they must be positive
#TODO: None lighting conditions
#TODO: edit dataset.sh_degree, I should use it for envlight_sh_mlp with assertion of it being >=4
#TODO: consider wehther to create a model class. because for ex init envlight sh doesn't make a lot of sense inside


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
        """
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        """

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
           model.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = model.scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device = 'cuda')
        gt_image = viewpoint_cam.original_image.cuda()
        if cfg.optimizer.lambda_envlight_sh_prior > 0:
            gt_light_condition = viewpoint_cam.image_name[:-9]
            init_light_condition = next((key for key in model.envlight_sh_priors if gt_light_condition in key), None)
            gt_envlight_sh_prior = model.envlight_sh_priors[init_light_condition]
            gt_envlight_sh_prior = torch.tensor(gt_envlight_sh_prior, dtype=torch.float32, device="cuda")

        # Get SH coefficients of environment lighting for current training image
        #TODO: edit here: sh_envlight_mlp needs an embedding vector as input. I probably need the camera index. Compare vewpoint stack content with scene_info in scne/__ini__.py
        embedding_gt_image = model.embeddings(viewpoint_cam_id)
        envlight_sh = model.envlight_sh_mlp(embedding_gt_image)
        # Get environment lighting object for the current training image
        model.envlight.set_base(envlight_sh)

        # Render
        render_pkg = render(viewpoint_cam, model.gaussians, model.envlight, cfg.pipe, background, debug=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - cfg.optimizer.lambda_dssim) * Ll1 + cfg.optimizer.lambda_dssim * (1.0 - ssim(image, gt_image))
        # Normal loss
        if cfg.optimizer.lambda_normal > 0 and iteration > cfg.optimizer.reg_normal_from_iter:
            normal_loss = predicted_normal_loss(render_pkg["normal"], render_pkg["normal_ref"], render_pkg["alpha"], sky_mask = viewpoint_cam.sky_mask.cuda())
            loss += cfg.optimizer.lambda_normal*normal_loss
        # Envlight losses
        if iteration <= cfg.optimizer.envlight_loss_until_iter:
            viewing_dirs = (model.gaussians.get_xyz - viewpoint_cam.camera_center.repeat(model.gaussians.get_opacity.shape[0], 1))
            viewing_dirs_norm = viewing_dirs/viewing_dirs.norm(dim=1, keepdim=True)
            # Create a copy of the normals without gradient tracking for envlight loss computation
            normals = model.gaussians.get_normal(dir_pp_normalized=viewing_dirs_norm).data.clone() #NOTE: could just use detach()
            envl_loss = envlight_loss(model.envlight, normals)
            loss += cfg.optimizer.lambda_envlight*envl_loss
        if cfg.optimizer.lambda_envlight_sh_prior > 0 and iteration <= cfg.optimizer.envlight_prior_until_iter:
            envl_init_loss = envlight_prior_loss(envlight_sh, gt_envlight_sh_prior)
            loss += cfg.optimizer.lambda_envlight_sh_prior * envl_init_loss
        # Planar regularization
        if cfg.optimizer.lambda_scale > 0:
            scale_loss = min_scale_loss(radii, model.gaussians)
            loss += cfg.optimizer.lambda_scale*scale_loss
        # Distortion loss
        if cfg.optimizer.lambda_dist > 0:
            dist_loss = cfg.optimizer.lambda_dist*(render_pkg["rendered_distance"]).mean()
            loss += cfg.optimizer.lambda_dist*dist_loss 
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(100)
            if iteration == cfg.optimizer.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            model.gaussians.max_radii2D[visibility_filter] = torch.max(model.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            losses_extra = {}
            losses_extra['psnr'] = psnr(image, gt_image).mean()
            training_report(tb_writer, iteration, Ll1, loss, losses_extra, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations,
                            model, render, (cfg.pipe, background))
            if iteration in saving_iterations or iteration == cfg.optimizer.iterations:
                #TODO: turn into  model.save iteration
                print(f" ITER: {iteration} saving model")
                model.save(iteration)

            # Densification
            if iteration < cfg.optimizer.densify_until_iter:
                model.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > cfg.optimizer.densify_from_iter and iteration % cfg.optimizer.densification_interval == 0:
                    size_threshold = 20 if iteration > cfg.optimizer.opacity_reset_interval else None
                    model.gaussians.densify_and_prune(cfg.optimizer.densify_grad_threshold, 0.005, model.scene.cameras_extent, size_threshold, viewing_dirs_norm)
                
                if iteration % cfg.optimizer.opacity_reset_interval == 0 or (cfg.dataset.white_background and iteration == cfg.optimizer.densify_from_iter):
                    model.gaussians.reset_opacity()

            # Optimizer step: for both gaussians parameters and envlight MLP (per-image)
            if iteration < cfg.optimizer.iterations:
                # torch.nn.utils.clip_grad_norm_(model.optimizer.param_groups[0]['params'], 1.0)
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

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #validation_configs = (#{'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            #  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        validation_configs = [{'name': 'train', 'cameras' : [model.scene.getTrainCameras()[idx % len(model.scene.getTrainCameras())] for idx in range(5, 30, 5)]}]
        with torch.no_grad():
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    images = torch.tensor([], device="cuda")
                    gts = torch.tensor([], device="cuda")
                    for idx, viewpoint in enumerate(config['cameras']):
                        gt_image = viewpoint.original_image.cuda()
                        if idx == 0:
                            h = gt_image.shape[0]
                            w = gt_image.shape[2]
                            resize_transform = transforms.Resize((h, w))
                        viewpoint_cam_id = torch.tensor([viewpoint.uid], device = 'cuda')
                        embedding_gt_image = model.embeddings(viewpoint_cam_id)
                        envlight_sh = model.envlight_sh_mlp(embedding_gt_image)
                        model.envlight.set_base(envlight_sh)
                        render_pkg = renderFunc(viewpoint, model.gaussians, model.envlight, *renderArgs)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        resize_image = transforms.ToPILImage()(image)
                        resized_image = resize_transform(resize_image)
                        final_image = transforms.ToTensor()(resized_image).cuda()
                        images = torch.cat((images, final_image.unsqueeze(0)), dim=0)
                        gt_image = torch.clamp(gt_image, 0.0, 1.0)
                        resize_gt_image = transforms.ToPILImage()(gt_image)
                        resized_gt_image = resize_transform(resize_gt_image)
                        final_gt_image = transforms.ToTensor()(resized_gt_image).cuda()
                        gts = torch.cat((gts, final_gt_image.unsqueeze(0)), dim=0)
                        if tb_writer and (idx < 10):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            for k in render_pkg.keys():
                                if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
                                    continue
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
                            
                            """if renderArgs[0].brdf:
                                lighting = render_lighting(scene.gaussians, resolution=(512, 1024))
                                if tb_writer:
                                    tb_writer.add_images(config['name'] + "/lighting", lighting[None], global_step=iteration)
                            """
                    l1_test = l1_loss(images, gts)
                    psnr_test = psnr(images, gts).mean()  
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

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
    parser.add_argument("--num_sky_points", type=int, default=15_000)
    args = parser.parse_args(sys.argv[1:])

    cl_args = [
        f"num_sky_points={args.num_sky_points}",
        f"dataset.eval={str(args.eval)}",
        f"dataset.model_path={args.model_path}",
        f"dataset.source_path={args.source_path}",
        f"+test_iterations={args.test_iterations}",
        f"+save_iterations={args.save_iterations}",
    ]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    sys.argv = [sys.argv[0]] + cl_args
    main()



