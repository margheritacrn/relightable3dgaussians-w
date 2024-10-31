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
from utils.loss_utils import l1_loss, ssim, predicted_normal_loss, delta_normal_loss, zero_one_loss, envlight_loss, envlight_loss2, envlight_init_loss
from gaussian_renderer import render, network_gui, render_lighting
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import apply_depth_colormap
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time
from scene.NVDIFFREC import load_sh_env
from scene.net_models import SHMlp, EmbeddingNet
#TODO: add training for envlight--> pretraining of AE and then train along with the Gaussians the MLP returning SH coefficients
#NOTE: I deactivated temporarily network gui (reactivate later for training with debug)
#TODO: add regularization term for environment light SH coefficients: they must be positive
#TODO: None lighting conditions
#TODO: edit dataset.sh_degree, I should use it for envlight_sh_mlp with assertion of it being >=4
#TODO: consider wehther to create a model class. because for ex init envlight sh doesn't make a lot of sense inside
# Scene-


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    envlight_sh_mlp = SHMlp(embedding_dim=dataset.embeddings_dim).cuda()
    scene = Scene(dataset, gaussians, envlight_sh_mlp)
    envlight_sh_inits = scene.envlight_sh_init
    embeddings = scene.embeddings
    gaussians.training_setup(envlight=envlight_sh_mlp, embeddings=embeddings, training_args=opt)
    viewpoint_stack = scene.getTrainCameras().copy()
    # Initialize embeddings
    if dataset.init_embeddings:
        embedding_network = EmbeddingNet(latent_dim=dataset.embeddings_dim)
        embedding_network.cuda()
        checkpoint = dataset.model_path + f'/EmbeddingNet_model_epoch_{opt.embednet_pretrain_epochs-1}.pth'
        #checkpoint = './output_adjusted_envloss_prior0_1/schloss'+ f'/EmbeddingNet_model_epoch_{opt.embednet_pretrain_epochs-1}.pth'
        if os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint, weights_only=True)
            embedding_network.load_state_dict(state_dict)
            data_transforms = embedding_network.optimize_ae(data_path=dataset.source_path, get_datatransforms_only=True)
        else:
            progress_bar_light_ae = tqdm(range(1, opt.embednet_pretrain_epochs + 1), desc = "Training images embeddings pretraining progress")
            output_embednet = embedding_network.optimize_ae(data_path=dataset.source_path,
                                                                    num_epochs = opt.embednet_pretrain_epochs,
                                                                    tensorboard_writer = tb_writer, progress_bar=progress_bar_light_ae,
                                                                    output_path=dataset.model_path, return_outputs=True
                                                                )
            data_transforms = output_embednet["data_transforms"]
        embedding_network.eval()
        with torch.no_grad():
            batch_imgs = torch.stack([data_transforms(viewpoint_cam.original_image) for viewpoint_cam in viewpoint_stack]).to(dtype=torch.float32,  device='cuda')
            embeddings_inits = embedding_network(batch_imgs)
        del embedding_network
        torch.cuda.empty_cache()
        # Initialize per-image embeddings
        embeddings.weight = torch.nn.Parameter(embeddings_inits)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1): 
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
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device = 'cuda')
        gt_image = viewpoint_cam.original_image.cuda()
        gt_light_condition = viewpoint_cam.image_name[:-9]
        init_light_condition = next((key for key in envlight_sh_inits if gt_light_condition in key), None)
        """
        if init_light_condition is None:
            continue
        else:
        """
        # gt_envlight_sh_init = envlight_sh_inits[init_light_condition]
       #  gt_envlight_sh_init = torch.tensor(gt_envlight_sh_init, dtype=torch.float32, device="cuda")

        gaussians.set_requires_grad("normal", state=iteration >= opt.normal_reg_from_iter)
        gaussians.set_requires_grad("normal2", state=iteration >= opt.normal_reg_from_iter)
        # Get SH coefficients of environment lighting for current training image
        #TODO: edit here: sh_envlight_mlp needs an embedding vector as input. I probably need the camera index. Compare vewpoint stack content with scene_info in scne/__ini__.py
        embedding_gt_image = embeddings(viewpoint_cam_id)
        envlight_sh = envlight_sh_mlp(embedding_gt_image)
        # Create environment lighting object for the current training image
        envlight = load_sh_env(envlight_sh)

        # Render
        render_pkg = render(viewpoint_cam, gaussians, envlight, pipe, background, debug=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        losses_extra = {}
        if iteration > opt.normal_reg_from_iter:
            if iteration<opt.normal_reg_util_iter:
                losses_extra['predicted_normal'] = predicted_normal_loss(render_pkg["normal"], render_pkg["normal_ref"], render_pkg["alpha"])
            losses_extra['zero_one'] = zero_one_loss(render_pkg["alpha"])
            if "delta_normal_norm" not in render_pkg.keys() and opt.lambda_delta_reg>0: assert()
            if "delta_normal_norm" in render_pkg.keys():
                losses_extra['delta_reg'] = delta_normal_loss(render_pkg["delta_normal_norm"], render_pkg["alpha"])

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        for k in losses_extra.keys():
            loss += getattr(opt, f'lambda_{k}')* losses_extra[k]
        # Envlight losses
        if iteration <= opt.envlight_loss_until_iter:
            dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_opacity.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # Create a copy of the normals without gradient tracking for envlight loss computation
            normals = gaussians.get_normal(dir_pp_normalized=dir_pp_normalized).data.clone()
            envl_loss = envlight_loss(envlight, normals)
            # roughness = gaussians.get_roughness
            # roughness = roughness.data.clone()
            # envl_loss = envlight_loss2(envlight, normals, roughness)
            loss += opt.lambda_envlight*envl_loss
        if opt.lambda_envlight_init > 0 and iteration <= opt.envlight_init_until_iter:
            envl_init_loss = envlight_init_loss(envlight_sh, gt_envlight_sh_init)
            loss += opt.lambda_envlight_init * envl_init_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            losses_extra['psnr'] = psnr(image, gt_image).mean()
            training_report(tb_writer, iteration, Ll1, loss, losses_extra, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, embeddings, envlight_sh_mlp, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step: for both gaussians parameters and envlight MLP (per-image)
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration)

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

def training_report(tb_writer, iteration, Ll1, loss, losses_extra, l1_loss, elapsed, testing_iterations, scene : Scene, embeddings, envlight_sh_mlp, renderFunc, renderArgs):
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
        validation_configs = [{'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}]
        with torch.no_grad():
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    images = torch.tensor([], device="cuda")
                    gts = torch.tensor([], device="cuda")
                    for idx, viewpoint in enumerate(config['cameras']):
                        gt_image = viewpoint.original_image.cuda()
                        viewpoint_cam_id = torch.tensor([viewpoint.uid], device = 'cuda')
                        embedding_gt_image = embeddings(viewpoint_cam_id)
                        envlight_sh = envlight_sh_mlp(embedding_gt_image)
                        envlight = load_sh_env(envlight_sh)
                        render_pkg = renderFunc(viewpoint, scene.gaussians, envlight, *renderArgs)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(gt_image, 0.0, 1.0)
                        images = torch.cat((images, image.unsqueeze(0)), dim=0)
                        gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
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
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/roughness_histogram", scene.gaussians.get_roughness, iteration)
            tb_writer.add_histogram("scene/metalness_histogram", scene.gaussians.get_metalness, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
