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

import torch
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
from math import exp
from utils.image_utils import erode
from utils.sh_utils import eval_sh
import numpy as np
from utils.general_utils import rand_hemisphere_dir
from scene.NVDIFFREC import util
from scene.NVDIFFREC.light import EnvironmentLight
import random
import torch.nn.functional as F


def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        assert mask.shape[0] == network_output.shape[0], "the mask must be expanded as the input images"
        num_pixels = torch.sum(mask == 1)
        return (torch.abs((network_output*mask - gt*mask)).sum())/num_pixels
    else:
        return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1_, img2_, window_size=11, size_average=True, mask=None):
    img1 = img1_.squeeze(0)
    img2 = img2_.squeeze(0)
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    img1 = img1[None,...]
    img2 = img2[None,...]
    mask = mask[None,...]
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if mask is not None:
            if(len(mask.shape) > 4):
                mask = mask.squeeze(0)
            assert mask.shape[1] == img1.shape[1], "the mask must be expanded as the input images"
            return ((ssim_map*mask).sum())/(torch.sum(mask == 1))
        return ssim_map.mean()
    else:
        if mask is not None:
            raise NotImplementedError
        return ssim_map.mean(1).mean(1).mean(1)

def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def predicted_depth_loss(depth_map, mask=None):
    with torch.no_grad():
        avg_depth_map = depth_map.permute(1,2,0).data.clone().cpu().numpy()
        avg_depth_map = cv2.blur(avg_depth_map.astype(np.float32),(5,5))
        avg_depth_map = torch.tensor(avg_depth_map).cuda()
    if mask is not None:
        depth_map = depth_map*mask
        avg_depth_map = avg_depth_map*mask.permute(1,2,0)
        loss = torch.abs(((depth_map.permute(1,2,0) - avg_depth_map))).sum()
        num_pixels = torch.sum(mask == 1)
        return torch.sum(loss)/num_pixels
    else:
        return torch.abs((depth_map.permute(1,2,0) - avg_depth_map)).mean()


def sky_depth_loss(depth_map, sky_mask, gamma = 0.02):
    """The function computes the mean depth in no-sky region and sky region and compares the difference.
    The function is based on the rendered depth map."""
    nosky_mask = 1 - sky_mask
    n_sky_pixels = torch.sum(nosky_mask == 1)
    if n_sky_pixels == 0:
        return 0
    n_no_sky_pixels = torch.sum(sky_mask == 1)
    with torch.no_grad():
        mean_depth_no_sky = (depth_map[0]*sky_mask).sum()/n_no_sky_pixels
    sky_depth = depth_map[0]*nosky_mask
    mean_depth_sky = (sky_depth).sum()/n_sky_pixels
    loss = torch.exp(-gamma*(mean_depth_sky-mean_depth_no_sky))
    return mean_depth_sky.detach(), loss


def depth_loss_gaussians(mean_depth_sky, mean_depth_non_sky, gamma = 0.02):
    """The function compares the difference of the average depth of sky and non sky gaussians using an exponential function."""
    loss = torch.exp(-gamma*(mean_depth_sky-mean_depth_non_sky))
    return loss


def predicted_normal_loss(normal, normal_ref, alpha=None, sky_mask = None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    if sky_mask is not None and torch.sum(sky_mask == 0) > 0:
        # Exclude sky region
        sky_mask = sky_mask.expand_as(normal)
        normal_ref = normal_ref*sky_mask
        normal = normal*sky_mask
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    # n_pred = F.normalize(n_pred, dim=-1)
    # l1_loss =(torch.abs(normal_ref - normal).sum(dim=-1)).mean()
    loss = ((1.0 - torch.sum(n * n_pred, axis=-1))).mean() # compute cos between n and n_pred, for them to be perfectly aligned it has to be 1

    return loss

def delta_normal_loss(delta_normal_norm, alpha=None):
    # delta_normal_norm: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(delta_normal_norm)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    l = delta_normal_norm.permute(1,2,0).reshape(-1,3)[...,0]
    loss = (w * l).mean()

    return loss


def envlight_loss(envlight: EnvironmentLight, normals: torch.Tensor, N_dirs: int = 1000, normals_subset_size = 100):
    """
    Regularization on environment lighting coefficients: incoming light should belong to R+.
    The loss is computed on a random subset of the input normals. For each normal N random directions
    in the hemisphere around it are sampled, then the irradiance corresponding to such direction is computed
    and the minimum between its values and 0 is taken.
    Args:
        envlight: environment lighting SH coefficients,
        normals: normal vectors of shape [..., 3],
        N_dirs: number of viewing directions samples,
        normals_subset_size: number of normal samples
    """
    assert normals.shape[-1] == 3 , "error: normals must have size  [...,3]"

    if normals.shape[0] > normals_subset_size:
        normals_rand_subset = random.sample(range(0, normals_subset_size), normals_subset_size)
        normals = normals[normals_rand_subset]

    # generate N_dirs random viewing directions in the hemisphere centered in n for each n in normals
    rand_hemisphere_dirs = rand_hemisphere_dir(N_dirs, normals) # (..., N, 3)
    # evaluate SH coefficients of env light
    light = eval_sh(envlight.sh_degree, envlight.base.transpose(0,1), rand_hemisphere_dirs)
    # extract negative values
    light = torch.minimum(light, torch.zeros_like(light))
    # average negative light values over number of viewing direction samples
    avg_light_per_normal = torch.mean(light, dim = 1)
    # average over normals
    avg_light = torch.mean(avg_light_per_normal, dim = 0)
    # take squared 2 norm
    envlight_loss = torch.mean((avg_light)**2)
    return envlight_loss


def envlight_loss_without_normals(envlight: EnvironmentLight, N_samples: int=10):

    viewing_dirs_unnorm = torch.empty(N_samples, 3, device=envlight.base.device).uniform_(-1,1)
    viewing_dirs_norm = viewing_dirs_unnorm/viewing_dirs_unnorm.norm(dim=1, keepdim=True)

    light = eval_sh(envlight.sh_degree, envlight.base.transpose(0,1), viewing_dirs_norm)

    envlight_loss = penalize_outside_range(light.view(-1), 0.0, torch.inf)

    return envlight_loss


def penalize_outside_range(tensor, lower_bound=0.0, upper_bound=1.0):
    error = 0
    below_lower_bound = tensor[tensor < lower_bound]
    above_upper_bound = tensor[tensor > upper_bound]
    if below_lower_bound.numel():
        error += torch.mean((below_lower_bound - lower_bound) ** 2)
    if above_upper_bound.numel():
        error += torch.mean((above_upper_bound - upper_bound) ** 2)
    return error


def envlight_prior_loss(sh_output: torch.Tensor, sh_envmap_init: torch.Tensor):
    return l2_loss(sh_output, sh_envmap_init)


def min_scale_loss(radii, gaussians):
    visibility_filter = radii > 0
    try:
        if visibility_filter.sum() > 0: # consider just visible gaussians
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0] # take minimum scales
            return min_scale_loss.mean()
    except Exception as e:
        raise RuntimeError(f"Failed to compute min_scale_loss: {e}")


def cam_depth2world_point(cam_z, pixel_idx, intrinsic, extrinsic):
    '''
    cam_z: (1, N)
    pixel_idx: (1, N, 2)
    intrinsic: (3, 3)
    extrinsic: (4, 4)
    world_xyz: (1, N, 3)
    '''
    valid_x = (pixel_idx[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    valid_y = (pixel_idx[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    ndc_xy = torch.stack([valid_x, valid_y], dim=-1)
    # inv_scale = torch.tensor([[W - 1, H - 1]], device=cam_z.device)
    # cam_xy = ndc_xy * inv_scale * cam_z[...,None]
    cam_xy = ndc_xy * cam_z[...,None]
    cam_xyz = torch.cat([cam_xy, cam_z[...,None]], dim=-1)
    world_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,0:1])], axis=-1) @ torch.inverse(extrinsic).transpose(0,1)
    world_xyz = world_xyz[...,:3]
    return world_xyz, cam_xyz

# from lumigauss

TINY_NUMBER = 1e-6

def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        if mask.shape[0] == x.shape[0]:
            return torch.sum((x - y) * (x - y) * mask.unsqueeze(0)) / (torch.sum(mask) + TINY_NUMBER)
        else:
            return torch.sum((x - y) * (x - y) * mask.unsqueeze(0)) / (torch.sum(mask)*x.shape[0] + TINY_NUMBER)

def img2mae(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.abs(x - y))
    else:
        if mask.shape[0] == x.shape[0]:
            return torch.sum(torch.abs(x - y) * mask.unsqueeze(0)) / (torch.sum(mask) + TINY_NUMBER)
        else:
            return torch.sum(torch.abs(x - y) * mask.unsqueeze(0)) / (torch.sum(mask) * x.shape[0] + TINY_NUMBER)

def mse2psnr(x):
    return -10. * torch.log(torch.tensor(x)+TINY_NUMBER) / torch.log(torch.tensor(10))

def img2mse_image(x, y, mask=None):
    
    # Compute squared difference per pixel per channel
    mse_image = (x - y) ** 2
    
    # If a mask is provided, apply the mask
    if mask is not None:
        # Ensure the mask is expanded to match the shape (3, W, H)
        mask = mask.unsqueeze(0)  # Add channel dimension (1, W, H) -> (3, W, H)
        mse_image = mse_image * mask

    return mse_image
