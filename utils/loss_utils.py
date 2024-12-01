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


def l1_loss(network_output, gt, pixel_subset_size=None):
    if pixel_subset_size is not None:
        return (torch.abs((network_output - gt)).sum())/pixel_subset_size
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

def ssim(img1_, img2_, window_size=11, size_average=True):
    img1 = img1_.squeeze(0)
    img2 = img2_.squeeze(0)
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    img1 = img1[None,...]
    img2 = img2[None,...]
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
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def predicted_depth_loss(depth_map, sky_mask=None):
    with torch.no_grad():
        avg_depth_map = depth_map.permute(1,2,0).data.clone().cpu().numpy()
        avg_depth_map = cv2.blur(avg_depth_map.astype(np.float32),(5,5))
        avg_depth_map = torch.tensor(avg_depth_map).cuda()
    if sky_mask is not None:
        sky_mask = sky_mask.expand_as(depth_map)
        depth_map = depth_map*sky_mask
        avg_depth_map = avg_depth_map*sky_mask.permute(1,2,0)
        loss = (((depth_map.permute(1,2,0) - avg_depth_map).abs()).sum(dim=-1))
        num_sky_pixels = torch.sum(sky_mask == 1)
        return torch.sum(loss)/num_sky_pixels
    else:
        return torch.abs((depth_map.permute(1,2,0) - avg_depth_map)).mean()


def sky_depth_loss(depth_map, sky_mask, gamma = 0.02):
    # Compute mean depth in no-sky region and sky region and compare differences
    n_no_sky_pixels = torch.sum(sky_mask == 1)
    with torch.no_grad():
        mean_depth_no_sky = (depth_map*sky_mask.expand_as(depth_map)).sum()/n_no_sky_pixels
    sky_mask = 1 - sky_mask
    n_sky_pixels = torch.sum(sky_mask == 1)
    sky_depth = depth_map*sky_mask.expand_as(depth_map)
    mean_depth_sky = (sky_depth).sum()/n_sky_pixels
    max_no_sky = mean_depth_no_sky.max()
    max_sky = mean_depth_sky.max()
    loss = torch.exp(-gamma*(mean_depth_sky-mean_depth_no_sky))
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
    if sky_mask is not None:
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


def envlight_loss(envlight: EnvironmentLight, normals: torch.Tensor, N: int = 1000):
    """
    Regularization on environment lighting coefficients: incoming light should belong to R+.
    If the number of normals vectors is greater than subset_size=100, extraxct a random subset.
    Args:
        envlight: environment lighting
        normals: normal vectors of shape [..., 3]
        N: number of directions samples
    """
    assert len(normals.shape) == 2 and normals.shape[-1] == 3 , "error: n must have size  L X 3"

    normals_subset_size = 100
    if normals.shape[0] > normals_subset_size:
        normals_rand_subset = random.sample(range(0, normals_subset_size), normals_subset_size)
        normals = normals[normals_rand_subset]
    # generate N random samples directions in the hemisphere centered in n for each n in normals
    rand_hemisphere_dirs = rand_hemisphere_dir(N, normals) # (..., N, 3)
    # evaluate SH coefficients of env light
    light = eval_sh(envlight.sh_degree, envlight.base.transpose(0,1), rand_hemisphere_dirs)
    light = torch.minimum(light, torch.zeros_like(light))
    # average light values over number of samples
    avg_light_per_normal = torch.mean(light, dim = 1)
    # average light values over normals
    avg_light = torch.mean(avg_light_per_normal, dim = 0)
    # take squared 2 norm
    l2_norm = ((avg_light - torch.zeros_like(avg_light))**2).mean()
    return l2_norm


def envlight_loss2(envlight: EnvironmentLight, normals: torch.Tensor, roughness: torch.Tensor, N: int = 1000, specular=True):
    """
    Regularization on environment lighting coefficients: both diffuse and specular irradiance should belong to R+.
    If the number of normals vectors is greater than subset_size=100, extraxct a random subset.
    Restrict roughness values accordingly.
    Args:
        envlight: environment lighting
        normals: normal vectors of shape L x 3
        roughness: surface roughness values of shape L x 3
        N: number of directions samples
    """
    assert len(normals.shape) == 2 and normals.shape[-1] == 3 , "error: normals must have size  L X 3"
    assert len(roughness.shape) == 2 and roughness.shape[-1] == 1 , "error: roughness must have size  L X 1"
    assert normals.shape[0] == roughness.shape[0] , "normals and roughness must have same batch size"

    normals_subset_size = 1000
    if normals.shape[0] > normals_subset_size:
        rand_subset = random.sample(range(0, normals_subset_size), normals_subset_size)
        normals = normals[rand_subset]
        roughness = roughness[rand_subset]

    # get diffuse irradiance for the given normal vectors
    diffuse_irrad = envlight.get_diffuse_irradiance(normals)
    diffuse_irrad = torch.nn.functional.softplus(diffuse_irrad)
    # diffuse_irrad = torch.minimum(diffuse_irrad, torch.zeros_like(diffuse_irrad)) 
    # average diff light over normals
    avg_diff_per_normal = torch.mean(diffuse_irrad, dim = 0)
    # take squared 2 norm
    l2_norm_diff = ((avg_diff_per_normal- torch.zeros_like(avg_diff_per_normal))**2).mean()
    if specular:
        # generate N random samples directions in the hemisphere centered in n for each n in normals
        rand_hemisphere_dirs = rand_hemisphere_dir(N, normals) # (L, N, 3)
        # get corresponding reflection directions
        reflection_dirs =  util.safe_normalize(util.reflect(rand_hemisphere_dirs.unsqueeze(-1),  normals.repeat(rand_hemisphere_dirs.shape[1], 1 , 1).transpose(0,1).unsqueeze(-1)))
        # get specular irradiance
        specular_light_sh = envlight.get_specular_irradiance(roughness).transpose(1,2) # (100, 3, 25)
        # adjust dims
        specular_light_sh = specular_light_sh.repeat(rand_hemisphere_dirs.shape[1], 1 , 1, 1).transpose(0,1)
        # evaluate SH coefficients for both diffuse and specular radiance
        specular_irrad = eval_sh(envlight.sh_degree, specular_light_sh, reflection_dirs.squeeze())
        specular_irrad = torch.nn.functional.softplus(specular_irrad)
        # specular_irrad = torch.minimum(specular_irrad, torch.zeros_like(specular_irrad)) 
        # average spec light values over number of samples
        avg_spec_per_normal = torch.mean(specular_irrad, dim = 1)
        # average spec light values over normals
        avg_spec = torch.mean(avg_spec_per_normal, dim = 0)
        # take squared 2 norm
        l2_norm_spec = ((avg_spec - torch.zeros_like(avg_spec))**2).mean()
        return l2_norm_diff + l2_norm_spec
    else:
        return l2_norm_diff
    

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
