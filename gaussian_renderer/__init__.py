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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image, depth_to_normal
from scene.NVDIFFREC.util import safe_normalize
from utils.general_utils import flip_align_view
# from scene.NVDIFFREC import extract_env_map
from scene.NVDIFFREC.light import EnvironmentLight
import numpy as np


def rendered_world2cam(viewpoint_cam, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1,2,0).reshape(-1, 3) # (HxW, 3)
    normal_cam = torch.cat([normal_world, torch.ones_like(normal_world[...,0:1])], axis=-1) @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0,1))[...,:3]
    normal_cam = normal_cam.reshape(H, W, 3).permute(2,0,1) # (H, W, 3)
    
    background = bg_color[...,None,None]
    normal_cam = normal_cam*alpha[None,...] + background*(1. - alpha[None,...])

    return normal_cam


def render_surf_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[..., None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref


def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)


def get_shaded_colors(envlight: EnvironmentLight, pos: torch.tensor, view_pos: torch.tensor, normal: torch.tensor=None, albedo: torch.tensor=None, 
                       roughness:torch.tensor=None, metalness:torch.tensor=None, specular:bool=True):
    if metalness is not None:
        metalness = metalness[None, None, ...]
    colors_precomp, brdf_pkg = envlight.shade(gb_pos=pos[None, None, ...], gb_normal=normal[None, None, ...], albedo=albedo[None, None, ...],
                            view_pos=view_pos[None, None, ...], kr=roughness[None, None, ...], km=metalness, specular=specular)
    return colors_precomp, brdf_pkg


def render(viewpoint_camera, pc : GaussianModel, envlight : EnvironmentLight, sky_sh: torch.tensor, sky_sh_degree: int, pipe,  bg_color : torch.Tensor, scaling_modifier = 1.0, debug=True,
           specular=True, fix_sky=False, normals_in_world_space=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=-1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    sky_gaussians_mask = pc.get_is_sky.squeeze() # (N)
    positions = pc.get_xyz # (N, 3)
    albedo = pc.get_albedo # (N, 3)
    roughness = pc.get_roughness # (N, 1)
    metalness = pc.get_metalness # (N,1)

    # Render under camera viewpoint
    view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3) 
    dir_pp = (pc.get_xyz - view_pos)
    dir_pp_normalized = safe_normalize(dir_pp) # (N, 3)
    normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized) # (N, 3)
    if normals_in_world_space:
        normal = pc.get_normal() # (N, 3)

    colors_precomp, diffuse_color, specular_color, sky_color = (torch.zeros(positions.shape[0], 3, dtype=torch.float32, device="cuda") for _ in range(4))

    # Compute color for the foreground Gaussians
    color_fg_gaussians, brdf_pkg = get_shaded_colors(envlight=envlight, pos=positions[~sky_gaussians_mask],
                                                          view_pos=view_pos[~sky_gaussians_mask], normal=normal[~sky_gaussians_mask],
                                                          albedo=albedo[~sky_gaussians_mask],
                                                          roughness=roughness[~sky_gaussians_mask], metalness=metalness[~sky_gaussians_mask],
                                                          specular=specular)
    colors_precomp[~sky_gaussians_mask] = color_fg_gaussians.squeeze()

    # Compute color for the sky (background) Gaussians
    if fix_sky:
        colors_precomp[sky_gaussians_mask] = torch.ones_like(positions[sky_gaussians_mask])
    else:
        sky_sh2rgb = eval_sh(sky_sh_degree, sky_sh.transpose(1,2), dir_pp_normalized[sky_gaussians_mask])
        color_sky_gaussians = torch.clamp_min(sky_sh2rgb + 0.5, 0.0)
        colors_precomp[sky_gaussians_mask] = color_sky_gaussians

    diffuse_color[sky_gaussians_mask] = torch.zeros_like(colors_precomp[sky_gaussians_mask])
    diffuse_color[~sky_gaussians_mask] = brdf_pkg['diffuse'].squeeze()
    specular_color[sky_gaussians_mask] = torch.zeros_like(colors_precomp[sky_gaussians_mask])
    specular_color[~sky_gaussians_mask] = brdf_pkg['specular'].squeeze()

    sky_color[sky_gaussians_mask] = colors_precomp[sky_gaussians_mask]
    sky_color[~sky_gaussians_mask] = torch.zeros_like(colors_precomp[~sky_gaussians_mask])


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
    }
    render_extras = {"diffuse_color": diffuse_color,
            "specular_color": specular_color}

    # Render depth and normals
    # Get Gaussians depth as z coordinate of their position in camera space
    depth = pc.get_depth(viewpoint_camera)
    depth = depth.repeat(1,3)
    render_extras.update({"depth": depth})


    normal = 0.5*normal + 0.5 # range (-1, 1) -> (0, 1)
    normal[sky_gaussians_mask] = 1
    render_extras.update({"normal": normal})

    if debug:
        render_extras.update({
            "sky_color": sky_color,
            "roughness": roughness.repeat(1, 3),
            "metalness": metalness.repeat(1, 3),
            "albedo": albedo})

    out_extras = {}
    for k in render_extras.keys():
        if render_extras[k] is None: continue
        image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = render_extras[k],
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        out_extras[k] = image

        if k == "normal" :
            out_extras[k] = (out_extras[k] - 0.5) * 2. # range (0, 1) -> (-1, 1)
    torch.cuda.empty_cache()

    # Render alpha
    # Rasterize visible Gaussians to alpha mask image. 
    raster_settings_alpha = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=-1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
    )
    rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
    alpha = torch.ones_like(means3D) 
    out_extras["alpha"] =  rasterizer_alpha(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = alpha,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)[0]


    # Get surface normal from depth map.
    # out_extras["normal_ref"] = render_surf_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0], bg_color= bg_color, alpha=out_extras['alpha'][0])
    out_extras["normal_ref"]  = depth_to_normal(viewpoint_camera, out_extras['depth'][0].unsqueeze(0))
    out_extras["normal_ref"] = out_extras["normal_ref"].permute(2,0,1)
    out_extras["normal_ref"] = (out_extras["normal_ref"] * (out_extras["alpha"]).detach())
    out_extras["normal_ref"] = out_extras["normal_ref"] * (viewpoint_camera.sky_mask.cuda().squeeze()) + torch.ones_like(out_extras["normal_ref"]) * (1-viewpoint_camera.sky_mask.cuda().squeeze())
    normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])
    out.update(out_extras)
    return out
