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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_const_lr_func, insert_zeros, cartesian_to_polar_coord
from utils.camera_utils import get_scene_center
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_minimum_axis, flip_align_view, sample_points_on_unit_hemisphere
from utils.graphics_utils import getWorld2View2
import open3d as o3d
import math


class GaussianModel:
    def __init__(self):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm


        self._xyz = torch.empty(0)
        self._albedo = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self._is_sky = torch.empty(0)
        self._sky_radius = torch.empty(0)
        self._sky_gauss_center = torch.empty(0)
        self._sky_angles = torch.empty(0)


        self._roughness = torch.empty(0)
        self._metalness = torch.empty(0)


        self.material_properties_activation = torch.sigmoid
        self.default_roughness = 0.6
        self.default_albedo = 1.0
        self.default_metalness = 0.1

        self.sky_angles_activation = torch.sigmoid


        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)


    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)


    @property
    def get_xyz(self):
        if torch.sum(self._is_sky) == 0:
            return self._xyz
        else:
            sky_xyz = self.get_sky_xyz
            all_xyz = torch.empty((sky_xyz.shape[0] + self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device)
            all_xyz[~self._is_sky.squeeze()] = self._xyz
            all_xyz[self._is_sky.squeeze()] = sky_xyz
            return all_xyz


    @property
    def get_sky_xyz(self):
        sky_angles = self.get_sky_angles_clamp
        # In COLMAP coordinate system
        x = torch.sin(sky_angles[...,0]) * torch.sin(sky_angles[...,1])
        y = -torch.cos(sky_angles[...,0])
        z = torch.sin(sky_angles[...,0]) * torch.cos(sky_angles[...,1])
        sky_xyz = self._sky_radius * torch.stack([x, y, z], dim=-1) +  self._sky_gauss_center.squeeze()
        return sky_xyz


    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_normal(self, dir_pp_normalized=None, normalize=False):
        normal_axis = get_minimum_axis(self.get_scaling, build_rotation(self.get_rotation))
        if dir_pp_normalized is not None:
            normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)
        if normalize:
            normal_axis = torch.nn.functional.normalize(normal_axis, p=2, 
                                                        dim=1)
        return normal_axis

    def get_depth(self, viewpoint_camera):
        p_hom = torch.cat([self.get_xyz, torch.ones_like(self.get_xyz[...,:1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
        p_view = p_view[...,:3,:]
        depth = p_view.squeeze()[...,2:3]
        return depth


    @property
    def get_albedo(self):
        return torch.where(self._is_sky, self._albedo, self.material_properties_activation(self._albedo))

    

    @property
    def get_metalness(self):
        return torch.where(self._is_sky, self._metalness, self.material_properties_activation(self._metalness))



    @property
    def get_roughness(self):
        return torch.where(self._is_sky, self._roughness, self.material_properties_activation(self._roughness))
    
    
    @property
    def get_sky_radius(self):
        return self._sky_radius


    @property
    def get_sky_gauss_center(self):
        return self._sky_gauss_center

    
    @property
    def get_sky_angles_clamp(self):
        # theta admitted range: [0, pi/2], phi admitted range: [-pi/2, pi/2]
        theta_mask = (self._sky_angles[self._is_sky.squeeze()][...,0] < 0) | (self._sky_angles[self._is_sky.squeeze()][...,0] > torch.pi/2)
        phi_mask = (self._sky_angles[self._is_sky.squeeze()][...,1] < -torch.pi/2) | (self._sky_angles[self._is_sky.squeeze()][...,1] > torch.pi/2)

        sky_theta = torch.where(theta_mask, torch.clamp(self._sky_angles[self._is_sky.squeeze()][...,0], 0, torch.pi/2),
                                self._sky_angles[self._is_sky.squeeze()][...,0])
        sky_phi = torch.where(phi_mask, torch.clamp(self._sky_angles[self._is_sky.squeeze()][...,1], -torch.pi/2, torch.pi/2),
                              self._sky_angles[self._is_sky.squeeze()][...,1])

        return torch.cat((sky_theta.unsqueeze(1), sky_phi.unsqueeze(1)), dim=1)


    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, build_rotation(self.get_rotation))


    @property
    def get_is_sky(self):
        return self._is_sky


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

 
        self._albedo = nn.Parameter(self.default_albedo * torch.ones((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True))
        self._metalness = nn.Parameter(self.default_metalness * torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._is_sky =  torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.bool, device="cuda")

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    
    @torch.no_grad()
    def get_sky_xyz_init(self, cameras):
        """Adapted from https://arxiv.org/abs/2407.08447"""
        mean = self._xyz.mean(0)[None]
        sky_distance = torch.quantile(torch.linalg.norm(self._xyz - mean, 2, -1), 0.99)
        scene_center = torch.tensor(get_scene_center(cameras), dtype=torch.float32, device="cuda").T
        num_sky_points = int(5000 * sky_distance.item())
        points = sample_points_on_unit_hemisphere(num_sky_points)
        points = points.to("cuda")
        points = points * sky_distance
        points = points + scene_center
        gmask = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        for cam in cameras:
            uv = cam.project(points[torch.logical_not(gmask)])
            mask = torch.logical_not(torch.isnan(uv).any(-1))
            # Only top 2/3 of the image
            assert cam.image_width is not None and cam.image_height is not None
            mask = torch.logical_and(mask, uv[..., -1] < 2/3 * cam.image_height)
            gmask[torch.logical_not(gmask)] = torch.logical_or(gmask[torch.logical_not(gmask)], mask)

        return points[gmask], sky_distance, scene_center


    def augment_with_sky_gaussians(self, cameras):
        sky_xyz, sky_distance, sky_gauss_center = self.get_sky_xyz_init(cameras)
        self._sky_gauss_center = sky_gauss_center
        print(f"Adding {sky_xyz.shape[0]} sky Gaussians")
        # Initialize polar coordinates:
        self._sky_radius = nn.Parameter(torch.tensor(sky_distance, dtype=torch.float32, device="cuda").requires_grad_(True))
        sky_angles = cartesian_to_polar_coord(sky_xyz, self._sky_gauss_center.squeeze(), self._sky_radius)#.detach())
        self._sky_angles = nn.Parameter(torch.cat((torch.full((self._xyz.shape[0], 2), torch.inf, device="cuda"), sky_angles), dim=0).requires_grad_(True))


        sky_albedo = torch.ones((sky_xyz.shape[0], 3), device=self._albedo.device, requires_grad=True)
        self._albedo = nn.Parameter(torch.cat([self._albedo, sky_albedo], dim=0))


        sky_metalness = torch.zeros((sky_xyz.shape[0], 1), device=self._metalness.device, requires_grad=True)
        self._metalness = nn.Parameter(torch.cat([self._metalness, sky_metalness], dim=0))

        sky_roughness = torch.zeros((sky_xyz.shape[0], 1), device=self._roughness.device, requires_grad=True)
        self._roughness = nn.Parameter(torch.cat([self._roughness, sky_roughness], dim=0))


        sky_opacity =  inverse_sigmoid(0.1 * torch.ones((sky_xyz.shape[0], 1), dtype=torch.float, device="cuda")) # inverse_sigmoid(0.95 * torch.ones((sky_xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.cat([self._opacity, sky_opacity]))

        self._is_sky = torch.cat((self._is_sky, torch.ones((sky_xyz.shape[0], 1), dtype=torch.bool, device=self._is_sky.device)), dim=0)


        dist2 = torch.clamp_min(distCUDA2(sky_xyz.float().cuda()), 0.0000001)
        sky_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        sky_rots = torch.zeros((sky_xyz.shape[0], 4), device=self._rotation.device)
        sky_rots[:, 0] = 1
        self._rotation = nn.Parameter(torch.cat([self._rotation, sky_rots.requires_grad_(True)], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, sky_scales.requires_grad_(True)]))
        sky_max_radii2D = torch.zeros((sky_xyz.shape[0]), device=self.max_radii2D.device)
        self.max_radii2D = nn.Parameter(torch.cat([self.max_radii2D, sky_max_radii2D]))


    def training_setup(self, training_args: dict):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._albedo], 'lr': training_args.feature_lr, "name": "albedo"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._metalness], 'lr': training_args.metalness_lr, "name": "metalness"},
            #'params': [self._sky_radius], 'lr': training_args.sky_radius_lr, "name": "sky_radius"},
            {'params': [self._sky_angles], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "sky_angles"},
        ]


        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        return l


    def set_optimizer(self, optimizer: torch.optim):
        self.optimizer = optimizer


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz" or param_group["name"] == "sky_angles":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('roughness')
        l.append('metalness')
        l.append('is_sky')
        if torch.sum(self._is_sky) > 0:
            l.append('sky_radius')
            for i in range(self._sky_gauss_center.shape[1]):
                l.append('sky_gauss_center_{}'.format(i))
            for i in range(self._sky_angles.shape[1]):
                l.append('sky_angles_{}'.format(i))
        return l


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metalness = self._metalness.detach().cpu().numpy()
        is_sky = self._is_sky.cpu().numpy()
        sky_radius = self._sky_radius.repeat(xyz.shape[0],1).cpu().numpy()
        sky_gauss_center = self._sky_gauss_center.repeat(xyz.shape[0],1).cpu().numpy()
        sky_angles = self._sky_angles.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if torch.sum(self._is_sky) > 0:
            attributes = np.concatenate((xyz, albedo, opacities, scale, rotation, roughness, metalness, is_sky, sky_radius, sky_gauss_center, sky_angles), axis=1)
        else:
            attributes = np.concatenate((xyz, albedo, opacities, scale, rotation, roughness, metalness, is_sky), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        albedo = np.zeros((xyz.shape[0], 3))
        albedo[:, 0] = np.asarray(plydata.elements[0]["albedo_0"])
        albedo[:, 1] = np.asarray(plydata.elements[0]["albedo_1"])
        albedo[:, 2] = np.asarray(plydata.elements[0]["albedo_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        metalness = np.asarray(plydata.elements[0]["metalness"])[..., np.newaxis]
    
        is_sky = np.asarray(plydata.elements[0]["is_sky"])[..., np.newaxis]
        
        sky_radius = np.asarray(plydata.elements[0]["sky_radius"])[0]
        sky_gauss_center_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sky_gauss_center_")]
        sky_gauss_center = np.zeros((xyz.shape[0], len(sky_gauss_center_names)))
        for idx, attr_name in enumerate(sky_gauss_center_names):
            sky_gauss_center[:, idx] = np.asarray(plydata.elements[0][attr_name])
        sky_gauss_center = sky_gauss_center[0]
        sky_angles_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sky_angles_")]
        sky_angles = np.zeros((xyz.shape[0], len(sky_angles_names)))
        for idx, attr_name in enumerate(sky_angles_names):
            sky_angles[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = xyz[is_sky.squeeze() == 0]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metalness = nn.Parameter(torch.tensor(metalness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._is_sky = torch.tensor(is_sky, dtype=torch.bool, device="cuda")
        self._sky_radius = nn.Parameter(torch.tensor(sky_radius, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sky_gauss_center = nn.Parameter(torch.tensor(sky_gauss_center, dtype=torch.float, device="cuda"))
        self._sky_angles = torch.tensor(sky_angles, dtype=torch.float, device="cuda")


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["sky_sh", "envlight_sh", "embeddings", "sky_radius", "shadow_mlp", "mlp"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask, mask_xyz):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["sky_sh", "envlight_sh", "embeddings", "sky_radius", "shadow_mlp", "mlp"]:
                continue
            if group["name"] == "xyz":
                mask_prune = mask_xyz
            else:
                mask_prune = mask
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask_prune]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask_prune]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask_prune].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask_prune].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        valid_xyz_mask = valid_points_mask[~self._is_sky.squeeze()]
        optimizable_tensors = self._prune_optimizer(valid_points_mask, valid_xyz_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._albedo = optimizable_tensors["albedo"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._roughness = optimizable_tensors["roughness"]
        self._metalness = optimizable_tensors["metalness"]
        self._sky_angles = optimizable_tensors["sky_angles"]
        self._is_sky = self._is_sky[valid_points_mask]


        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"]in ["sky_sh", "envlight_sh", "embeddings", "sky_radius", "shadow_mlp", "mlp"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, new_xyz, new_albedo, new_opacities, new_scaling, new_rotation, \
                              new_roughness, new_metalness, new_is_sky, new_sky_angles):
        d = {
        "albedo": new_albedo,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "roughness": new_roughness,
        "metalness": new_metalness,
        "sky_angles": new_sky_angles}
        if new_xyz is not None:
            d.update({"xyz": new_xyz})

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        if "xyz" in optimizable_tensors.keys():
            self._xyz = optimizable_tensors["xyz"]
        self._albedo = optimizable_tensors["albedo"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._roughness = optimizable_tensors["roughness"]
        self._metalness = optimizable_tensors["metalness"]
        self._sky_angles = optimizable_tensors["sky_angles"]
        self._is_sky = torch.cat((self._is_sky, new_is_sky), dim=0)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if torch.sum(selected_pts_mask) == 0:
            return
        """
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        """
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1) # (n,3)
        stds, sorted_idx = torch.sort(stds, dim=1, descending=True)
        stds = stds[:,:2]
        means = torch.zeros((stds.size(0), 2),device="cuda")
        # Get 2D samples from standard Gaussian
        samples = torch.normal(mean=means, std=stds)
        # Project samples in 3D such that the centers lie in the ellipse defined by the two greatest axis
        samples = insert_zeros(samples, sorted_idx[..., -1])
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) # (n*N, 3, 3)
        rots = torch.gather(rots, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) +  self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_metalness = self._metalness[selected_pts_mask].repeat(N,1)
        new_is_sky = self._is_sky[selected_pts_mask].repeat(N,1)
        # Project sampled positions for sky Gaussians on the sphere
        new_xyz[new_is_sky.squeeze()] = self._sky_gauss_center + self._sky_radius * (new_xyz[new_is_sky.squeeze()] - self._sky_gauss_center)/torch.norm(new_xyz[new_is_sky.squeeze()] - self._sky_gauss_center, dim=1)[..., None]
        new_sky_angles = torch.where(new_is_sky, cartesian_to_polar_coord(new_xyz, self._sky_gauss_center.squeeze(), self._sky_radius),
                                     self._sky_angles[selected_pts_mask].repeat(N,1))
    
        self.densification_postfix(new_xyz[~(new_is_sky.squeeze())], new_albedo, new_opacity, new_scaling, new_rotation, 
                                   new_roughness, new_metalness, new_is_sky, new_sky_angles)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent, viewing_dir):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        selected_pts_mask_non_sky = torch.where(self.get_is_sky.squeeze(), False, selected_pts_mask)
        if torch.sum(selected_pts_mask) == 0:
            return
        if torch.sum(selected_pts_mask_non_sky) == 0:
            new_xyz = None
        else:
            new_xyz = self.get_xyz[selected_pts_mask_non_sky]
        new_albedo = self._albedo[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_metalness = self._metalness[selected_pts_mask]
        new_sky_angles = self._sky_angles[selected_pts_mask]
        new_is_sky = self._is_sky[selected_pts_mask]



        if new_xyz is not None:
            with torch.no_grad():
                normals = self.get_normal(dir_pp_normalized=viewing_dir, normalize=True)
            new_xyz = new_xyz + grads[selected_pts_mask_non_sky] * normals[selected_pts_mask_non_sky]


        self.densification_postfix(new_xyz, new_albedo, new_opacities, new_scaling, new_rotation, 
                                   new_roughness, new_metalness, new_is_sky, new_sky_angles)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, viewing_dir):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, viewing_dir)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state

    
    def get_scene_extent(self, cams_info):
        points = self.get_xyz
        points = points.detach().cpu().numpy()
        cam_centers = []
        for cam in cams_info:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        distances = np.linalg.norm(avg_cam_center.transpose() - points, axis=0, keepdims=True)
        scene_extent =  np.mean(distances)
        return scene_extent
