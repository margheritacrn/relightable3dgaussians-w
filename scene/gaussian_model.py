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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_const_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_minimum_axis, flip_align_view, get_uniform_points_on_sphere_fibonacci
from utils.graphics_utils import getWorld2View2
import open3d as o3d


class GaussianModel:
    def __init__(self, sh_degree : int):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  

        self._xyz = torch.empty(0)
        self._albedo = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

         
        self._normal = torch.empty(0)
        self._normal2 = torch.empty(0)
        self._specular = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metalness = torch.empty(0)

   
        

        self.specular_activation = torch.sigmoid
        self.metalness_activation = torch.sigmoid
        self.albedo_activation = torch.sigmoid
        self.roughness_activation = torch.sigmoid
        self.roughness_bias = 0.
        self.default_roughness = 0.6

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
        return self._xyz


    @property
    def get_features(self):
        albedo = self._albedo
        features_rest = self._features_rest
        return torch.cat((albedo, features_rest), dim=1)


    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    #TODO. remove return_delta, it's here not to edit gaussian_renderer module.
    def get_normal(self, dir_pp_normalized=None, return_delta=False, normalize=False):
        normal_axis = self.get_minimum_axis
        normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)
        if normalize:
            normal_axis = torch.nn.functional.normalize(normal_axis, p=2, dim=1)
        if return_delta:
            return normal_axis, normal_axis
        else:
            return normal_axis


    def get_normal_gshader(self, dir_pp_normalized=None, return_delta=False, normalize=False):
        normal_axis = self.get_minimum_axis
        normal_axis = normal_axis
        normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
        delta_normal1 = self._normal  # (N, 3) 
        delta_normal2 = self._normal2 # (N, 3) 
        delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) # (N, 3, 2)
        idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) # (N, 3, 1)
        delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) # (N, 3)
        normal = delta_normal + normal_axis 
        normal = normal/normal.norm(dim=1, keepdim=True) # (N, 3)
        if normalize:
            normal_axis = torch.nn.functional.normalize(normal_axis, p=2, dim=1)
        if return_delta:
            return normal, delta_normal
        else:
            return normal


    @property
    def get_specular(self):
        return self.specular_activation(self._specular)


    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)


    @property
    def get_metalness(self):
        return self.metalness_activation(self._metalness)


    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness + self.roughness_bias)


    @property
    def get_features_rest(self):
        return self._features_rest


    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, build_rotation(self.get_rotation))


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # get fused color in RGB format
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3)).float().cuda()
        features[:, :3 ] = fused_color
        features[:, 3: ] = 0.0


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        #TODO: check here
        # self._albedo = nn.Parameter(features[:,:3].contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,3:].contiguous().requires_grad_(True))
        normals = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
        normals2 = np.copy(normals)

        self._normal = nn.Parameter(torch.from_numpy(normals).to(self._xyz.device).requires_grad_(True))
        specular_len = 3 
        self._albedo = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True))
        self._specular = nn.Parameter(torch.ones((fused_point_cloud.shape[0], specular_len), device="cuda").requires_grad_(True))
        self._metalness = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(self.default_roughness*torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    @torch.no_grad()
    def get_sky_xyz(self, num_points: int, cameras):
        """Adapted from https://arxiv.org/abs/2407.08447"""
        points = get_uniform_points_on_sphere_fibonacci(num_points)
        points = points.to("cuda")
        mean = self._xyz.mean(0)[None]
        sky_distance = torch.quantile(torch.linalg.norm(self._xyz - mean, 2, -1), 0.97) * 10
        points = points * sky_distance
        points = points + mean
        gmask = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        for cam in cameras:
            uv = cam.project(points[torch.logical_not(gmask)])
            mask = torch.logical_not(torch.isnan(uv).any(-1))
            # Only top 2/3 of the image
            assert cam.image_width is not None and cam.image_height is not None
            mask = torch.logical_and(mask, uv[..., -1] < 2/3 * cam.image_height)
            gmask[torch.logical_not(gmask)] = torch.logical_or(gmask[torch.logical_not(gmask)], mask)

        return points[gmask], sky_distance



    def extend_with_sky_gaussians(self, num_points: int, cameras):
        sky_xyz, _ = self.get_sky_xyz(num_points, cameras)
        print(f"Adding {sky_xyz.shape[0]} sky Gaussians")

        self._xyz = nn.Parameter(torch.cat([self._xyz, sky_xyz], dim=0).requires_grad_(True))

        sky_albedo = 0.5*torch.ones((sky_xyz.shape[0], 3), device=self._albedo.device, requires_grad=True)
        self._albedo = nn.Parameter(torch.cat([self._albedo, sky_albedo], dim=0))

        sky_specular = torch.zeros((sky_xyz.shape[0], 3), device=self._specular.device, requires_grad=True)
        self._specular = nn.Parameter(torch.cat([self._specular, sky_specular], dim=0))

        sky_metalness = torch.zeros((sky_xyz.shape[0], 1), device=self._metalness.device, requires_grad=True)
        self._metalness = nn.Parameter(torch.cat([self._metalness, sky_metalness], dim=0))

        sky_roughness = torch.zeros((sky_xyz.shape[0], 1), device=self._roughness.device, requires_grad=True)
        self._roughness = nn.Parameter(torch.cat([self._roughness, sky_roughness], dim=0))

        sky_normals = torch.from_numpy(np.zeros((sky_xyz.shape[0], 3), dtype=np.float32)).to(self._normal.device).requires_grad_(True)
        self._normal = nn.Parameter(torch.cat([self._normal, sky_normals], dim=0))

        sky_normals2 = torch.zeros_like(sky_normals)
        self._normal2 = nn.Parameter(torch.cat([self._normal2, sky_normals2], dim=0))

        sky_opacity = torch.ones((sky_xyz.shape[0], 1), device=self._opacity.device, requires_grad=True)
        self._opacity = nn.Parameter(torch.cat([self._opacity, sky_opacity]))

        dist2 = torch.clamp_min(distCUDA2(sky_xyz.float().cuda()), 0.0000001)
        sky_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        sky_rots = torch.zeros((sky_xyz.shape[0], 4), device=self._rotation.device)
        sky_rots[:, 0] = 1
        self._rotation = nn.Parameter(torch.cat([self._rotation, sky_rots.requires_grad_(True)], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, sky_scales.requires_grad_(True)]))
        sky_max_radii2D = torch.zeros((sky_xyz.shape[0]), device=self.max_radii2D.device)
        self.max_radii2D = nn.Parameter(torch.cat([self.max_radii2D, sky_max_radii2D]))

        #TODO: remove feature_rest
        sky_f_rest = torch.zeros((sky_xyz.shape[0], 0), device=self._features_rest.device, requires_grad=True)
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, sky_f_rest]))

    #TODO: remove self.normal2 and fetures_rest
    def training_setup(self, envlight, embeddings, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._albedo], 'lr': training_args.feature_lr, "name": "albedo"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"},
            {'params': [self._metalness], 'lr': training_args.metalness_lr, "name": "metalness"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': envlight.parameters(), 'lr': training_args.envlight_sh_lr, 'weight_decay': training_args.envlight_sh_wd, "name": 'envlight'},
            {'params': embeddings.parameters(), 'lr': training_args.embedding_lr, "name": 'embeddings'}
        ]
        self._normal2.requires_grad_(requires_grad=False)
        l.extend([
            {'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"},
        ])

        self.optimizer = torch.optim.Adam(l, lr=0.01, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def training_setup_relit3DGW(self, training_args: dict):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._albedo], 'lr': training_args.feature_lr, "name": "albedo"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"},
            {'params': [self._metalness], 'lr': training_args.metalness_lr, "name": "metalness"},
        ]
        self._normal.requires_grad_(requires_grad=False)
        self._normal2.requires_grad_(requires_grad=False)
        l.extend([
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"},
        ])

        # self.optimizer = torch.optim.Adam(l, lr=0.01, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        return l


    def training_setup_SHoptim(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.f_rest_scheduler_args = get_const_lr_func(training_args.feature_lr / 20.0)


    def set_optimizer(self, optimizer: torch.optim):
        self.optimizer = optimizer


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr


    def construct_list_of_attributes(self, viewer_fmt=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        """
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        """
        l.extend(['nx2', 'ny2', 'nz2'])
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('roughness')
        l.append('metalness')
        for i in range(self._specular.shape[1]):
            l.append('specular{}'.format(i))
        return l


    def save_ply(self, path, viewer_fmt=False):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        normals2 = self._normal2.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        f_rest = self._features_rest.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metalness = self._metalness.detach().cpu().numpy()
        specular = self._specular.detach().cpu().numpy()
        
        if viewer_fmt:
            albedo = 0.5 + (0.5*normals)
            f_rest = np.zeros((f_rest.shape[0], 45))
            normals = np.zeros_like(normals)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(viewer_fmt)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if not viewer_fmt:
            attributes = np.concatenate((xyz, normals, albedo, normals2, opacities, scale, rotation, roughness, metalness, specular), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, albedo, opacities, scale, rotation, roughness, metalness, specular), axis=1)
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

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        features_extra = np.zeros((xyz.shape[0], 3**2))
        if len(extra_f_names)==3**2:
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_extra = features_extra.reshape((features_extra.shape[0], 2, 3))
            features_extra = features_extra.swapaxes(1,2)

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

        specular_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("specular")]
        specular = np.zeros((xyz.shape[0], len(specular_names)))
        for idx, attr_name in enumerate(specular_names):
            specular[:, idx] = np.asarray(plydata.elements[0][attr_name])

        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)
        normal2 = np.stack((np.asarray(plydata.elements[0]["nx2"]),
                        np.asarray(plydata.elements[0]["ny2"]),
                        np.asarray(plydata.elements[0]["nz2"])),  axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metalness = nn.Parameter(torch.tensor(metalness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._specular = nn.Parameter(torch.tensor(specular, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "envlight_sh" or group["name"] == "embeddings":
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


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "envlight_sh" or group["name"] == "embeddings":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._albedo = optimizable_tensors["albedo"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._roughness = optimizable_tensors["roughness"]
        self._metalness = optimizable_tensors["metalness"]
        self._specular = optimizable_tensors["specular"]
        self._normal = optimizable_tensors["normal"]
        self._normal2 = optimizable_tensors["normal2"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "envlight_sh" or group["name"] == "embeddings" :
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


    def densification_postfix(self, new_xyz, new_albedo, new_features_rest, new_opacities, new_scaling, new_rotation, \
                              new_roughness, new_metalness, new_specular, new_normal, new_normal2):
        d = {"xyz": new_xyz,
        "albedo": new_albedo,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "roughness": new_roughness,
        "metalness": new_metalness,
        "specular" : new_specular,
        "normal" : new_normal,
        "normal2" : new_normal2}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._albedo = optimizable_tensors["albedo"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._roughness = optimizable_tensors["roughness"]
        self._metalness = optimizable_tensors["metalness"]
        self._specular = optimizable_tensors["specular"]
        self._normal = optimizable_tensors["normal"]
        self._normal2 = optimizable_tensors["normal2"]
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

        stds = self.get_scaling[selected_pts_mask].repeat(N,1) # (n,3)
        stds, sorted_idx = torch.sort(stds, dim=1, descending=True)
        stds = stds[:,:2]
        means =torch.zeros((stds.size(0), 2),device="cuda") # 2 for norml
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) # (n*N, 3, 3)
        rots = torch.gather(rots, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
        new_xyz = torch.bmm(rots[:,:,:2], samples.unsqueeze(-1)).squeeze(-1) +  self.get_xyz[selected_pts_mask].repeat(N, 1)
        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1) # sample from gaussian dist
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_metalness = self._metalness[selected_pts_mask].repeat(N,1)
        new_specular = self._specular[selected_pts_mask].repeat(N,1)
        new_normal = self._normal[selected_pts_mask].repeat(N,1)
        new_normal2 = self._normal2[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_albedo, new_features_rest, new_opacity, new_scaling, new_rotation, 
                                   new_roughness, new_metalness, new_specular, new_normal, new_normal2)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent, viewing_dir):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if torch.sum(selected_pts_mask) == 0:
            return
        new_xyz = self._xyz[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_metalness = self._metalness[selected_pts_mask]
        new_specular = self._specular[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_normal2 = self._normal2[selected_pts_mask]

        with torch.no_grad():
            normals = self.get_normal(dir_pp_normalized=viewing_dir, normalize=True)
        new_xyz += grads[selected_pts_mask]*normals[selected_pts_mask]


        self.densification_postfix(new_xyz, new_albedo, new_features_rest, new_opacities, new_scaling, new_rotation, 
                                   new_roughness, new_metalness, new_specular, new_normal, new_normal2)


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
