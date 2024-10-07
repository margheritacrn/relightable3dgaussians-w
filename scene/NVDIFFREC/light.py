# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
from . import util
from . import renderutils as ru
from utils.general_utils import get_homogeneous
from utils.sh_utils import gauss_weierstrass_kernel, eval_sh
#NOTE: I also need to load envlights that don't need training, so I should have base attribute not directly initialized to LightNet object
#TODO: add dimensionality control to load_env function
#TODO: check consistency of SH coefficients dimension of envlight, should be 3xnum_sh_coeffs. Also, add error handling for SH coeffs > 25 (i.e deg > 4)
#TODO: decide what to do with specularity: in case is not used it must be removed from shade's parameters
#TODO: fix squeezing of self.ba

class EnvironmentLight(torch.nn.Module):

    def __init__(self, base: torch.Tensor, base_is_SH: bool, sh_degree : int = 4):
        self.base = base.squeeze(0) # (sh_degree + 1)**2 x 1
        self.base_is_SH = base_is_SH
        self.sh_degree = sh_degree
        self.sh_dim = (sh_degree +1)**2
        self.mtx = None  
        if not self.base_is_SH:
            # convert base (other option cna be that is a cubemap) to SH (degree 2)
            # implement the function
            # update attribute
            self.base_is_SH = True
        # define constant attributes for diffuse irradiance computationn
        self.C1 = 0.429043
        self.C2 = 0.511664
        self.C3 = 0.743125
        self.C4 = 0.886227
        self.C5 = 0.247708
        self.M_i, self.M_j = torch.triu_indices(4,4)


    def clone(self):
        return EnvironmentLight(self.base.clone().detach())


    def get_diffuse_irradiance(self, normal):
        """The function returns diffuse irradiance by performing convolution
        between environment light and cosine term in frequency domain. In the
        SH expansion only terms up to degree 2 are considered.
        The implementation refers to section 3.2 of "An efficient representaiton for Irradiacne Environment Maps
        by Ramamoorthi and Pat Hanrahan."""
        # move normal to homogeneous coordinates:
        normal_h = get_homogeneous(normal) # (N, 4)
        # build symmetric matrix M
        M = torch.zeros((4,4)).cuda()
        triu_entries = torch.zeros(10).cuda()
        envc_sh = self.base.squeeze()
        triu_entries = torch.Tensor([self.C1 *envc_sh[8], self.C1 *envc_sh[4], self.C1 *envc_sh[7], self.C2*envc_sh[3],
                                        -self.C1*envc_sh[8], self.C1*envc_sh[5], self.C2*envc_sh[1],
                                        self.C3*envc_sh[6], self.C2*envc_sh[2], self.C4*envc_sh[0]-self.C5*envc_sh[6]]).cuda()
        M[self.M_i, self.M_j] = triu_entries
        M.T[self.M_i, self.M_j] = triu_entries

        # get diffuse irradiance
        M = M.unsqueeze(0) # (1, 4, 4)
        M = M.repeat(normal_h.shape[0], 1, 1) # (N, 4, 4)
        Mn = torch.matmul(M, normal_h.unsqueeze(-1)) # (N, 4, 4)* (N, 4, 1) = (N, 4, 1)
        diffuse_irradiance = (Mn*normal_h.unsqueeze(-1)).sum(dim=1) # (N, 1) 
        return diffuse_irradiance



    def get_specular_irradiance(self, roughness: torch.Tensor):
        """The function computes specular irradiance by convolving
        the SH coefficients (degree 4) of the environment light with a Gaussian blur kernel of
        standard deviation = roughness. Roughness is assumed to be of dim (N,1).
        """
        # build coefficients of blur kernel in frequency (SH) domain
        gwk_sh = gauss_weierstrass_kernel(roughness, self.sh_degree) # (N, 25)
        gwk_sh = gwk_sh.unsqueeze(-1) # (N, 25, 1)
        # adjust dimensions
        envlight_sh = self.base.unsqueeze(0)   # (1, 25, 1)
        envlight_sh = envlight_sh.repeat(gwk_sh.shape[0], 1, 1) # (N, 25, 1)
        # perform convolution
        spec_irradiance = gwk_sh * envlight_sh # (N, 25, 1)

        return spec_irradiance



    def shade(self, gb_pos, gb_normal, albedo, ks, kr, km, view_pos, specular=True, tone=False):
        """
       The function returns emitted radiance in outgoing direction view_pos. If specular is 
       True a microfacet reflectanc model is assumed, otherwise a Lambertian model. 
        Args:
            gb_pos: world position
            gb_normal. normal vector
            albedo : albedo of the surface, base color
            ks: specularity
            kr: roughness
            km: metalness
            view_pos: viewing direction
            envlight: SH coefficients of environment light
        """
        assert self.base_is_SH, "envlight can be only represented through Spherical Harmonics"

        # (H, W, N, C)
        wo = util.safe_normalize(view_pos - gb_pos)

        diff_col = albedo
        if specular:
            metalness = km
            roughness = kr # (H,W,N,1)

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        diffuse_irradiance = self.get_diffuse_irradiance(nrmvec.squeeze())
        # get diffuse radiance
        shaded_col = diff_col*diffuse_irradiance 
        extras = {"diffuse": diff_col*diffuse_irradiance}

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, nrmvec), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')
            roughness = roughness.squeeze(0).squeeze(0) # (N,1)
            # convovlve self.base with SH coeffs of Gaussian blur kernel of std roughness 
            spec_irradiance = self.get_specular_irradiance(roughness) # (N, 25, 1)
            # transpose for eval_sh
            spec_irradiance = spec_irradiance.transpose(1,2) # (N, 1, 25)
            # compute specular radiance
            spec_radiance = eval_sh(self.sh_degree, spec_irradiance, reflvec.squeeze()) # (N, 3)
            # adjust dimensions
            spec_radiance = spec_radiance[None, None, ...] # (H, W, N, 3)
            # Compute aggregate lighting
            if metalness is None:
                F0 = torch.ones_like(albedo) * 0.04  # (1, H, W, 3)
            else:
                F0 = (1.0 - metalness) * 0.04 + albedo * metalness
            reflectance = F0* fg_lookup[...,0:1] + fg_lookup[...,1:2]
            print(reflectance.shape)
            specular_color = spec_radiance*reflectance
            shaded_col += specular_color
            extras['specular'] = spec_radiance*reflectance
        else: #TODO: remove this else statement
            extras['specular'] = shaded_col

        if tone:
            # apply tone mapping
            rgb = util.aces_film(shaded_col)
        else:
            rgb = torch.sigmoid(shaded_col)

        return rgb, extras


# Load and store envmaps (cubemap-SH representations)
def load_sh_env(envlight_sh: torch.Tensor):
    return EnvironmentLight(envlight_sh, base_is_SH=True)


def load_hdr_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() != ".hdr":
        raise OSError("Unknown envlight extension")
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])
    envlight_sh = get_SH_from_cubemap(cubemap)
    l = EnvironmentLight(envlight_sh, base_is_SH=True)
    return l


def get_SH_from_cubemap(cubemap, sh_degree: int = 2):
    pass


def get_cubemap_fromSH(envlight_sh):
    # use 
    pass

