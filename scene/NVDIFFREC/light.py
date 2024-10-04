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
from utils.sh_utils import gauss_weierstrass_kernel
#NOTE: I also need to load envlights that don't need training, so I should have base attribute not directly initialized to LightNet object
#TODO: add dimensionality control to load_env function
#TODO: check consistency of SH coefficients dimension of envlight, should be 3xnum_sh_coeffs. Also, add error handling for SH coeffs > 25 (i.e deg > 4)


class EnvironmentLight(torch.nn.Module):

    def __init__(self, base: torch.Tensor, base_is_SH: bool, sh_degree : int = 4):
        self.base = base
        self.base_is_SH = base_is_SH
        self.sh_dim = (sh_degree +1)**2
        if not self.base_is_SH:
            # convert base (other option cna be that is a cubemap) to SH (degree 2)
            # implement the function
            # update attribute
            self.base_is_SH = True
        # define constant attributes for diffuse irradiance computationn
        self.NUM_CHANNELS = 3
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
        normal_h = get_homogeneous(normal)
        # build symmetric matrix M
        M = torch.zeros((self.NUM_CHANNELS,4,4))
        triu_entries = torch.zeros(self.NUM_CHANNELS, 10)
        for c in self.NUM_CHANNELS:
            envc_sh = self.base[c]
            triu_entries[c] = torch.Tensor([self.C1 *envc_sh[8], self.C1 *envc_sh[4], self.C1 *envc_sh[7], self.C2*envc_sh[3],
                                            -self.C1*envc_sh[8], self.C1*envc_sh[5], self.C2*envc_sh[1],
                                            self.C3*envc_sh[6], self.C2*envc_sh[2], self.C4*envc_sh[0]-self.C5*envc_sh[6]])
            M[c][self.M_i, self.M_j] = triu_entries[c]
            M[c].T[self.M_i, self.M_j] = triu_entries[c]

        # triu_entries[:, :3] = self.C1 * triu_entries[:,:3]
        # triu_entries[:,[3,6,8]] = self.C2*triu_entries[:,[3,6,8]]

        # get diffuse irradiance
        Mn = torch.matmul(M, normal_h)
        diffuse_irradiance = torch.matmul(normal_h, Mn)
        return diffuse_irradiance



    def get_specular_irradiance(self, roughness: torch.Tensor):
        """The function computes specular irradiance by convolving
        the SH coefficients (degree 4) of the environment light with a Gaussian blur kernel of
        standard deviation = roughness. Roughness is assumed to be of dim (N,1).
        """
        # build coefficients of blur kernel in frequency (SH) domain
        gwk_sh = gauss_weierstrass_kernel(roughness, self.sh_dim) # N x 25
        gwk_sh = gwk_sh.unsqueeze(1) # N x 1 x 25
        # adjust dimensions
        envlight_sh = self.base.unsqueeze(0)   # 1 x 25 x 3
        envlight_sh = envlight_sh.repreat(gwk_sh.shape[0], 1, 1) # N x 25 x 3  
        # perform convolution
        spec_irradiance = torch.bmm(gwk_sh, envlight_sh.transpose(1, 2))  # N x 1 x 3
        spec_irradiance = spec_irradiance.squeeze(1) # N x 3
        #TODO: check if N x 1  x 3 or N x 3 x 1 pr N x 3 x 1

        return spec_irradiance



    def shade(self, gb_pos, gb_normal, albedo, ks, kr, km, view_pos, specular=True):
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

        if specular:
            metalness = km
            diff_col = albedo*(1-metalness)
            roughness = kr # (H,W,N,1)
            specularity  = ks
        else:
            diff_col = albedo # (H,W,N,3)

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        diffuse_irradiance = self.get_diffuse_irradiance() # convolve self.base (SH coeffs for envlight) with SH coefficients of cosine term, then element wise multiplication with diff_col (=albedo, in [0,1]**3 for each point)
        # alternative: use matrix representation from an efficient representaiton for irradiance environment maps.  In any case build a function
        shaded_col = diff_col*diffuse_irradiance  # diffuse radiance #*(1-specularity), /pi ?
        extras = {"diffuse": diffuse_irradiance * diff_col}

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

            # TODO edit the below commented part by convolving lighting SH coeffs given Gaussian blur. Update documentation accordingly
            roughness = roughness.squeeze(0).squeeze(0) # (N,1)
            spec_irradiance = self.get_specular_irradiance(roughness) # (N, 3) convovlve self.base with SH coeffs of Gaussian blur kernel of std roughness 
            # adjust dimensions
            spec_irradiance = spec_irradiance[None, None, ...]
            # Compute aggregate lighting
            reflectance = specularity * fg_lookup[...,0:1] + fg_lookup[...,1:2] 
            shaded_col += spec_irradiance*reflectance
            extras['specular'] = spec_irradiance*reflectance

        rgb = shaded_col

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

