import os
import numpy as np
import torch
import nvdiffrast.torch as dr
from . import util
from . import renderutils as ru
from utils.general_utils import get_homogeneous
from utils.sh_utils import  eval_sh
from utils.sh_additional_utils import sh_render
from utils.sh_utils import gauss_weierstrass_kernel
from typing import Dict, Tuple


class EnvironmentLight(torch.nn.Module):

    def __init__(self, base: torch.Tensor, sh_degree : int = 4):
        """
        The class implements a shader based on IBL by following the implementation of NVDIFFREC, https://github.com/NVlabs/nvdiffrecmc.

        Attributes:
            base (torch.tesnor): Spherical Harmonics (SH) coefficients
            sh_degree (int): SH degree,
            sh_dim (int): number of SH coefficients
        Constants
            NUM_CHANNELS (int): number of channels of base, which is RGB,
            C1,C2,...,C5 (int): constants for computing diffuse irradiance
            M_i,M_j (torch.tensor): indices of the upper triangular part of the matrix involved in diffuse irradiance computation.
        """
        self.base = base.squeeze()
        self.sh_degree = sh_degree
        self.sh_dim = (sh_degree +1)**2 
        # Define constant attributes for diffuse irradiance computation
        self.NUM_CHANNELS = 3
        self.C1 = 0.429043
        self.C2 = 0.511664
        self.C3 = 0.743125
        self.C4 = 0.886227
        self.C5 = 0.247708
        self.M_i, self.M_j = torch.triu_indices(4,4)


    def clone(self):
        return EnvironmentLight(self.base.clone().detach())


    def get_shdim(self):
        return self.sh_dim


    def get_shdegree(self):
        return self.sh_degree


    def get_base(self):
        return self.base


    def set_base(self, base: torch.Tensor):
        assert base.squeeze().shape[0] == self.sh_dim, f"The number of SH coefficients must be {self.sh_dim}"
        self.base = base.squeeze()


    def get_diffuse_irradiance(self, normal: torch.tensor)->torch.tensor:
        """
        The function computes the diffuse irradiance according to section 3.2 of "An efficient representaiton for Irradiance Environment Maps"
        by Ramamoorthi and Pat Hanrahan, https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf.
        The diffuse irradiance is computed by convolving environment light and cosine term in frequency domain. In the
        SH expansion of the environment light only terms up to degree 2 are considered.

        Args:
            normal: tensor of shape (N,3) containing normal vectors in R³.
        Returns:
            diffuse_irradiance: tensor of shape (N,1) containing the diffuse irradiance for each normal vector.
        """
        # Move normal to homogeneous coordinates:
        normal_h = get_homogeneous(normal) # N x 4
        # Build symmetric matrix M
        M = torch.zeros((self.NUM_CHANNELS,4,4)).cuda()
        triu_entries = torch.zeros(self.NUM_CHANNELS, 10).cuda()
        for c in range(0, self.NUM_CHANNELS):
            envc_sh = self.base[:, c]
            triu_entries[c] = torch.Tensor([self.C1 *envc_sh[8], self.C1 *envc_sh[4], self.C1 *envc_sh[7], self.C2*envc_sh[3],
                                            -self.C1*envc_sh[8], self.C1*envc_sh[5], self.C2*envc_sh[1],
                                            self.C3*envc_sh[6], self.C2*envc_sh[2], self.C4*envc_sh[0]-self.C5*envc_sh[6]])
            M[c][self.M_i, self.M_j] = triu_entries[c]
            M[c].T[self.M_i, self.M_j] = triu_entries[c]
        # Get diffuse irradiance
        M = M.unsqueeze(0) # 1 x 3 x 4 x 4
        M = M.repeat(normal_h.shape[0], 1, 1, 1) # N x 3 x 4 x 4
        Mn = torch.matmul(M, normal_h.unsqueeze(1).repeat(1,3,1).unsqueeze(-1)) # N x 3 x 4 x 4 * N x 3 x 4 x 1 = N x 3 x 4 x 1
        diffuse_irradiance = (Mn.squeeze(-1)*normal_h.unsqueeze(1)).sum(dim=-1)
 
        return diffuse_irradiance


    def get_specular_light_sh(self, roughness: torch.Tensor)->torch.tensor:
        """
        The function computes specular lighting SH coefficients by convolving
        envionment light and a Gaussian blur kernel of std = roughness in frequency domain.
        For what concerns the Gaussian blur filter its representation in frequency domain,
        the Gauss-Weierstrass kernel, is used to derive the corresponding SH coefficients.

        Args: 
            roughness: tensor of shape N x 1 containing N roughness values.
        Returns:
            spec_light: tensor of shape N x self.sh_dim x 3 storing the SH coefficients
                                       of specular light for each roughness value and channel.
        """
        # Build coefficients of blur kernel in frequency (SH) domain
        gwk_sh = gauss_weierstrass_kernel(roughness, self.sh_degree) # N x 25
        gwk_sh = gwk_sh.unsqueeze(-1) # N x 25 x 1
        # Adjust dimensions
        envlight_sh = self.base.unsqueeze(0)   # 1 x 25 x 3
        envlight_sh = envlight_sh.repeat(gwk_sh.shape[0], 1, 1) # N x 25 x 3
        # Perform convolution
        spec_light = gwk_sh * envlight_sh # N x 25 x 3

        return spec_light



    def shade(self, gb_pos:torch.tensor, gb_normal:torch.tensor, albedo:torch.tensor, view_pos:torch.tensor,
              kr:torch.tensor=None, km:torch.tensor=None, specular:bool=True, tone:bool=True)->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
       The function, based on NVDIFFREC implementation https://github.com/NVlabs/nvdiffrecmc,
        returns the emitted radiance in the input outgoing direction. 
        If specular is True a Microfacets Cook-Torrane reflectance model is assumed, otherwise the model is assumed to be Lambertian. 
        In the specular case the final radiance is the sum of the diffuse and specular radiances.
        Args:
            gb_pos: world positions HxWxNx3
            gb_normal: normal vectors HxWxNx3
            albedo : albedo of the surface, base color HxWxNx3
            kr: roughness of points HxWxNx1
            km: metalness of points HxWxNx1
            view_pos: viewing directions HxWxNx3
            envlight: SH coefficients of environment light 1xself.sh_dimx3
        Returns:
            rgb: shaded rgb color of shape HxWxNx3.
            extras: dictionary storing diffuse and specular radiance.
        """        

        diff_col = albedo
        if specular:
            metalness = km
            roughness = kr # (H,W,N,1)
            diff_col = (1-metalness)*albedo

        nrmvec = gb_normal

        diffuse_irradiance = self.get_diffuse_irradiance(nrmvec.squeeze())
        diffuse_irradiance = torch.nn.functional.relu(diffuse_irradiance)
        diffuse_radiance = diff_col*diffuse_irradiance
        shaded_col = diffuse_radiance
        extras = {"diffuse": util.gamma_correction(diffuse_radiance)}

        if specular:
            wo = util.safe_normalize(view_pos - gb_pos) # (H, W, N, 3)
            reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, nrmvec), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')
            roughness = roughness.squeeze(0).squeeze(0) # (N,1)
            # convovlve self.base with SH coeffs of Gaussian blur kernel of std roughness 
            spec_light = self.get_specular_light_sh(roughness) # (N, 25, 3)
            # transpose for eval_sh
            spec_light = spec_light.transpose(1,2)
            # compute specular irradiance in reflection direction
            spec_irradiance = eval_sh(self.sh_degree, spec_light, reflvec.squeeze())
            # adjust dimensions
            spec_irradiance = spec_irradiance[None, None, ...] # (H, W, N, 3)
            spec_irradiance = torch.nn.functional.relu(spec_irradiance)
            # Compute aggregate lighting
            if metalness is None:
                F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
            else:
                F0 = (1.0 - metalness) * 0.04 + albedo * metalness
            reflectance = F0* fg_lookup[...,0:1] + fg_lookup[...,1:2]
            specular_radiance = spec_irradiance*reflectance
            shaded_col = shaded_col + specular_radiance
            extras['specular'] = util.gamma_correction(specular_radiance)
        else:
            extras['specular'] = torch.zeros_like(extras["diffuse"])

        if tone:
            # apply tone mapping and clamp in range [0,1]: linear --> sRGB
            rgb = util.gamma_correction(shaded_col)
        else:
            rgb = shaded_col.clamp(min=0.0, max=1.0)

        return rgb, extras


    def render_sh(self, width: int = 600)->np.array:
        """Render environment light SH coefficients in equirectangular format"""
        self.base = self.base.squeeze()
        rendered_sh = sh_render(self.base, width = width)
        rendered_sh = (rendered_sh - rendered_sh.min()) / (rendered_sh.max() - rendered_sh.min()) * 255
        rendered_sh = rendered_sh.astype(np.uint8)
        return rendered_sh


