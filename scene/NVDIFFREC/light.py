import os
import numpy as np
import torch
import nvdiffrast.torch as dr
from . import util
from . import renderutils as ru
from utils.general_utils import get_homogeneous
from utils.sh_utils import gauss_weierstrass_kernel, eval_sh, sh_render
#NOTE: I also need to load envlights that don't need training, so I should have base attribute not directly initialized to LightNet object
#TODO: add dimensionality control to load_env function
#TODO: decide wehteher to use equrectangular or cubemap representation/texture for envmaps.

class EnvironmentLight(torch.nn.Module):

    def __init__(self, base: torch.Tensor, base_is_SH: bool =True, sh_degree : int = 4):
        self.base = base.squeeze()
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
        """
        The function computes diffuse irradiance by convolving
        environment light and cosine term in frequency domain. In the
        SH expansion only terms up to degree 2 are considered.
        The implementation refers to section 3.2 of "An efficient representaiton for Irradiacne Environment Maps"
        by Ramamoorthi and Pat Hanrahan.

        Args:
            normal(torch.Tensor): tensor of shape N x 3 containing N normal vectors in R³.
        Returns:
            diffuse_irradiance (torch.Tensor): tensor of shape N x 1 containing the diffuse irradiance for each normal vector.
        """
        # move normal to homogeneous coordinates:
        normal_h = get_homogeneous(normal) # N x 4
        # build symmetric matrix M
        M = torch.zeros((self.NUM_CHANNELS,4,4)).cuda()
        triu_entries = torch.zeros(self.NUM_CHANNELS, 10).cuda()
        for c in range(0, self.NUM_CHANNELS):
            envc_sh = self.base[:, c]
            triu_entries[c] = torch.Tensor([self.C1 *envc_sh[8], self.C1 *envc_sh[4], self.C1 *envc_sh[7], self.C2*envc_sh[3],
                                            -self.C1*envc_sh[8], self.C1*envc_sh[5], self.C2*envc_sh[1],
                                            self.C3*envc_sh[6], self.C2*envc_sh[2], self.C4*envc_sh[0]-self.C5*envc_sh[6]])
            M[c][self.M_i, self.M_j] = triu_entries[c]
            M[c].T[self.M_i, self.M_j] = triu_entries[c]
        # get diffuse irradiance
        M = M.unsqueeze(0) # 1 x 3 x 4 x 4
        M = M.repeat(normal_h.shape[0], 1, 1, 1) # N x 3 x 4 x 4
        Mn = torch.matmul(M, normal_h.unsqueeze(1).repeat(1,3,1).unsqueeze(-1)) # N x 3 x 4 x 4 * N x 3 x 4 x 1 = N x 3 x 4 x 1
        diffuse_irradiance = (Mn.squeeze(-1)*normal_h.unsqueeze(1)).sum(dim=-1)
 
        return diffuse_irradiance



    def get_specular_irradiance(self, roughness: torch.Tensor):
        """
        The function computes specular lighting SH coefficients by convolving
        envionment light and a Gaussian blur kernel of std = roughness in frequency domain.
        The SH coefficients are of degree 4 (= 25  coefficients).
        For what concerns the Gaussian blur filter its representation in frequency domain,
        the Gauss-Weierstrass kernel, is used to derive the corresponding SH coefficients.

        Args: 
            roughness (torch.Tensor): tensor of shape N x 1 containing N roughness values.
        Returns:
            spec_light (torch.Tensor): tensor of shape N x 25 x 3 storing the SH coefficients
                                       of specular light for each roughness value and channel.
        """
        # build coefficients of blur kernel in frequency (SH) domain
        gwk_sh = gauss_weierstrass_kernel(roughness, self.sh_degree) # N x 25
        gwk_sh = gwk_sh.unsqueeze(-1) # N x 25 x 1
        # adjust dimensions
        envlight_sh = self.base.unsqueeze(0)   # 1 x 25 x 3
        envlight_sh = envlight_sh.repeat(gwk_sh.shape[0], 1, 1) # N x 25 x 3
        # perform convolution
        spec_light = gwk_sh * envlight_sh # N x 25 x 3

        return spec_light



    def shade(self, gb_pos, gb_normal, albedo, ks, kr, km, view_pos, specular=True, tone=True):
        """
       The function returns emitted radiance in outgoing direction view_pos. If specular is 
       True a microfacet reflectance model is assumed, otherwise the model is Lambertian. 
       In the specular case the final radiance is the sum of diffuse and specular radiances.
        Args:
            gb_pos: world position
            gb_normal: normal vector
            albedo : albedo of the surface, base color
            ks: specularity
            kr: roughness
            km: metalness
            view_pos: viewing direction
            envlight: SH coefficients of environment light
        Retursn:
            rgb (torch.Tensor): shaded rgb color of shape 1 x 1 x N x 3.
            extras (dict): dictionary storing diffuse and specular radiance.
        """
        assert self.base_is_SH, "envlight can be only represented through Spherical Harmonics"

        # (H, W, N, C)
        wo = util.safe_normalize(view_pos - gb_pos)

        diff_col = albedo
        if specular:
            metalness = km
            roughness = kr # (H,W,N,1)
            specularity  = ks
            diff_col = (1-metalness)*diff_col

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal

        diffuse_irradiance = self.get_diffuse_irradiance(nrmvec.squeeze())
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
            spec_light = self.get_specular_irradiance(roughness) # (N, 25, 3)
            # transpose for eval_sh
            spec_light = spec_light.transpose(1,2)
            # compute specular irradiance
            spec_irradiance = eval_sh(self.sh_degree, spec_light, reflvec.squeeze())
            # adjust dimensions
            spec_irradiance = spec_irradiance[None, None, ...] # (H, W, N, 3)
            # Compute aggregate lighting
            if metalness is None:
                F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
            else:
                F0 = (1.0 - metalness) * 0.04 + albedo * metalness
            reflectance = F0* fg_lookup[...,0:1] + fg_lookup[...,1:2]
            specular_radiance = spec_irradiance*reflectance
            shaded_col += specular_radiance
            extras['specular'] = spec_irradiance*reflectance
        else:
            extras['specular'] = None

        if tone:
            # apply tone mapping and clamp in range [0,1]: linear --> sRGB
            rgb = util.gamma_correction(shaded_col)
        else:
            rgb = shaded_col.clamp(min=0.0, max=1.0)

        return rgb, extras
    

    def set_base(self, base: torch.Tensor, base_is_SH: bool = True):
        self.base = base.squeeze()
        self.base_is_SH = base_is_SH


    def render_sh(self, width: int = 600):
        """Render light SH coefficients in equirectangular format"""
        assert self.base_is_SH == True, "sh environment light base is required"
        self.base = self.base.squeeze()
        rendered_sh = sh_render(self.base, width = width)
        rendered_sh = (rendered_sh - rendered_sh.min()) / (rendered_sh.max() - rendered_sh.min()) * 255
        rendered_sh = rendered_sh.astype(np.uint8)
        return rendered_sh



def load_hdr_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() != ".hdr":
        raise OSError("Unknown envlight extension")
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])
    envlight_sh = get_SH_from_cubemap(cubemap)
    l = EnvironmentLight(envlight_sh, base_is_SH=True)
    return l

