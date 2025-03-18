#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
import cv2
import imageio.v3 as im
import math
from utils.sh_additional_utils import sh_render


# Coefficients to evaluate Y_lm(phi,theta) using cartesian coordinates, with l>=0 and -l<=m<=l (l in N, m in Z)

C0 = 0.28209479177387814   # Y_00
C1 = 0.4886025119029199    # Y_11, Y_10 Y_1-1
C2 = [
    1.0925484305920792, #  Y_2-2, 
    -1.0925484305920792, # Y_2-1
    0.31539156525252005, # Y_20
    -1.0925484305920792, # Y_21
    0.5462742152960396   # Y_22
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

C5 = [
    -0.6563820568401703, 
    8.302649259524165,
    -0.48923829943525043, 
    4.793536784973324,
    -0.452946651195697,
    0.1169503224534236,
    -0.452946651195697,
    2.3967683924866,
    -0.48923829943525043,
    2.075662314881041, 
    -0.6563820568401701
]



def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    The hardcoded polynomials for deg 5 refer to https://www.ppsloan.org/publications/StupidSH36.pdf.
    Args:
        deg: int SH deg. Currently, 0-5 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 5 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +  # -1, 1
                C1 * z * sh[..., 2] -   # 0,1
                C1 * x * sh[..., 3]) # 1, 1

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])

                if deg > 4:
                        result = (result +
                                C5[0] * (5 * xx * xx - 10 * yy * xx + yy * yy) * sh[..., 25] +
                                C5[1] * xy * z * (xx - yy) * sh[..., 26] +
                                C5[2] * y * (9 * zz - 1) * (3 * xx - yy) * sh[..., 27] +
                                C5[3] * xy * z * (3 * zz - 1) * sh[..., 28] +
                                C5[4] * y * (zz * (-14 + 21 * zz) + 1) * sh[..., 29] +
                                C5[5] * z * (zz * (63 * zz - 70) + 15) * sh[..., 30] +
                                C5[6] * x * (zz * (21 * zz - 14) + 15) * sh[..., 31] +
                                C5[7] * z * (xx - yy) * (-1 + 3 * zz) * sh[..., 32] +
                                C5[8] * x * (xx - 3 * yy) * (-1 + 9 * zz) * sh[..., 33] +
                                C5[9] * z * (xx * (xx - 6 * yy) + yy * yy) * sh[..., 34] +
                                C5[10] * x * (xx * (xx - 10 * yy) + 5 * yy * yy) * sh[..., 35])
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def gauss_kernel(roughness, sh_degree):
    """The function computes the sh_dim coefficients of
    Gauss Weierstrass kernel for smoothing in SH domain. The smoothing 
    strenght is proportional to the roughness.
    Args:
        roughness (torch.tensor): tensor of shape [..., 1] with roughness values
        sh_degree (int): degree of the spherical harmonics coefficients.
    Returns:
        SH coefficients of Gaussian smoothing kernel of windowing strength proportional to the roughness
    """
    l_idxs = torch.arange(sh_degree + 1, dtype=torch.float32).view(1, -1).cuda()
    gw_kernel_sh = torch.zeros((roughness.shape[0],(sh_degree + 1)**2)).cuda()
    gw_kernel_sh_l = torch.exp(-l_idxs*(l_idxs+1)*(0.3 * roughness)).cuda()
    for l in range(l_idxs.shape[1]):
        if l == 0:
            gw_kernel_sh[..., l] = gw_kernel_sh_l[..., l]
            continue 
        gw_kernel_sh[..., l**2:(l+1)**2] = gw_kernel_sh_l[..., l].unsqueeze(-1).expand_as(gw_kernel_sh[..., l**2:(l+1)**2])

    return gw_kernel_sh


def render_sh_map(sh, width: int = 600)->torch.tensor:
    """Render sh map given sh coefficients
        Args:
        sh (torch.tensor, numpy.ndarray): sh coefficients of shape [..., (sh_deg + 1)**2, 3]
        Returns:
        rendered sh map """
    if isinstance(sh, torch.Tensor):     
        rendered_sh = torch.tensor(sh_render(sh.cpu().numpy(), width = width))
    else:
        rendered_sh = torch.tensor(sh_render(sh, width = width))
    rendered_sh = torch.clamp(rendered_sh, 0, 1)

    return rendered_sh


