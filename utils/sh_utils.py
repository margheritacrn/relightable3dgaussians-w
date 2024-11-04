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
import utils.spherical_harmonics as  sh
import utils.sh_additional_utils as utility
import imageio.v3 as im

# coefficients to evaluate Y_lm(phi,theta) using cartesian coordinates, with l>=0 and -l<=m<=l (l in N, m in Z)

C0 = 0.28209479177387814   # Y_00
C1 = 0.4886025119029199    # Y_11, Y_10 Y_1-1
C2 = [
    1.0925484305920792, # Y_21,- Y_2-1, Y_2-2
    -1.0925484305920792,
    0.31539156525252005, # Y_20
    -1.0925484305920792,
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


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

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
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def gauss_weierstrass_kernel(roughness, sh_degree):
    """The function computes the sh_dim coefficients of
    Gauss Weierstrass kernel for smoothing in SH domain.
    """
    l_idxs = torch.arange(sh_degree + 1, dtype=torch.float32).view(1, -1).cuda()
    gw_kernel_sh = torch.zeros((roughness.shape[0],(sh_degree + 1)**2)).cuda()
    gw_kernel_sh_l = torch.exp(-l_idxs*(l_idxs+1)*(roughness**2)).cuda()
    for l in range(l_idxs.shape[0]):
        if l == 0:
            gw_kernel_sh[..., l] = gw_kernel_sh_l[..., l]
            continue
        gw_kernel_sh[..., l**2:(l+1)**2] = gw_kernel_sh_l[..., l]
        

    return gw_kernel_sh

'''
MIT License

Copyright (c) 2018 Andrew Chalmers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
Utility functions to demonstrate usage of spherical harmonics
'''

def get_coefficients_matrix(xres,l_max=2):
	yres = int(xres/2)
	# setup fast vectorisation
	x = np.arange(0,xres)
	y = np.arange(0,yres).reshape(yres,1)

	# Setup polar coordinates
	lat_lon = xy_to_ll(x,y,xres,yres)

	# Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
	Ylm = sh.sh_evaluate(lat_lon[0], lat_lon[1], l_max)
	return Ylm

def get_coefficients_from_file(ibl_filename, l_max=2, resize_width=None, filder_amount=None):
	ibl = im.imread(os.path.join(os.path.dirname(__file__), ibl_filename), plugin='EXR-FI')
	return get_coefficients_from_image(ibl, l_max=l_max, resize_width=resize_width, filder_amount=filder_amount)

def get_coefficients_from_image(ibl, l_max=2, resize_width=None, filder_amount=None):
	# Resize if necessary (I recommend it for large images)
	if resize_width is not None:
		#ibl = cv2.resize(ibl, dsize=(resize_width,int(resize_width/2)), interpolation=cv2.INTER_CUBIC)
		ibl = utility.resize_image(ibl, resize_width, int(resize_width/2), cv2.INTER_CUBIC)
	elif ibl.shape[1] > 1000:
		#print("Input resolution is large, reducing for efficiency")
		#ibl = cv2.resize(ibl, dsize=(1000,500), interpolation=cv2.INTER_CUBIC)
		ibl = utility.resize_image(ibl, 1000, 500, cv2.INTER_CUBIC)
	xres = ibl.shape[1]
	yres = ibl.shape[0]

	# Pre-filtering, windowing
	if filder_amount is not None:
		ibl = blur_ibl(ibl, amount=filder_amount)

	# Compute sh coefficients
	sh_basis_matrix = get_coefficients_matrix(xres,l_max)

	# Sampling weights
	solid_angles = utility.get_solid_angle_map(xres)

	# Project IBL into SH basis
	n_coeffs = sh.sh_terms(l_max)
	ibl_coeffs = np.zeros((n_coeffs,3))
	for i in range(0,sh.sh_terms(l_max)):
		ibl_coeffs[i,0] = np.sum(ibl[:,:,0]*sh_basis_matrix[:,:,i]*solid_angles)
		ibl_coeffs[i,1] = np.sum(ibl[:,:,1]*sh_basis_matrix[:,:,i]*solid_angles)
		ibl_coeffs[i,2] = np.sum(ibl[:,:,2]*sh_basis_matrix[:,:,i]*solid_angles)

	return ibl_coeffs

def find_windowing_factor(coeffs, max_laplacian=10.0):
	# http://www.ppsloan.org/publications/StupidSH36.pdf 
	# Based on probulator implementation, empirically chosen max_laplacian
	l_max = sh_l_max_from_terms(coeffs.shape[0])
	tableL = np.zeros((l_max+1))
	tableB = np.zeros((l_max+1))

	def sqr(x):
		return x*x
	def cube(x):
		return x*x*x

	for l in range(1, l_max+1):
		tableL[l] = float(sqr(l) * sqr(l + 1))
		B = 0.0
		for m in range(-1, l+1):
			B += np.mean(coeffs[sh_index(l,m),:])
		tableB[l] = B;

	squared_laplacian = 0.0
	for l in range(1, l_max+1):
		squared_laplacian += tableL[l] * tableB[l]

	target_squared_laplacian = max_laplacian * max_laplacian;
	if (squared_laplacian <= target_squared_laplacian): return 0.0

	windowing_factor = 0.0
	iterationLimit = 10000000;
	for i in range(0, iterationLimit):
		f = 0.0
		fd = 0.0
		for l in range(1, l_max+1):
			f += tableL[l] * tableB[l] / sqr(1.0 + windowing_factor * tableL[l]);
			fd += (2.0 * sqr(tableL[l]) * tableB[l]) / cube(1.0 + windowing_factor * tableL[l]);

		f = target_squared_laplacian - f;

		delta = -f / fd;
		windowing_factor += delta;
		if (abs(delta) < 0.0000001):
			break
	return windowing_factor

def apply_windowing(coeffs, windowing_factor=None, verbose=False):
	# http://www.ppsloan.org/publications/StupidSH36.pdf 
	l_max = sh_l_max_from_terms(coeffs.shape[0])
	if windowing_factor is None:
		windowing_factor = find_windowing_factor(coeffs)
	if windowing_factor <= 0: 
		if verbose: print("No windowing applied")
		return coeffs
	if verbose: print("Using windowing_factor: %s" % (windowing_factor))
	for l in range(0, l_max+1):
		s = 1.0 / (1.0 + windowing_factor * l * l * (l + 1.0) * (l + 1.0))
		for m in range(-l, l+1):
			coeffs[sh_index(l,m),:] *= s;
	return coeffs

# Spherical harmonics reconstruction
def get_diffuse_coefficients(l_max):
	# From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
	diffuse_coeffs = [np.pi, (2*np.pi)/3]
	for l in range(2,l_max+1):
		if l%2==0:
			a = (-1.)**((l/2.)-1.)
			b = (l+2.)*(l-1.)
			#c = float(np.math.factorial(l)) / (2**l * np.math.factorial(l/2)**2)
			c = math.factorial(int(l)) / (2**l * math.factorial(int(l//2))**2)
			#s = ((2*l+1)/(4*np.pi))**0.5
			diffuse_coeffs.append(2*np.pi*(a/b)*c)
		else:
			diffuse_coeffs.append(0)
	return np.asarray(diffuse_coeffs) / np.pi

def sh_reconstruct_signal(coeffs, sh_basis_matrix=None, width=600):
	if sh_basis_matrix is None:
		l_max = sh_l_max_from_terms(coeffs.shape[0])
		sh_basis_matrix = get_coefficients_matrix(width,l_max)
	return np.dot(sh_basis_matrix,coeffs).astype(np.float32)

def sh_render(ibl_coeffs, width=600):
	l_max = sh_l_max_from_terms(ibl_coeffs.shape[0])
	diffuse_coeffs =get_diffuse_coefficients(l_max)
	sh_basis_matrix = get_coefficients_matrix(width,l_max)
	rendered_image = np.zeros((int(width/2),width,3))
	for idx in range(0,ibl_coeffs.shape[0]):
		l = l_from_idx(idx)
		coeff_rgb = diffuse_coeffs[l] * ibl_coeffs[idx,:]
		rendered_image[:,:,0] += sh_basis_matrix[:,:,idx] * coeff_rgb[0]
		rendered_image[:,:,1] += sh_basis_matrix[:,:,idx] * coeff_rgb[1]
		rendered_image[:,:,2] += sh_basis_matrix[:,:,idx] * coeff_rgb[2]
	return rendered_image

def get_normal_map_axes_duplicate_rgb(normal_map):
	# Make normal for each axis, but in 3D so we can multiply against RGB
	N3Dx = np.repeat(normal_map[:,:,0][:, :, np.newaxis], 3, axis=2)
	N3Dy = np.repeat(normal_map[:,:,1][:, :, np.newaxis], 3, axis=2)
	N3Dz = np.repeat(normal_map[:,:,2][:, :, np.newaxis], 3, axis=2)
	return N3Dx, N3Dy, N3Dz

def sh_render_l2(ibl_coeffs, normal_map):
	# From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
	C1 = 0.429043
	C2 = 0.511664
	C3 = 0.743125
	C4 = 0.886227
	C5 = 0.247708
	N3Dx, N3Dy, N3Dz = get_normal_map_axes_duplicate_rgb(normal_map)
	return	(C4 * ibl_coeffs[0,:] + \
			2.0 * C2 * ibl_coeffs[3,:] * N3Dx + \
			2.0 * C2 * ibl_coeffs[1,:] * N3Dy + \
			2.0 * C2 * ibl_coeffs[2,:] * N3Dz + \
			C1 * ibl_coeffs[8,:] * (N3Dx * N3Dx - N3Dy * N3Dy) + \
			C3 * ibl_coeffs[6,:] * N3Dz * N3Dz - C5 * ibl_coeffs[6] + \
			2.0 * C1 * ibl_coeffs[4,:] * N3Dx * N3Dy + \
			2.0 * C1 * ibl_coeffs[7,:] * N3Dx * N3Dz + \
			2.0 * C1 * ibl_coeffs[5,:] * N3Dy * N3Dz ) / np.pi

def get_normal_map(width):
	height = int(width/2)
	x = np.arange(0,width)
	y = np.arange(0,height).reshape(height,1)
	lat_lon = xy_to_ll(x,y,width,height)
	return spherical_to_cartesian_2(lat_lon[0], lat_lon[1])

def sh_reconstruct_diffuse_map(ibl_coeffs, width=600):
	# Rendering 
	if ibl_coeffs.shape[0] == 9: # L2
		# setup fast vectorisation
		xyz = get_normal_map(width)
		rendered_image = sh_render_l2(ibl_coeffs, xyz)
	else: # !L2
		rendered_image = sh_render(ibl_coeffs, width)

	return rendered_image.astype(np.float32)

def write_reconstruction(c, l_max, fn='', width=600, output_dir='./output/'):
	im.imwrite(output_dir+'_sh_light_l'+str(l_max)+fn+'.exr',sh_reconstruct_signal(c, width=width))
	im.imwrite(output_dir+'_sh_render_l'+str(l_max)+fn+'.exr',sh_reconstruct_diffuse_map(c, width=width))

# Utility functions for SPH
def sh_print(coeffs, precision=3):
	n_coeffs = coeffs.shape[0]
	l_max = sh_l_max_from_terms(coeffs.shape[0])
	currentBand = -1
	for idx in range(0,n_coeffs):
		band = l_from_idx(idx)
		if currentBand!=band:
			currentBand = band
			print('L'+str(currentBand)+":")
		print(np.around(coeffs[idx,:],precision))
	print('')

def sh_print_to_file(coeffs, precision=3, output_file_path="./output/_coefficients.txt"):
	n_coeffs = coeffs.shape[0]
	l_max = sh_l_max_from_terms(coeffs.shape[0])
	currentBand = -1

	with open(output_file_path, 'w') as file:		
		for idx in range(0,n_coeffs):

			# Print the band level
			band = l_from_idx(idx)
			if currentBand!=band:
				currentBand = band
				outputData_BandLevel = "L"+str(currentBand)+":"
				print(outputData_BandLevel)
				file.write(outputData_BandLevel+'\n')

			# Print the coefficients at this band level
			outputData_Coefficients = np.around(coeffs[idx,:],precision)
			print(outputData_Coefficients)
			file.write("[")
			for i in range(0,len(outputData_Coefficients)):
				file.write(str(outputData_Coefficients[i]))
				if i<len(outputData_Coefficients)-1: # don't add white space after last coefficient
					file.write(' ')
			file.write(']\n')

def sh_terms_within_band(l):
	return (l*2)+1

def sh_l_max_from_terms(terms):
	return int(np.sqrt(terms)-1)

def l_from_idx(idx):
	return int(np.sqrt(idx))

def paint_negatives(img):
	indices = [img[:,:,0] < 0 or img[:,:,1] < 0 or img[:,:,2] < 0]
	img[indices[0],0] = abs((img[indices[0],0]+img[indices[0],1]+img[indices[0],2])/3)*10
	img[indices[0],1] = 0
	img[indices[0],2] = 0

def spherical_to_cartesian_2(theta, phi):
	phi = phi + np.pi
	x = np.sin(theta)*np.cos(phi)
	y = np.cos(theta)
	z = np.sin(theta)*np.sin(phi)
	if not np.isscalar(x):
		y = np.repeat(y, x.shape[1], axis=1)
	return np.moveaxis(np.asarray([x,z,y]), 0,2)

def spherical_to_cartesian(theta, phi):
	x = np.sin(theta)*np.cos(phi)
	y = np.sin(theta)*np.sin(phi)
	z = np.cos(theta)
	if not np.isscalar(x):
		z = np.repeat(z, x.shape[1], axis=1)
	return np.moveaxis(np.asarray([x,z,y]), 0,2)

def xy_to_ll(x,y,width,height):
	def yLocToLat(yLoc, height):
		return (yLoc / (float(height)/np.pi))
	def xLocToLon(xLoc, width):
		return (xLoc / (float(width)/(np.pi * 2)))
	return np.asarray([yLocToLat(y, height), xLocToLon(x, width)], dtype=object)

def get_cartesian_map(width):
	height = int(width/2)
	image = np.zeros((height,width))
	x = np.arange(0,width)
	y = np.arange(0,height).reshape(height,1)
	lat_lon = xy_to_ll(x,y,width,height)
	return spherical_to_cartesian(lat_lon[0], lat_lon[1])

def sh_visualise(l_max=2, sh_basis_matrix=None, show=False, output_dir='./output/'):
	cdict =	{'red':	((0.0, 1.0, 1.0),
					(0.5, 0.0, 0.0),
					(1.0, 0.0, 0.0)),
			'green':((0.0, 0.0, 0.0),
					(0.5, 0.0, 0.0),
					(1.0, 1.0, 1.0)),
			'blue':	((0.0, 0.0, 0.0),
					(1.0, 0.0, 0.0))}

	if sh_basis_matrix is None:
		sh_basis_matrix = get_coefficients_matrix(600,l_max)

	l_max = sh_l_max_from_terms(sh_basis_matrix.shape[2])

	rows = l_max+1
	cols = sh_terms_within_band(l_max)
	img_index = 0

	if l_max==0:
		plt.imshow(sh_basis_matrix[:,:,0], cmap=LinearSegmentedColormap('RedGreen', cdict), vmin=-1, vmax=1)
		plt.axis("off")
		plt.savefig(output_dir+'_fig_sh_l'+str(l_max)+'.jpg')
		if show:
			plt.show()
		return

	fig, axs = plt.subplots(nrows=rows, ncols=cols, gridspec_kw={'wspace':0.1, 'hspace':-0.4}, squeeze=True, figsize=(16, 8))
	for c in range(0,cols):
		for r in range(0,rows):
			axs[r,c].axis('off')

	for l in range(0,l_max+1):
		n_in_band = sh_terms_within_band(l)
		col_offset = int(cols/2) - int(n_in_band/2)
		row_offset = (l*cols)+1
		index = row_offset+col_offset
		for i in range(0,n_in_band):
			axs[l, i+col_offset].axis("off")
			axs[l, i+col_offset].imshow(sh_basis_matrix[:,:,img_index], cmap=LinearSegmentedColormap('RedGreen', cdict), vmin=-1, vmax=1)
			img_index+=1

	#plt.tight_layout()
	plt.savefig(output_dir+'_fig_sh_l'+str(l_max)+'.jpg')
	if show:
		plt.show()


