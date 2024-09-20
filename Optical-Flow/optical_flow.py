import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from ex1_utils import *


def lucas_kanade(im1 ,im2 , N, sigma=1, epsilon=1e-7, harris=False, harris_threshold=0.01, harris_alpha=0.05,
                 return_derivatives=False):
    # im1 − first image matrix (grayscale)
    # im2 − second image matrix (grayscale)
    # N − size of the neighborhood (N x N)

    I1 = im1.copy().astype(np.float64)
    I2 = im2.copy().astype(np.float64)
    sum_kernel = np.ones((N, N))

    I1 /= 255.0
    I2 /= 255.0


    Ix1, Iy1 = gaussderiv(I1, sigma)
    Ix2, Iy2 = gaussderiv(I2, sigma)
    Ix = (Ix1 + Ix2) / 2.0
    Iy = (Iy1 + Iy2) / 2.0
    It = I2 - I1
    It = gausssmooth(It, sigma)

    IxIy = Ix * Iy
    
    
    I_y2 = cv2.filter2D(Iy**2, -1, sum_kernel)
    I_x2 = cv2.filter2D(Ix**2, -1, sum_kernel)
    Ixy = cv2.filter2D(IxIy, -1, sum_kernel)

    D = I_x2 * I_y2 - Ixy ** 2
    
    


    IxIt = cv2.filter2D(Ix * It, -1, sum_kernel)
    IyIt = cv2.filter2D(Iy * It, -1, sum_kernel)

    u = - (I_y2 * IxIt - Ixy * IyIt)
    v = - (I_x2 * IyIt - Ixy * IxIt)

    D[D == 0] = epsilon

    if harris:
        harris_response = D - harris_alpha * (Iy2 + Ix2) ** 2

        unstable_values = harris_response < harris_threshold
        u[unstable_values] = 0
        v[unstable_values] = 0
        u[~unstable_values] = u[~unstable_values] / D[~unstable_values]
        v[~unstable_values] = v[~unstable_values] / D[~unstable_values]
    else:

        u = u / D
        v = v / D

    if return_derivatives:
        return u, v, Ix, Iy, It
    else:        
        return u, v
    

def calculate_pyramid(im, levels):
    pyramid = [im]
    for _ in range(levels):
        im = cv2.pyrDown(im)
        pyramid.append(im)
    return pyramid


def lucas_kanade_pyramidal(im1, im2, N, levels, sigma=1, harris=False, harris_threshold=0.01, harris_alpha=0.05):

    g_pyramid_1 = calculate_pyramid(im1, levels)
    g_pyramid_2 = calculate_pyramid(im2, levels)
    
    g_pyramid_1 = g_pyramid_1[::-1]
    g_pyramid_2 = g_pyramid_2[::-1]

    u, v = lucas_kanade(g_pyramid_1[0], g_pyramid_2[0], N,
                        sigma=sigma,
                        harris_alpha=harris_alpha, 
                        harris_threshold=harris_threshold, 
                        harris=harris,
                        return_derivatives=False)
    print(u.shape, v.shape)
    
    for i in range(1, len(g_pyramid_1)):
        u = cv2.pyrUp(u)
        v = cv2.pyrUp(v)

        u = cv2.resize(u, (g_pyramid_1[i].shape[1], g_pyramid_1[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (g_pyramid_1[i].shape[1], g_pyramid_1[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        u_wrap, v_wrap = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

        u_wrap = (u_wrap + u).astype(np.float32)
        v_wrap = (v_wrap + v).astype(np.float32)

        I2_wrap = cv2.remap(g_pyramid_2[i], u_wrap, v_wrap, cv2.INTER_LINEAR)
    
        u_current, v_current = lucas_kanade(g_pyramid_1[i], I2_wrap, N, return_derivatives=False)
        u += u_current
        v += v_current
        


    return u, v


def horns_chunck(im1, im2, n_iters, lmbd, sigma=1, init_lucas_kanade=False):
    # im1 − f i r s t image mat rix ( g r a y s c a l e )
    # im2 − sec ond image ma t rix ( g r a y s c a l e )
    # n_i t e r s − number o f i t e r a t i o n s ( t r y s e v e r a l hundred )
    # lmbd − parameter
    # TODO
    
    if init_lucas_kanade:
        u, v, Ix, Iy, It = lucas_kanade(im1, im2, 11, sigma, return_derivatives=True)
    
    else:
        u = np.zeros(im1.shape)
        v = np.zeros(im1.shape)
    
        I1 = im1.copy().astype(np.float32)
        I2 = im2.copy().astype(np.float32)
        I1 = I1 / 255.0
        I2 = I2 / 255.0


        Ix1, Iy1 = gaussderiv(I1, sigma)
        Ix2, Iy2 = gaussderiv(I2, sigma)
        Ix = (Ix1 + Ix2) / 2.0
        Iy = (Iy1 + Iy2) / 2.0
        It = I2 - I1
        It = gausssmooth(It, sigma)


    D = lmbd + Ix ** 2 + Iy ** 2    
    laplacian = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4

    for _ in range(n_iters):        
        u_a = cv2.filter2D(u, -1, laplacian)
        v_a = cv2.filter2D(v, -1, laplacian)
        P = Ix * u_a + Iy * v_a + It
        u = u_a - Ix * (P / D)
        v = v_a - Iy * (P / D)


    return u, v
