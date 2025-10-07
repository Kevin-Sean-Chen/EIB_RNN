# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 12:00:31 2024

@author: kevin
"""

import torch
from scipy.signal import convolve2d
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)


# %%
N = 50  # neurons
rescale = 10 #1
Wee = 1.*rescale  # recurrent weights
Wei = -2.*rescale
Wie = 1.*rescale
Wii = -2.*rescale
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
mu_e = 1.  #1e-8#.001*1  # offset
mu_i = .8  #1e-8#.001*1
tau_i, sig_i = 10*0.001, 0.1

kernel_size = 20

# %% build probablistic network connetion (with dilution Cij)
def g_kernel(sigma, size=kernel_size):
    """
    Generates a 2D Gaussian kernel.
    """
    sigma = sigma*size
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)), 
        (size, size)
    )
    return kernel / np.sum(kernel)

def create_grid(K):
    x = np.linspace(0, 1, K)
    y = np.linspace(0, 1, K)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack((xx, yy), axis=-1)
    return grid

def local_network(N, sig):
    k = g_kernel(sig)
    nx = N**2
    cij = np.zeros((nx, nx))
    location_2d = create_grid(N)
    for xx in range(nx):
        delta2d = np.zeros((N,N))
        delta2d[xx] = 1
        connect_2d_prob = sp.signal.convolve2d(delta2d, k, mode='same',  boundary='wrap')
        ci = np.random.rand(N, N) < connect_2d_prob
        cij[xx,:] = ci.reshape(-1)
    return cij

# %%
