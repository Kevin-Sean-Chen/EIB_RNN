# -*- coding: utf-8 -*-
"""
Created on Sun May  4 12:19:58 2025

@author: kevin
"""

import torch
from scipy.signal import convolve2d
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from scipy.signal import correlate

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

np.random.seed(1) #1, 37

# %% testing response profile
### repeat stim and compute average response
### compare balanced vs. unbalance resposne profile
###############################################################################

# %% test to simplify
N = 41 #50  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
tau_i, sig_i = 15*0.001, 0.2   ### important parameters!!
#### 5, 0.14  ### grid parameter
#### 15, 0.2  ### chaos parameter
#### 10, 0.11 ### waves/strips!!!
#### 8,  0.2  ### blinking
##################################

rescale = 1. ##(N*sig_e*np.pi*1)**0.5 #1
Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale * N/10 #1.5
mu_i = .8*rescale * N/10 #1.5

### MF
rescale = 5. #7 #N/2  #8 20 30... linear with N
Wee = 1. *rescale  # recurrent weights
Wei = -2. *rescale
Wie = 1. *rescale
Wii = -2. *rescale
mu_e = .1 *1
mu_i = .1 *1

# %% network setup
### setting up space and time
dt = 0.0001  # 1ms time steps
T = .1 #10.  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
space_vec = np.linspace(0,1,N)
kernel_size = 27 #N-1 #37  # pick this for numerical convolution

### stim
amp = 0.1
step = 10
stim = (np.sin(time*100)+1)/2 *0.
stim[lt//2:lt//2+step] = np.arange(0,step)/step *amp
stim[lt//2+step:] = amp
plt.plot(stim)
stim_pattern = np.random.randn(N,N)*0 + 1 #np.zeros((N,N))
# stim_pattern[10:20, 10:20] = 1

### random initial conditions
re_xy[:,:,0] = np.random.rand(N,N)*.1
ri_xy[:,:,0] = np.random.rand(N,N)*.1
he_xy = re_xy*1
hi_xy = ri_xy*1

### measure one cell
measure_e = np.zeros(lt)
measure_i = np.zeros(lt)

### measure the field
measure_mu = np.zeros((N,N,lt))
measure_mu_ex = np.zeros((N,N,lt))

# %% dynamics
def phi(x):
    """
    rectified quardratic nonlinearity
    """
    nl = np.where(x > 0, np.tanh(x)*1, 0)
    return nl

def g_kernel(sigma, size=kernel_size):
    """
    Generates a 2D Gaussian kernel.
    """
    sigma = sigma*size
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2) + (y-x)*0.), 
        (size, size)
    )
    return kernel / np.sum(kernel)

def spatial_convolution(r,k):
    """
    2D spatial convolution given kernel k and neural field r
    """
    gr = sp.signal.convolve2d(r.squeeze(), k, mode='same',  boundary='wrap') #, fillvalue=0,
    return gr

### neural dynamics
def sim_2D_EI(N, lt=lt, k_size=kernel_size, stim=stim, pattern=stim_pattern):
    
    ### scaling params
    rescale = 1. ##(N*sig_e*np.pi*1)**0.5 #1
    Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
    Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
    Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
    Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
    mu_e = 1.*rescale *N/10
    mu_i = .8*rescale *N/10#*1.5 ############# tune this??? yes!!
    
    ### MF w/o scaling ###
    # rescale = 15. #7 #N/2  #8 20 30... linear with N
    # Wee = 1. *rescale  # recurrent weights
    # Wei = -2. *rescale
    # Wie = 1. *rescale
    # Wii = -2. *rescale
    # mu_e = .65 *1 #.1
    # mu_i = .1 *1
    
    ### prep
    re_xy = np.zeros((N,N, lt))
    ri_xy = re_xy*1

    ### random initial conditions
    re_xy[:,:,0] = np.random.rand(N,N)*.1
    ri_xy[:,:,0] = np.random.rand(N,N)*.1
    he_xy = re_xy*1
    hi_xy = ri_xy*1

    ### measure the field
    measure_mu = np.zeros((N,N,lt))
    measure_mu_ex = np.zeros((N,N,lt))
    
    ### dynamics
    for tt in range(lt-1):
        
        ### modifying for 2D rate EI-RNN
        ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e, k_size))
        gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i, k_size))
        he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) ) \
                            + stim[tt]*pattern
                        
        hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
            
        re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])       
        
        ### mean measurements
        measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
        measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
    
    beta_t = measure_mu / measure_mu_ex 
    return re_xy, beta_t
    
# %% test
re_xy, beta_t = sim_2D_EI(N, lt, k_size=37)

# %% stim response
flat_re = re_xy[:30,:30,:].reshape(-1, lt)
plt.figure()
plt.plot(np.mean(flat_re[:, 450:600],0))
plt.plot(stim[450:600]/amp)

# %% recordings
reps = 20
response = np.zeros((reps, lt))

for rr in range(reps):
    print(rr)
    re_xyi, beta_it = sim_2D_EI(N, lt, k_size=37)
    flat_re_i = re_xyi[10:20,10:20,:].reshape(-1, lt)
    response[rr,:] = np.mean(flat_re_i,0)
    
# %% plot
plt.figure()
plt.plot(np.mean(response[:,450:600],0), label='balanced')
plt.plot(unbala, label='unbalanced')
plt.plot(stim[450:600]/amp, label='stim')
plt.xlabel('time steps', fontsize=20); plt.ylabel('average rate', fontsize=20)
plt.legend(fontsize=20)