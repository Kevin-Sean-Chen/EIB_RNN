# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 00:54:39 2024

@author: kevin
"""

import torch
from scipy.signal import convolve2d
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from scipy.signal import correlate
import random

import scipy as sp
from scipy.ndimage import shift
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %%
# Define the Lorenz system
def lorenz(x, y, z, s=10, r=28, b=8/3):
    dx = s * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return dx, dy, dz

# Parameters
dt = 0.02  # Time step
num_steps = 550  # Number of steps

# Initial conditions
x = np.zeros(num_steps)
y = np.zeros(num_steps)
z = np.zeros(num_steps)
x[0], y[0], z[0] = 0.0, 1.0, 1.05  # Initial position

# Euler method
for i in range(1, num_steps):
    dx, dy, dz = lorenz(x[i-1], y[i-1], z[i-1])
    x[i] = x[i-1] + dx * dt
    y[i] = y[i-1] + dy * dt
    z[i] = z[i-1] + dz * dt

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlabel("X Axis", fontsize=20)
ax.set_ylabel("Y Axis", fontsize=20)
ax.set_zlabel("Z Axis", fontsize=20)
ax.set_title("Lorenz Attractor", fontsize=20)

plt.show()

# %% processing for image
offset = 50
x = x[offset:]
y = y[offset:]
z = z[offset:]

x = (x - np.mean(x))/np.max(x)
y = (y - np.mean(y))/np.max(y)
z = (z - np.mean(z))/np.max(z)

# %% setup
N = 50
lt = num_steps*1 - offset

# %% functions
def phi(x):
    """
    rectified quardratic nonlinearity
    """
    # nl = np.where(x > 0, x**2, 0)
    # nl = np.where(x > 0, x*1, 0)  ### why a scaling factor needed!?????????????????????
    nl = np.where(x > 0, np.tanh(x)*1, 0)
    return nl

def g_kernel(sigma, size=N):
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

def spectrum_analysis(time_series, dt=dt):
    fft_result = np.fft.fft(time_series)
    
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    
    # Get the corresponding frequencies
    frequencies = np.fft.fftfreq(len(time_series), d=dt)
    return power_spectrum[:len(frequencies)//2], frequencies[:len(frequencies)//2]
    
def group_spectrum(data, dt=dt):
    N,_,T = data.shape
    pp,ff = spectrum_analysis(data[0,0,:])
    spec_all = np.zeros((N,N, len(ff)))  
    for ii in range(N):
        for jj in range(N):
            temp = data[ii,jj,:].squeeze()
            spec_all[ii,jj,:],_ = spectrum_analysis(temp)
    return np.mean(np.mean(spec_all,0),0), ff

# %% cheating with 2D basis
### spatial pattern
sigma_xy = 0.1
tau_stim = .5
mu = 0
sig_noise = 2
temp_space1 = np.random.randn(N,N)
temp_space2 = np.random.randn(N,N)
temp_k = g_kernel(sigma_xy, N)
pattern1 = spatial_convolution(temp_space1, temp_k) ########### TEST THE SPTAIAL FEATURE!!! ##############
pattern2 = spatial_convolution(temp_space2, temp_k)

lorenz_xy = np.zeros((N,N, lt))

for tt in range(lt):
    lorenz_xy[:,:,tt] = x[tt]*pattern1 + z[tt]*pattern2
    
# %% init 2D chaotic network
dt = 0.001
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
tau_i, sig_i = 15*0.001, 0.2   ### important parameters!!
#### 5, 0.14  ### grid parameter
#### 15, 0.2  ### chaos parameter
#### 10, 0.11 ### waves/strips!!!
#### 8,  0.2  ### blinking
##################################

rescale = 2. ##(N*sig_e*np.pi*1)**0.5 #1
Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale
mu_i = .8*rescale

### setting up space and time
kernel_size = 23 #37  # pick this for numerical convolution

### stim configuration
I_xy = lorenz_xy/np.max(lorenz_xy)  # 2D input video
Iamp = 2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale *1

# %%
def make_chaotic(I_xy):
    ### random initial conditions
    re_init = np.random.rand(N,N)*.1
    ri_init = np.random.rand(N,N)*.1
    re_xy = np.zeros((N,N, lt))
    ri_xy = re_xy*1
    re_xy[:,:,0] = re_init
    ri_xy[:,:,0] = ri_init
    he_xy = re_xy*1
    hi_xy = ri_xy*1
    for tt in range(lt-1):
        ### varying parameters
        rescale = 2. ##(N*sig_e*np.pi*1)**0.5 #1
        Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
        Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
        Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
        Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
        mu_e = 1.*rescale
        mu_i = .8*rescale
        
        ### modifying for 2D rate EI-RNN
        ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
        gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
        he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e \
                                                                       + I_xy[:,:,tt]*Iamp) )
        hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
        re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
    return re_xy

# %% run all sims
### chaotic
tau_i, sig_i = 15*0.001, 0.2
spontaneous_r_chaos = make_chaotic(I_xy*0)
driven_r_chaos = make_chaotic(I_xy)

tau_i, sig_i = 5*0.001, 0.14
spontaneous_r_ctr = make_chaotic(I_xy*0)
driven_r_ctr = make_chaotic(I_xy)

# %%
###############################################################################
### spectral analysis
###############################################################################
# %%
plt.figure()
plt.subplot(121)
test,ff = group_spectrum(spontaneous_r_chaos[:,:,50:])
plt.loglog(ff[2:],test[2:], label='spontaneous')
test,ff = group_spectrum(driven_r_chaos[:,:,50:])
plt.loglog(ff[2:],test[2:], label='driven')
test,ff = group_spectrum(I_xy[:,:,50:])
plt.loglog(ff[2:],test[2:], label='stim')
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power', fontsize=20)
plt.title('chaotic regime', fontsize=20)
plt.legend(fontsize=20)
plt.ylim(1e-2, 1e4)
plt.subplot(122)
test,ff = group_spectrum(spontaneous_r_ctr[:,:,50:])
plt.loglog(ff[2:],test[2:], label='spontaneous')
test,ff = group_spectrum(driven_r_ctr[:,:,50:])
plt.loglog(ff[2:],test[2:], label='driven')
test,ff = group_spectrum(I_xy[:,:,50:])
plt.loglog(ff[2:],test[2:], label='stim')
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power', fontsize=20)
plt.title('lattice regime', fontsize=20)
plt.legend(fontsize=20)
plt.ylim(1e-2, 1e4)
