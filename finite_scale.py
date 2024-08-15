# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:24:41 2024

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

# %% finite scale analysis
### given the NxN scaling check, we want to confirm that in the chaotic regime,
### the variance for neural firing rate should decrease as simulation lengthens,
### this should proof that the system is truely ergotic in long time scale
###############################################################################

# %% test to simplify
N = 50  # neurons
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

# %% network setup
### setting up space and time
dt = 0.001  # 1ms time steps
T = 10.  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
space_vec = np.linspace(0,1,N)
kernel_size = 23 #37  # pick this for numerical convolution

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

def phi(x):
    """
    rectified quardratic nonlinearity
    """
    # nl = np.where(x > 0, x**2, 0)
    # nl = np.where(x > 0, x*1, 0)  ### why a scaling factor needed!?????????????????????
    # nl = 1/(1+np.exp(-x))
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

# %% dynamics
def spatial_convolution(r,k):
    """
    2D spatial convolution given kernel k and neural field r
    """
    gr = sp.signal.convolve2d(r.squeeze(), k, mode='same',  boundary='wrap') #, fillvalue=0,
    return gr

### neural dynamics
for tt in range(lt-1):
    
    ### modifying for 2D rate EI-RNN
    ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
    re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
    ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
    
    ### make E-I measurements
    measure_e[tt+1] = (Wee*ge_conv_re + mu_e)[20,20]
    measure_i[tt+1] = (Wei*gi_conv_ri)[20,20]
    
    
    ### mean measurements
    measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
    
# %%
# offset = 50
# plt.figure()
# plt.plot(time[offset:], measure_e[offset:], label='E')
# plt.plot(time[offset:], measure_i[offset:], label='I')
# plt.plot(time[offset:], (measure_e+measure_i)[offset:],label='total')
# plt.xlabel('time (s)', fontsize=20)
# plt.ylabel('current', fontsize=20)
# plt.title('spatial balancing', fontsize=20)
# plt.legend(fontsize=15)

# # %% plot dynamics
# offset = 1
# plt.figure()
# plt.plot(time[offset:], re_xy[20,20,offset:].squeeze())
# plt.plot(time[offset:], re_xy[15,15,offset:].squeeze())
# plt.plot(time[offset:], ri_xy[15,15,offset:].squeeze(),'-o')
# plt.xlabel('time (s)', fontsize=20)
# plt.ylabel('rate (Hz)', fontsize=20)
# # plt.xlim([0.1,0.14])

# # %% pot beta for a single cell
# beta_i = np.abs(measure_e + measure_i)/measure_e

# plt.figure()
# plt.plot(time, beta_i)
# plt.xlabel('time', fontsize=20)
# plt.ylabel('beta', fontsize=20)

# %% visualize
### for rate
# data = re_xy[:,:,1:]*1
### for beta
data = measure_mu / measure_mu_ex
fig, ax = plt.subplots()
cax = ax.matshow(data[:, :, 0], cmap='gray')
fig.colorbar(cax)

def update(frame):
    ax.clear()
    cax = ax.matshow(data[:, :, frame], cmap='gray')
    ax.set_title(f"Iteration {frame+1}")
    return cax,

# Create the animation
ani = FuncAnimation(fig, update, frames=data.shape[-1], blit=False)

plt.show()

# %% finite time scaling
tts = np.array([100,500,1000,2500, 5000,7500, 10000])
var_r = np.zeros(len(tts))
mea_r = var_r*1

fig, axs = plt.subplots(1, len(tts), figsize=(19, 7))
axs = axs.flatten()
for ti in range(len(tts)):
    temp_re = re_xy[:,:,:tts[ti]]  ### truncating long simulation
    temp_mean = np.mean(temp_re,2).reshape(-1)
    var_r[ti] = np.var(temp_mean)
    mea_r[ti] = np.mean(temp_mean)
    
    ax = axs[ti]
    ax.hist(temp_mean,50)
    ax.set_title(f'T={tts[ti]}', fontsize=20)
    ax.set_xlim([0,0.4])
    
# %%
plt.figure()
plt.plot(tts, var_r, '-o')
plt.xlabel('sim length', fontsize=20)
plt.ylabel('population var', fontsize=20)

plt.figure()
plt.errorbar(tts, mea_r, yerr=var_r**0.5, fmt='-o')
plt.xlabel('sim length', fontsize=20)
plt.ylabel('population mean', fontsize=20)