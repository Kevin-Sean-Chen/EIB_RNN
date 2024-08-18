# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:09:02 2024

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

# %% simulating a chaotic network with varying parameters
###############################################################################
# %% time varying ratio
dt = 0.001   # 1ms time step
T = 0.5 + 0.05   # in seconds
time = np.arange(0, T, dt)  # time vector
lt = len(time)

tau_t = (np.sin(time*20)+1) + 1.5  # ranging across the ratio

# Convert sine wave to sawtooth wave
sine_wave = np.sin(time*25)
sawtooth_wave = (2 * np.arcsin(sine_wave) / np.pi)  # Normalize to sawtooth wave

# Normalize the sawtooth to the same max/min as sine wave
sawtooth_wave = (sawtooth_wave - np.min(sawtooth_wave)) / (np.max(sawtooth_wave) - np.min(sawtooth_wave))  # Normalize to 0-1 range
sawtooth_wave = (sawtooth_wave * 2 - 1) * (np.max(sine_wave) - np.min(sine_wave)) / 2  # Scale to match sine wave

tau_t = (sawtooth_wave+1) + 1.5
plt.figure()
plt.plot(tau_t)
plt.xlabel('time steps', fontsize=20)
plt.ylabel('tau_i/tau_e', fontsize=20)

# %% init 2D chaotic network
N = 50  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
# tau_i, sig_i = 20*0.001, 0.2   ### important parameters!!
#### 5, 0.14  ### grid parameter
#### 15, 0.2  ### chaos parameter
#### 10, 0.11 ### waves/strips!!!
#### 8,  0.2  ### blinking
##################################

### setting up space and time
dt = 0.001  # 1ms time steps
T = .5  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
space_vec = np.linspace(0,1,N)
kernel_size = 23 #37  # pick this for numerical convolution

# %% functions
def phi(x):
    """
    rectified quardratic nonlinearity
    """
    # nl = np.where(x > 0, x**1, 0)
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

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# %% chaotic dynamics with varying ratio
### neural dynamics
def make_chaotic():
    lt_sim = lt+50
    ### random initial conditions
    re_init = np.random.rand(N,N)*.1
    ri_init = np.random.rand(N,N)*.1
    re_xy = np.zeros((N,N, lt_sim))
    ri_xy = re_xy*1
    re_xy[:,:,0] = re_init
    ri_xy[:,:,0] = ri_init
    he_xy = re_xy*1
    hi_xy = ri_xy*1
    for tt in range(lt_sim-1):
        ### varying parameters
        tau_i, sig_i = tau_t[tt]*tau_e, sig_e*2   #### time varying tau ratio
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
        he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
        hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
        re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
    re_xy = re_xy[:,:,50:]  # remove weird initial conditions
    return re_xy

re_xy = make_chaotic()
re_xy_2 = make_chaotic()   
 
# %% making it an input stimuli
sp_temp_chaos = re_xy*1
I_xy2 = re_xy_2 / np.max(re_xy_2)   # for generliaztion test

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

### random initial conditions
re_init = np.random.rand(N,N)*.0
ri_init = np.random.rand(N,N)*.0
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
re_xy[:,:,0] = re_init
ri_xy[:,:,0] = ri_init
he_xy = re_xy*1
hi_xy = ri_xy*1

# %%
beta = 1
NN = N*N  # real number of neurons
w_dim = 1500
subsamp = random.sample(range(NN), w_dim)
P = np.eye(w_dim)
w = np.random.randn(w_dim)*0.1
reps = 15

### I-O setup
I_xy = sp_temp_chaos/np.max(sp_temp_chaos)  # 2D input video
Iamp = 2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale / 1
f_t = tau_t*1  # target
y_t = np.zeros(lt)  # readout

# %% training loop!
for rr in range(reps):
    print(rr)
    
    ### test with different video per round!
    temp_stim = make_chaotic()
    I_xy = temp_stim / np.max(temp_stim)
    ### testing with rand init each round ########
    re_init = np.random.rand(N,N)
    ri_init = np.random.rand(N,N)
    re_xy = np.zeros((N,N, lt))
    ri_xy = re_xy*1
    re_xy[:,:,0] = re_init
    ri_xy[:,:,0] = ri_init
    ##############################################
    
    for tt in range(lt-1):
        ### neural dynamics
        ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
        gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
        he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e \
                                                                       + I_xy[:,:,tt]*Iamp) )
        hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
        re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
        
        ### training linear readout
        temp = re_xy[:,:,tt+1].reshape(-1)
        xx = temp[subsamp]
        
        y_t[tt] = np.dot(w, xx)
        E_t = y_t[tt] - f_t[tt]
        dP = - beta/(1+xx@P@xx) * P @ np.outer(xx,xx) @ P  # IRLS
        P += dP
        dw = -E_t* xx @ P
        w += dw
    
# %%
plt.figure()
plt.plot(y_t[:-1], label='readout')
plt.plot(f_t, label='target')
plt.legend(fontsize=20)
plt.ylabel('drift angle', fontsize=20)
plt.xlabel('time steps', fontsize=20)
plt.title('training (RLS)', fontsize=20)

# %% testing!!!
y_test = np.zeros(lt)
### initial condition
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
### perturbations
re_xy[:,:,0] = re_init + np.random.rand(N,N)*.1 #
# sig_i = 0.2
# tau_i = 0.015
ri_xy[:,:,0] = ri_init
he_xy = re_xy*1
hi_xy = ri_xy*1

for tt in range(lt-1):
    ### neural dynamics
    ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e \
                                                                   + I_xy[:,:,tt]*Iamp) )
    hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
    re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
    ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
    
    ### training linear readout
    temp = re_xy[:,:,tt+1].reshape(-1)
    x = temp[subsamp]
    
    y_test[tt] = np.dot(w,x)

# %%
plt.figure()
plt.plot(y_test[:-1], label='readout')
# plt.plot(y_test_right,alpha=.5, label='alter sigma_i')
plt.plot(f_t, label='target')
plt.legend(fontsize=20)
plt.ylabel('drift angle', fontsize=20)
plt.xlabel('time steps', fontsize=20)
plt.title('testing', fontsize=20)

