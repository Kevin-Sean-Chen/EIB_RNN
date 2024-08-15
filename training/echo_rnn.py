# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:57:42 2024

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

# %% init random chaotic network that roughly matches the 2D case!
N = 50  # neurons
NN = N**2

# %% functions
def dilute_net(size, pc):
    cij = np.zeros((size, size))
    randM = np.random.rand(size,size)
    pos = np.where(randM < pc)
    cij[pos] = 1
    return cij

def phi(x):
    # nl = np.where(x > 0, x, 0)  # ReLU nonlinearity
    nl = np.where(x > 0, np.tanh(x), 0)  # nonlinearity for rate equation
    return nl

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def g_kernel(sigma, size):
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

# %% EI-RNN setup
### network and weights
Ne = NN*1  # E neurons
Ni = NN*1  # I neurons

finite_scale = 2. ### this is hand-tuned to correct for finite size
### N = total number of neurons.
### K = mean number of connections per neuron
pc = 0.5 # connection probability (fraction that are 0)
K = pc*(Ne)  # degree of connectivity
rescale_c = 1/(K**0.5)*finite_scale  # am not sure if this is correct... but might be for finite size network
c_ee, c_ei, c_ie, c_ii = dilute_net(Ne, pc), dilute_net(Ne, pc),\
                         dilute_net(Ne, pc), dilute_net(Ne, pc)

### weights and rescaling
Jee = 1.0*rescale_c  # recurrent weights
Jei = -2.0*rescale_c
Jie = 1.0*rescale_c
Jii = -1.8*rescale_c # -1.8
Je0 = 1.*1 #rescale_c   # does NOT scale with K if we are using a baseline!
Ji0 = .8*1 #rescale_c #0.8

### time scales
tau = 0.005  # in seconds
dt = 0.001  # time step
T = .5
time = np.arange(0, T, dt)
lt = len(time)

# %% simple vanilla RNN setup
g = 1.5  ### important parameter for chaotic dynamics
pc = 0.5  ### sparsity parameter
scale = 1.0/np.sqrt(pc*NN)  #scaling connectivity
M = np.random.randn(NN,NN)*g*scale
sparse = np.random.rand(NN,NN)
#M[sparse>p] = 0
mask = np.random.rand(NN,NN)
mask[sparse>pc] = 0
mask[sparse<=pc] = 1
Jij = M * mask

# %% create target I/O stimuli
### create a 2D spatial pattern
N = N*1  # Size of the image
T = lt   # Number of time steps

### spatial pattern
sigma_xy = 0.1
tau_stim = .5
mu = 0
sig_noise = 2
temp_space = np.random.randn(N,N)
temp_k = g_kernel(sigma_xy, N)
pattern = spatial_convolution(temp_space, temp_k)

angt = np.zeros(lt)
for tt in range(lt-1):
    ang = angt[tt] + dt/tau_stim*(mu - angt[tt]) + sig_noise*np.sqrt(dt)*np.random.randn()
    angt[tt+1] = wrap_to_pi(ang)

angt = np.sin(time/dt/np.pi/20)*np.pi
distance = 100  # Fixed distance to shift

plt.figure()
plt.plot(angt)

# %%
# Step 3: Initialize the 3D matrix to store the shifted images
shifted_images = np.zeros((N, N, T))

# Step 4: Loop over time and shift the image with varying angle
for i, angle in enumerate(angt):
    # Decompose the shift into x and y components
    shift_x = distance * np.cos(angle)
    shift_y = distance * np.sin(angle)

    # Use scipy.ndimage.shift to apply the shift with wrap mode for periodic boundaries
    shifted_image = shift(pattern, shift=[shift_y, shift_x], mode='wrap')

    # Store the shifted image in the 3D matrix
    shifted_images[:, :, i] = shifted_image

# %% take a look
data = shifted_images*1
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

# %%
###############################################################################
# %% neural dynamics with training !!!
beta = 1
NN = N*N  # real number of neurons
w_dim = 1000
subsamp = random.sample(range(NN), w_dim)
P = np.eye(w_dim)
w = np.random.randn(w_dim)*0.1
reps = 7

### I-O setup
I_xy = shifted_images*1  # 2D input video
Iamp = 1/np.max(I_xy)*1
f_t = angt*1  # target
y_t = np.zeros(lt)  # readout

# %% training loop!
for rr in range(reps):
    print(rr)
    
    ### testing with rand init each round ########
    rt = np.zeros((NN, lt))
    rt_init = np.random.rand(NN)*1.
    rt[:,0] = rt_init
    xt = rt*1
    ##############################################
    for tt in range(lt-1):
        ### RNN dynamics
        It = I_xy[:,:,tt].reshape(-1)  ### vectorized spatial pattern
        xt[:,tt+1] = xt[:,tt] + dt/tau*(-xt[:,tt] + Jij @ rt[:,tt] + It*Iamp)  # RNN form
        rt[:,tt+1] = np.tanh(xt[:,tt])
        
        ### training linear readout
        temp = rt[:,tt+1]*1
        x = temp[subsamp]  ### same subsample as 2D RNN
        
        y_t[tt] = np.dot(w,x)
        E_t = y_t[tt] - f_t[tt]
        dP = - beta/(1+x@P@x) * P @ np.outer(x,x) @ P  # IRLS
        P += dP
        dw = -E_t* x @ P
        w += dw
        
# %%
plt.figure()
plt.plot(y_t, label='readout')
plt.plot(f_t, label='target')
plt.legend(fontsize=20)
plt.ylabel('drift angle', fontsize=20)
plt.xlabel('time steps', fontsize=20)
plt.title('training (RLS)', fontsize=20)

# %% dynamics with EI RNN for a better match...
# vet = np.zeros((Ne, lt))
# vet[:,0] = np.random.randn(Ne)
# vit = np.zeros((Ni, lt))
# vit[:,1] = np.random.randn(Ni)
# ret = vet*1
# rit = vit*1
# measure_e = np.zeros(lt)
# measure_i = np.zeros(lt)

# ### for stimulus scanning
# stim = np.zeros(lt) + Je0
# # stim[lt//2:] = 2  # lifting input

# # amps = np.array([1,1.5,2, 2.5,3])
# # ei_responses = np.zeros(len(amps))
# # for ss in range(len(amps)):
# #     stim = np.zeros(lt) + Je0
# #     stim[lt//2:] = amps[ss]

# for tt in range(lt-1):
#     ### EI-RNN dynamics
#     vet[:,tt+1] = vet[:,tt] + dt/tau*( -vet[:,tt] + Jee*c_ee@phi(vet[:,tt]) + Jei*c_ei@phi(vit[:,tt]) + Je0*0 + stim[tt])
#     vit[:,tt+1] = vit[:,tt] + dt/tau*( -vit[:,tt] + Jie*c_ie@phi(vet[:,tt]) + Jii*c_ii@phi(vit[:,tt]) + Ji0)
#     ret[:,tt+1] = phi(vet[:,tt+1])*1
#     rit[:,tt+1] = phi(vit[:,tt+1])*1
    
#     ### measuring the input current
#     measure_e[tt+1] = (Jee*c_ee@phi(vet[:,tt+1]) + Je0)[30]
#     measure_i[tt+1] = (Jei*c_ei@phi(vit[:,tt+1]))[30]
        
# %% testing!!!
y_test = np.zeros(lt)
### initial condition
rt = np.zeros((NN, lt))
rt_init = np.random.rand(NN)*1.
### perturbations
rt[:,0] = rt_init + np.random.rand(NN)*.000 #

for tt in range(lt-1):
    ### neural dynamics
    It = I_xy[:,:,tt].reshape(-1)  ### vectorized spatial pattern
    xt[:,tt+1] = xt[:,tt] + dt/tau*(-xt[:,tt] + Jij @ rt[:,tt] + It*Iamp)  # RNN form
    rt[:,tt+1] = np.tanh(xt[:,tt])
    
    ### training linear readout
    temp = rt[:,tt+1]*1
    x = temp[subsamp]  ### same subsample as 2D RNN
    
    y_test[tt] = np.dot(w,x)

# %%
plt.figure()
plt.plot(y_test, label='readout')
# plt.plot(y_test_right,alpha=.5, label='alter sigma_i')
plt.plot(f_t, label='target')
plt.legend(fontsize=20)
plt.ylabel('drift angle', fontsize=20)
plt.xlabel('time steps', fontsize=20)
plt.title('testing', fontsize=20)
