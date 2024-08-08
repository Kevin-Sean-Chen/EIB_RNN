# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:45:27 2024

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

# %% proposal: Exploring computation through spatiotemporal neural dynamics
###############################################################################
# Simple implementation of rate model from Noga Mosheiff, Bard Ermentrout, Chengcheng Huang 2023
# which is a rate-based model for ChengCheng et al. 2019 (with spiking population)
# The plan is to implement 2D chaotic neural model and explore its dynamics/computation
# then study the input-output computation of this model, as well as effects from random connections (is there a transition?)
# further exporations: 
#   1. effects of short-term synaptic plasiticity in space
#   2. possible optimization for tasks
#   3. fitting to data??

### Haim's suggestions: (here till Sunday and back on Thursday)
    # use ReLu, order one units, and check equations and response timescales!!
    # is it really balanced??
    
    # if yes, is it fluctuating around coherence? how does the dilution affect?
    # even if not, anaylize initial condition effects and find the attractor through spatiotemporal analysis
    # further think about encoding and working memory

###############################################################################
# %% network parameters
# N = 50  # neurons
# Wee = 80  # recurrent weights
# Wei = -160
# Wie = 80
# Wii = -150
# tau_e = 0.005  # time constant ( 5ms in seconds )
# sig_e = 0.1  # spatial kernel
# mu_e = 0.48*10  # offset
# mu_i = 0.32*10
# tau_i, sig_i = 9*0.001, 0.096
### A, traveling waves solution (τi = 8, σi = 0.1). 
### B, alternating bumps solution (τi = 9, σi = 0.06). 
### C, alternating stripes solution (τi = 9, σi = 0.1). 
### D, chaotic solution (τi = 12.8, σi = 0.096).

# %% test to simplify
N = 70  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
tau_i, sig_i = 10*0.001, 0.1   ### imporat parameters!!
### moving dots 5ms, 0.2
### little coherence 5ms, 0.1
### chaotic waves 10ms, 0.1
### drifting changing blobs 10ms, 0.2

rescale = 2. ##(N*sig_e*np.pi*1)**0.5 #1
Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale #*(N*sig_i*np.pi*1)**0.5 *rescale  #1e-8#.001*1  # offset
mu_i = 1.*rescale #*(N*sig_i*np.pi*1)**0.5 *rescale  #1e-8#.001*1

# %% network setup
### setting up space and time
dt = 0.001  # 1ms time steps
T = .5  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
space_vec = np.linspace(0,1,N)
kernel_size = 20  # pick this for numerical convolution

### random initial conditions
re_xy[:,:,0] = np.random.rand(N,N)*.1
ri_xy[:,:,0] = np.random.rand(N,N)*.1
he_xy = re_xy*1
hi_xy = ri_xy*1
measure_e = np.zeros(lt)
measure_i = np.zeros(lt)

def phi(x):
    """
    rectified quardratic nonlinearity
    """
    # nl = np.where(x > 0, x**2, 0)
    # nl = np.where(x > 0, x*10, 0)  ### why a scaling factor needed!?????????????????????
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
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)), 
        (size, size)
    )
    return kernel / np.sum(kernel)

# %% dynamics
def spatial_convolution(r,k):
    """
    2D spatial convolution given kernel k and neural field r
    """
    gr = sp.signal.convolve2d(r.squeeze(), k, mode='same',  boundary='wrap') #, fillvalue=0,
    # gr = sp.signal.convolve2d(r.squeeze(), k, mode='same', boundary='symm')
    ###########################################################################
    #### double check on setting boundary conditions!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ###########################################################################
    return gr

### neural dynamics
for tt in range(lt-1):
    ### stim
    # if tt>lt//2:
        # mu_e = 5
    ### Noga Mosheiff paper
    # ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    # gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    # re_xy[:,:,tt+1] = re_xy[:,:,tt] + dt/tau_e*( -re_xy[:,:,tt] + phi(Wee*ge_conv_re + Wei*gi_conv_ri + mu_e) )
    # ri_xy[:,:,tt+1] = ri_xy[:,:,tt] + dt/tau_i*( -ri_xy[:,:,tt] + phi(Wie*ge_conv_re + Wii*gi_conv_ri + mu_i) )
    
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
    
# %%
offset = 50
plt.figure()
plt.plot(time[offset:], measure_e[offset:])
plt.plot(time[offset:], measure_i[offset:])
plt.plot(time[offset:], (measure_e+measure_i)[offset:])
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('current', fontsize=20)
plt.title('spatial balancing', fontsize=20)

# %% plot dynamics
offset = 1
plt.figure()
plt.plot(time[offset:], re_xy[20,20,offset:].squeeze())
plt.plot(time[offset:], re_xy[15,15,offset:].squeeze())
plt.plot(time[offset:], ri_xy[15,15,offset:].squeeze(),'-o')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('rate (Hz)', fontsize=20)
# plt.xlim([0.1,0.14])

# %%
# plt.figure()
# plt.plot(time[offset:]-.25, np.mean(np.mean(re_xy[:,:,offset:],0),0), label='sigma=.1')
# plt.plot(time[offset:]-.25, temp, label='sigma=.15')
# plt.legend(fontsize=20)
# plt.xlim([-0.04, 0.04])

# %% visualize
data = re_xy[:,:,1:]*1
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

# %% make video
###Generate example data (random 10x10x100 tensor)
# gif_name = 'wave_like'
# data = re_xy[:,:,100:]*1

# # Function to create a frame with the iteration number in the title
# def create_frame(data, frame):
#     fig, ax = plt.subplots()
#     ax.imshow(data[:, :, frame], cmap='viridis')
#     ax.set_title(f'Iteration: {frame}')
#     plt.colorbar(ax.images[0], ax=ax)
#     plt.close(fig)  # Close the figure to avoid displaying it
#     return fig

# # Create and save frames as images
# frames = []
# for frame in range(data.shape[2]):
#     fig = create_frame(data, frame)
#     fig.canvas.draw()
#     # Convert the canvas to an image
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     frames.append(Image.fromarray(image))

# # Save frames as a GIF
# frames[0].save(gif_name+'.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)

# # Check if the GIF plays correctly
# from IPython.display import display, Image as IPImage
# display(IPImage(filename=gif_name+'.gif'))


