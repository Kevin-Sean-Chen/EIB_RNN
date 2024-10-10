# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:52:11 2024

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

# %% init 2D chaotic network
N = 50  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
tau_i, sig_i = 15*0.001, 0.2   ### important parameters!!
#### 5, 0.14  ### grid parameter
#### 15, 0.2  ### chaos parameter
#### 10, 0.11 ### waves/strips!!!
#### 8,  0.2  ### blinking
##################################
# EI-balanced
rescale = 1. ##(N*sig_e*np.pi*1)**0.5 #1
Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale
mu_i = .8*rescale

Iamp = 0.3 *2.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale / 1.  # same scale as the spontaneous input?

# # # unbalanced (mean-field)
rescale = 3 #N/2  #8 20 30... linear with N
Wee = 1. *rescale  # recurrent weights
Wei = -1. *rescale
Wie = 1. *rescale
Wii = -1. *rescale
mu_e = .01 *1
mu_i = .01 *1

Iamp = .5  # he~0.5 10*median?

##################################
### setting up space and time
dt = 0.001  # 1ms time steps
T = .5  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
space_vec = np.linspace(0,1,N)
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

# %% functions
def phi(x):
    """
    rectified quardratic nonlinearity
    """
    # nl = np.where(x > 0, x**2, 0)
    # nl = np.where(x > 0, x*1, 0)  ### why a scaling factor needed!?????????????????????
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

def generate_points(n, N):
    points = []
    for _ in range(n):
        x = random.randint(0, N)
        y = random.randint(0, N)
        points.append((x, y))
    return points

def kernel_readout(n, N):
    ### the version with space
    points = generate_points(n, N)
    x_vec = np.zeros(n)

    return x_vec

def fixed_readout_group(n, n_group, N):
    averages = []
    temp = np.arange(N)
    # Perform this 100 times
    for _ in range(n):
        # Randomly select 50 samples from the vector (with replacement if needed)
        selected_samples = random.choices(temp, k=n_group)
        # Compute the average of the selected samples
        averages.append(selected_samples)
    
    return np.array(averages)

def average_group(temp, w_readout):
    ### the versiton with averagin
    nd = w_readout.shape[0]
    readout = np.zeros(nd)
    for nn in range(nd):
        readout[nn] = np.mean(temp[w_readout[nn,:]])
    
    return readout


# %% create target I/O stimuli
### create a 2D spatial pattern
N = N*1  # Size of the image
T = lt   # Number of time steps

### spatial pattern
sigma_xy = 5/N #0.1
tau = .5
mu = 0
sig_noise = 2
temp_space = np.random.randn(N,N)
temp_k = g_kernel(sigma_xy, N)
pattern = spatial_convolution(temp_space, temp_k)
pattern = pattern - np.mean(pattern)  ### remove baseline for zero net
pattern = pattern / np.max(np.abs(pattern))  ### unit strength

angt = np.zeros(lt)
for tt in range(lt-1):
    ang = angt[tt] + dt/tau*(mu - angt[tt]) + sig_noise*np.sqrt(dt)*np.random.randn()
    angt[tt+1] = wrap_to_pi(ang)

# angt = np.sin(time/dt/np.pi/20)*np.pi
angt = np.sin(time/dt/np.pi/10)
angt = angt + np.sin(time/dt/np.pi/30)
angt = angt/np.max(angt)*np.pi
distance = 100  # Fixed distance to shift

### abrupt change
angt = np.zeros(lt)
angt[lt//2 : lt//2 + 20] = 2
angt[lt//2 +20 : lt//2 + 80] = -2


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

# %% neural dynamics with training !!!
beta = 1
NN = N*N  # real number of neurons
w_dim = 1000#N*1  # number of readouts
###############################################################################
### test with introducing local or random averages before linear readout~!!!!!!!!
n_group = 500  # number for averaging
w_readout = fixed_readout_group(w_dim, n_group, NN)  # fixed grouping

###############################################################################
subsamp = random.sample(range(NN), w_dim)
P = np.eye(w_dim)
w = np.random.randn(w_dim)*0.1
reps = 1

### I-O setup
I_xy = shifted_images*1  # 2D input video
f_t = angt*1  # target
y_t = np.zeros(lt)  # readout

### measure the field
measure_mu = np.zeros((N,N,lt))
measure_mu_ex = np.zeros((N,N,lt))

# %% training loop!
for rr in range(reps):
    print(rr)
    
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
        x = temp[subsamp]
        # x = average_group(temp, w_readout)  ### averaged readout
        
        y_t[tt] = np.dot(w,x)
        E_t = y_t[tt] - f_t[tt]
        dP = - beta/(1+x@P@x) * P @ np.outer(x,x) @ P  # IRLS
        P += dP
        dw = -E_t* x @ P
        w += dw
        
        ### mean measurements
        measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
        measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
    
# %%
plt.figure()
plt.plot(y_t, label='readout')
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
plt.plot(y_test, label='readout')
# plt.plot(y_test_right,alpha=.5, label='alter sigma_i')
plt.plot(f_t, label='target')
plt.legend(fontsize=20)
plt.ylabel('drift angle', fontsize=20)
plt.xlabel('time steps', fontsize=20)
plt.title('testing', fontsize=20)

# %%
MSE = np.mean((y_test[50:]-f_t[50:])**2)
print(MSE)

# %%
mse_2d = np.array([0.041, 0.074, 0.093, 0.096, 0.160, 0.162, 0.114])
mse_rnn = np.array([0.639, 0.093, 0.051, 0.499, 0.433, 0.188, 0.359])

plt.figure()
plt.bar(['2D','vanila RNN'], [np.mean(mse_2d), np.mean(mse_rnn)], yerr =  [np.std(mse_2d), np.std(mse_rnn)])
plt.ylabel('MSE', fontsize=20)