# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:47:17 2024

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
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.set_zlabel("z", fontsize=20)
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

# %% cheating with 2D basis
### spatial pattern
sigma_xy = 0.1
tau_stim = .5
mu = 0
sig_noise = 2
temp_space1 = np.random.randn(N,N)
temp_space2 = np.random.randn(N,N)
temp_k = g_kernel(sigma_xy, N)
pattern1 = spatial_convolution(temp_space1, temp_k)
pattern2 = spatial_convolution(temp_space2, temp_k)

temp_space_w = np.random.randn(N,N)
pattern_w = spatial_convolution(temp_space_w, temp_k)

lorenz_xy = np.zeros((N,N, lt))

for tt in range(lt):
    lorenz_xy[:,:,tt] = x[tt]*pattern1 + z[tt]*pattern2
    
# %% show video
data = lorenz_xy*1
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

# %% function to generate Lorenz pattern
def gen_spatail_Lorenz():
    # Parameters
    dt = 0.02  # Time step
    num_steps = 550  # Number of steps
    
    # Initial conditions
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    z = np.zeros(num_steps)
    x[0], y[0], z[0] = 0.0 + \
        np.random.randn()*0.1, 1.0 + np.random.randn()*0.1, 1.05 + np.random.randn()*0.1  # Initial position

    # Euler method
    for i in range(1, num_steps):
        dx, dy, dz = lorenz(x[i-1], y[i-1], z[i-1])
        x[i] = x[i-1] + dx * dt
        y[i] = y[i-1] + dy * dt
        z[i] = z[i-1] + dz * dt
    ### processing for image
    offset = 50
    x = x[offset:]
    y = y[offset:]
    z = z[offset:]
    x = (x - np.mean(x))/np.max(x)
    y = (y - np.mean(y))/np.max(y)
    z = (z - np.mean(z))/np.max(z)
    N = 50
    lt = num_steps*1 - offset

    ### make spatial patterns
    temp_space1 = np.random.randn(N,N)
    temp_space2 = np.random.randn(N,N)
    temp_k = g_kernel(sigma_xy, N)
    pattern1 = spatial_convolution(temp_space1, temp_k)
    pattern2 = spatial_convolution(temp_space2, temp_k)

    lorenz_xy = np.zeros((N,N, lt))

    for tt in range(lt):
        lorenz_xy[:,:,tt] = x[tt]*pattern1 + z[tt]*pattern2
        
    return lorenz_xy, y

lorenz_xy_test, f_test = gen_spatail_Lorenz()
I_xy_test = lorenz_xy_test/np.max(lorenz_xy_test)

# %% setup
N = 50
lt = num_steps*1 - offset

# %% testing with Lorenz network
###############################################################################
### initialization
# dt = 0.005
# lt = 1500
# N = 50
# xt = np.zeros((N,N, lt))
# yt = xt*1
# zt = xt*1
# yt[:,:,0] = np.random.randn(N,N)
# zt[:,:,0] = np.random.randn(N,N)
# ### parameter setting
# sig = 0.1
# alpha = 10
# beta = 8/3
# rho = np.random.rand(N,N)*40 + 30

# # %% run spatiotemporal dynamics
# for tt in range(lt-1):
#     Aij_yj = spatial_convolution(yt[:,:,tt], g_kernel(sig)*1) #/np.max(g_kernel(sig))/10)
#     xt[:,:,tt+1] = xt[:,:,tt] + dt*(-alpha*xt[:,:,tt] + alpha/2* Aij_yj)
#     yt[:,:,tt+1] = yt[:,:,tt] + dt*(xt[:,:,tt]*(rho - zt[:,:,tt]) - yt[:,:,tt])
#     zt[:,:,tt+1] = zt[:,:,tt] + dt*(xt[:,:,tt]*yt[:,:,tt] - beta*yt[:,:,tt])

###############################################################################
# %% training the network
# %%
###############################################################################
# %% init 2D chaotic network
dt = 0.001
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
tau_i, sig_i = 5*0.001, 0.11   ### important parameters!!
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
w_dim = 1000
subsamp = random.sample(range(NN), w_dim)
P = np.eye(w_dim)
w = np.random.randn(w_dim)*0.1
reps = 1 

### I-O setup
I_xy = lorenz_xy/np.max(lorenz_xy)  # 2D input video
Iamp = 2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale / 1
f_t = y*1  # target
y_t = np.zeros(lt)  # readout

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
    ### train with different Lorenz everytime! ###
    lorenz_xy_train, f_train = gen_spatail_Lorenz()
    I_xy = lorenz_xy_train/np.max(lorenz_xy_train)
    ##############################################
    
    for tt in range(lt-1):
        ### neural dynamics
        ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
        gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
        he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e \
                                                                       + I_xy[:,:,tt]*Iamp + Iamp*y_t[tt]*pattern_w) )
        hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
        re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
        
        ### training linear readout
        temp = re_xy[:,:,tt+1].reshape(-1)
        xx = temp[subsamp]
        
        y_t[tt] = np.dot(w, xx)
        E_t = y_t[tt] - f_train[tt]
        dP = - beta/(1+xx@P@xx) * P @ np.outer(xx,xx) @ P  # IRLS
        P += dP
        dw = -E_t* xx @ P
        w += dw
    
# %%
plt.figure()
plt.plot(y_t, label='readout')
plt.plot(f_train, label='target')
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
re_xy[:,:,0] = re_init + np.random.rand(N,N)*1 #
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
                                                                   + I_xy[:,:,tt]*Iamp + Iamp*y_test[tt]*pattern_w) )
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
plt.plot(f_train, label='target')
plt.legend(fontsize=20)
plt.ylabel('Lorenz z(t)', fontsize=20)
plt.xlabel('time steps', fontsize=20)
# plt.title('testing', fontsize=20)
