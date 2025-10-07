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

np.random.seed(1) #1, 37

# %% finite scale analysis
### given the NxN scaling check, we want to confirm that in the chaotic regime,
### the variance for neural firing rate should decrease as simulation lengthens,
### this should proof that the system is truely ergotic in long time scale
###############################################################################

# %% test to simplify
N = 41 #50  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1* 1.  # spatial kernel
tau_i, sig_i = 15*0.001, 0.20* 1.   ### important parameters!!
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
# rescale = 10. #7 #N/2  #8 20 30... linear with N
# Wee = 1. *rescale  # recurrent weights
# Wei = -2. *rescale
# Wie = 1. *rescale
# Wii = -2. *rescale
# mu_e = .1 *1
# mu_i = .1 *1

# %% network setup
### setting up space and time
dt = 0.001  # 1ms time steps
T = 1 #10.  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
space_vec = np.linspace(0,1,N)
kernel_size = 37 #N-1 #37  # pick this for numerical convolution

### stim
stim = (np.sin(time*100)+1)/2 *0.
# stim[lt//2:lt//2+50] = np.arange(0,50)/50
# stim[lt//2+50:] = 1
# plt.plot(stim)
stim_pattern = np.random.randn(N,N)*0 #np.zeros((N,N))
stim_pattern[10:20, 10:20] = 1.

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
    he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) + stim[tt]*stim_pattern) \
                        + np.random.randn(N,N)*0. 
                        
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
offset = 50
plt.figure()
plt.plot(time[offset:], measure_e[offset:], label='E')
plt.plot(time[offset:], measure_i[offset:], label='I')
plt.plot(time[offset:], (measure_e+measure_i)[offset:],label='total')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('current', fontsize=20)
plt.title('spatial balancing', fontsize=20)
plt.legend(fontsize=15)

# %% plot dynamics
offset = 1
plt.figure()
plt.plot(time[offset:], re_xy[20,20,offset:].squeeze())
plt.plot(time[offset:], re_xy[15,15,offset:].squeeze())
plt.plot(time[offset:], ri_xy[15,15,offset:].squeeze(),'-o')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('rate (Hz)', fontsize=20)
# plt.xlim([0.1,0.14])

# %% pot beta for a single cell
beta_i = np.abs(measure_e + measure_i)/measure_e

plt.figure()
plt.plot(time, beta_i)
plt.xlabel('time', fontsize=20)
plt.ylabel('beta', fontsize=20)

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
tts = np.array([100,500,1000,2500, 5000])#,7500, 10000])
var_r = np.zeros(len(tts))
mea_r = var_r*1
mea_beta = var_r*1
var_beta = var_r*1

fig, axs = plt.subplots(1, len(tts), figsize=(19, 7))
axs = axs.flatten()
for ti in range(len(tts)):
    ### activity
    temp_re = re_xy[:,:,:tts[ti]]  ### truncating long simulation
    temp_mean = np.mean(temp_re,2).reshape(-1)
    var_r[ti] = np.var(temp_mean)
    mea_r[ti] = np.mean(temp_mean)
    
    ### balance
    beta_t = measure_mu[:,:,:tts[ti]] / measure_mu_ex[:,:,:tts[ti]] 
    temp_beta = np.nanmedian(beta_t,2).reshape(-1)
    mea_beta[ti] = np.mean(temp_beta)
    var_beta[ti] = np.var(temp_beta)
    
    ax = axs[ti]
    ax.hist(temp_mean,50)
    ax.set_title(f'T={tts[ti]}', fontsize=20)
    # ax.set_xlim([0,0.4])
    
# %% for mean rate
plt.figure()
plt.plot(tts, var_r, '-o')
plt.xlabel('sim length', fontsize=20)
plt.ylabel('population var', fontsize=20)

plt.figure()
plt.errorbar(tts, mea_r, yerr=var_r**0.5, fmt='-o')
plt.xlabel('sim length', fontsize=20)
plt.ylabel('population mean', fontsize=20)

# %% for balance
plt.figure()
plt.plot(tts, var_beta, '-o')
plt.xlabel('sim length', fontsize=20)
plt.ylabel('beta var', fontsize=20)

plt.figure()
plt.errorbar(tts, mea_beta, yerr=var_beta**0.5, fmt='-o')
plt.xlabel('sim length', fontsize=20)
plt.ylabel('beta mean', fontsize=20)

# %% for dimension
def pca_dim(re_xy, dim=0.9):
    N = re_xy.shape[0]
    flatten_r = re_xy.reshape(N**2, lt)
    uu,ss,vv = np.linalg.svd(np.cov(flatten_r))
    cums = np.cumsum(ss/np.sum(ss))
    if len(np.where(cums<dim)[0])>0:
        dim = np.where(cums<dim)[0][-1] + 1
    else: dim=0
    return dim, cums

dim,cums = pca_dim(re_xy)
print('dimension is: ', dim)

# %% sparsity
def sparse_theta(data):
    # Compute non-zero fraction along the T dimension for each (i, j)
    non_zero_fraction = 1-np.count_nonzero(data, axis=2) / data.shape[2]

    # Average this fraction over all (i, j)
    average_fraction = np.mean(non_zero_fraction)

    return average_fraction

# %% stim response
flat_re = re_xy[10:15,10:15,:].reshape(-1, lt)
plt.figure()
plt.plot(np.mean(flat_re[:, 450:600],0))
plt.plot(stim[450:600])

# %% scaling with size
###############################################################################
# %%
def sim_2D_EI(N, lt=lt, k_size=kernel_size, iptI = 0, stim=stim, sigs=(sig_e, sig_i), scale=1):
    
    sig_e, sig_i = sigs
    sig_e, sig_i = sig_e*scale, sig_i*scale
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
    # mu_e = .1 *1
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
                            + 0#stim[tt]*iptI
                        
        hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
        re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])       
        
        ### mean measurements
        measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
        measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
    
    beta_t = measure_mu / measure_mu_ex 
    return re_xy, beta_t

# %% scanning size
#######################
# repeat this for error bar #### check beta histogram!!
#######################
reps = 1
Ns = np.array([30, 40, 50, 60, 70, 80])-1
Ss = np.array([0, 0.1, 0.2,0.3, 0.4, 0.5])
Sigs = np.array([0.125, 0.15, 0.175, 0.2, 0.225, 0.25])
Sigs = np.array([0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025])
T = 1.
time = np.arange(0, T, dt)
lt = len(time)
stim = np.sin(time*100)

var_r = np.zeros(len(Ns))
mea_r = var_r*1
mea_beta = var_r*1
var_beta = var_r*1
beta_store = []
dims, cums = np.zeros((len(Ns), reps)), []
sparsity = np.zeros((len(Ns), reps))

for ni in range(len(Ns)):
    temp_mean = []
    temp_beta = []
    temp_beta_raw = []
    temp_dim = []
    for rr in range(reps):
        print(ni)
        ### sim network
        re_xyi, beta_it = sim_2D_EI(Ns[ni], lt, k_size=29)#Ns[ni]-1)
        # re_xyi, beta_it = sim_2D_EI(Ns[2], lt, k_size=37, iptI=Ss[0], stim=stim, sigs=(sig_e, 0.2)) ############ N-1 #############
        ### activity
        temp_mean.append(np.mean(re_xyi,2).reshape(-1))
        ### beta
        temp_beta.append(np.nanmedian(beta_it[:,:,50:], 2).reshape(-1))
        temp_beta_raw.append(beta_it[:,:,lt-1].reshape(-1))
        temp_beta_raw.append(beta_it[:,:,lt//2].reshape(-1))
        dim_, cum_ = pca_dim(re_xyi)
        dims[ni, rr] = dim_
        sparsity[ni, rr] = sparse_theta(re_xyi)
        
    ### record
    var_r[ni] = np.var(np.array(temp_mean))
    mea_r[ni] = np.mean(np.array(temp_mean))  
    mea_beta[ni] = np.mean(np.array(temp_beta))
    var_beta[ni] = np.var(np.array(temp_beta))
    
    beta_store.append(np.concatenate(temp_beta_raw)) ### store all beta_t
    cums.append(cum_)

# %% plotting
plt.figure()
plt.plot(Ns**2, var_r, '-o')
plt.xlabel('netwotk size', fontsize=20)
plt.ylabel('population var', fontsize=20)

plt.figure()
plt.errorbar(Ns**2, mea_r, yerr=var_r**0.5, fmt='-o')
plt.xlabel('network size', fontsize=20)
plt.ylabel('population mean', fontsize=20)

plt.figure()
plt.plot(Ns**2, var_beta, '-o')
plt.xlabel('network size', fontsize=20)
plt.ylabel('beta var', fontsize=20)

plt.figure()
plt.errorbar(Ns**2, mea_beta, yerr=var_beta**0.5, fmt='-o')
plt.xlabel('network size', fontsize=20)
plt.ylabel('beta mean', fontsize=20)

plt.figure()
plt.plot(Ns**2, sparsity, '-o')
plt.xlabel('network size', fontsize=20)
plt.ylabel('sparsity', fontsize=20)

# %% raw betas
plt.figure()
for ii in range(len(beta_store)):
    shuff_x = Ns[ii]**2 + np.random.randn(len(beta_store[ii]))*50
    plt.plot(shuff_x, beta_store[ii],'k.',alpha=.01)
plt.xlabel('size N',fontsize=20); plt.ylabel('mean beta',fontsize=20)
    
# %% violin
x_positions = np.arange(len(Ns))
data_arrays = beta_store[:]*1

# Initialize figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Loop through each x and data array to add violin plots
for x, values in zip(x_positions, data_arrays):
    parts = ax.violinplot(
        dataset=values,
        positions=[x],
        widths=0.5,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    # Customize each violin
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

# Format plot
ax.set_xticks(x_positions)
ax.set_xticklabels([f"{label:.0f}" for label in Ns**2-1]) 
ax.set_xlabel("network size", fontsize=25)
ax.set_ylabel(r"$\beta_t$", fontsize=25)
ax.grid(True)

# %% dimension scaling
plt.figure()
plt.plot(Ns**2,dims,'-o') #/Ns[:,None]**2,'-o')
plt.xlabel('size N',fontsize=20); plt.ylabel('dimension',fontsize=20)
# plt.title('MF/N', fontsize=20)

# %% sensitivity
plt.figure()
plt.plot(Ss,dims,'-o') #/Ns[:,None]**2,'-o')
plt.xlabel('input strength',fontsize=20); plt.ylabel('dimension',fontsize=20)

# %% scale
plt.figure()
plt.plot(Sigs,dims,'-o') #/Ns[:,None]**2,'-o')
plt.xlabel(r'$\tau_i$',fontsize=20); plt.ylabel('dimension',fontsize=20)