# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:14:44 2024

@author: kevin
"""

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

# %% comparison of 1/N vs. 1/sqrt(N) network
### weak connection on 1/N order, which potentially generates unbalanced 
### strong connection on 1/sqrt(N) order, with balanced conditions

# %% parameter settings
N = 100  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
tau_i = 0.015
sig_i = 0.2    ### important parameters!!

amp = 0

# %% balanced condition
rescale = .5

Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale*1
mu_i = .8*rescale*1

# Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
# Wei = -1.5*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# Wie = .9*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
# Wii = -1.6*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# mu_e = 1.*rescale*1
# mu_i = .6*rescale*1

stim_scal = amp #/ N**0.5

# %% unbalanced condition
###########################
### find a way to confirm 1/N connection!!
### find MF chaos that is NOT balanced...
###########################
# rescale = 3 #N/2  #8 20 30... linear with N

# Wee = 1. *rescale  # recurrent weights
# Wei = -1. *rescale
# Wie = 1. *rescale
# Wii = -1. *rescale
# mu_e = .01 *1
# mu_i = .01 *1

# Wee = 1. *rescale  # recurrent weights
# Wei = -2. *rescale
# Wie = .99 *rescale
# Wii = -1.8 *rescale
# mu_e = 1 *1
# mu_i = .8 *1

# stim_scal = amp #/ N

# %% network setup
### setting up space and time
dt = 0.001  # 1ms time steps
T = 1.0  # a few seconds of simulation
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

def spatial_convolution(r,k):
    """
    2D spatial convolution given kernel k and neural field r
    """
    gr = sp.signal.convolve2d(r.squeeze(), k, mode='same',  boundary='wrap') #, fillvalue=0,
    return gr

# %% stimuli
### spatial pattern
sigma_xy = 0.1
temp_space = np.random.randn(N,N)
temp_k = g_kernel(sigma_xy, N)
pattern = spatial_convolution(temp_space, temp_k)
pattern = pattern/np.max(pattern)
# pattern = 1
### time series
stim = ((np.sin(time/np.pi*50)+1)/2)*stim_scal
# plt.plot(stim)

# %% dynamics

### neural dynamics
for tt in range(lt-1):
    ### stim
    # if tt>lt//2:
        # mu_e = 5
    
    ### modifying for 2D rate EI-RNN
    ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e)  + stim[tt]*pattern )
    hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
    re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
    ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
    
    # if tt>100 and tt<110:
    #     re_xy[60:,60:,tt+1] = 1
    
    ### make E-I measurements
    measure_e[tt+1] = (Wee*ge_conv_re + mu_e)[20,20]
    measure_i[tt+1] = (Wei*gi_conv_ri)[20,20]
    
    
    ### mean measurements
    measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    # measure_mu[:,:,tt+1] = (  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
    
# %%
offset = 50
plt.figure()
plt.plot(time[offset:], measure_e[offset:], label='E')
plt.plot(time[offset:], measure_i[offset:], label='I')
plt.plot(time[offset:], (measure_e+measure_i)[offset:],label='total')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('current', fontsize=20)
# plt.title('spatial balancing', fontsize=20)
plt.legend(fontsize=15)

# %% plot dynamics
offset = 1
plt.figure()
plt.plot(time[offset:], re_xy[20,20,offset:].squeeze(), label='E cell1')
plt.plot(time[offset:], re_xy[15,15,offset:].squeeze(), label='E cell2')
plt.plot(time[offset:], ri_xy[15,15,offset:].squeeze(), label='I cell2')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('activity', fontsize=20)
plt.legend(loc='lower center', bbox_to_anchor=(1, -.3), fontsize=7)
# Adjust the layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.75, 1])
# plt.xlim([0.1,0.14])

# %% pot beta for a single cell
beta_i = np.abs(measure_e + measure_i)/measure_e

plt.figure()
plt.plot(time, beta_i)
plt.xlabel('time', fontsize=20)
plt.ylabel('beta', fontsize=20)
print(np.abs(np.mean(measure_e + measure_i)) / np.mean(measure_e))


# %% visualize
### for rate
data = re_xy[:,:,1:]*1
### for beta
# data = measure_mu / measure_mu_ex
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
# %% ideas for analysis:
    ### spectrum
    ### perturbation
    ### "computation":
        ### reliability for readout!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ### diversity for high-D
        ### noise tolerance???
    
    ### idea:
        ### scan through spectrum for balanced vs. unbalanced
        ### plot beta vs. MSE, with dots across parameters
        
# %% spectral analysis
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

###
plt.figure()
test,ff = group_spectrum(re_xy[:,:,offset:])
plt.loglog(ff,test)
# plt.plot(ff[2:],test[2:])
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power', fontsize=20)

# %% 'beta' parameter and rate distribution
plt.figure()
# plt.hist(re_xy.reshape(-1),100)
plt.hist(np.mean(re_xy[:,:,offset:],2).reshape(-1),50)
plt.xlabel(r'$<r>_t$', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.title(f'N= {N}', fontsize=20)

beta_t = measure_mu / measure_mu_ex  # beta dynamics
plt.figure()
plt.hist(beta_t.reshape(-1),100)
plt.xlabel(r'$\beta_t$', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.title(f'N= {N}', fontsize=20)
plt.xlim([0,6])
print('median of beta: ', np.nanmedian(beta_t))  #nanmedian?

# %% do RLS
###############################################################################
plt.figure()
plt.hist(beta_mf2.reshape(-1), bins=100, alpha=0.5, label='mean-field 1')
plt.hist(beta_mf.reshape(-1), bins=500, alpha=0.5, label='mean-field 2')
plt.hist(beta_ei.reshape(-1), bins=100, alpha=0.5, label='EI')
plt.xlim([0,5])
plt.legend(fontsize=20)
plt.xlabel('beta', fontsize=20); plt.ylabel('counts', fontsize=20)