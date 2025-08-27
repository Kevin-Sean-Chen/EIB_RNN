# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:49:56 2024

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

# %% investigate driven spatiotemporal dynamics
### test for direction selectivity in 2D EI network

# %% init 2D chaotic network
N = 30
dt = 0.001
T = 1.0  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
#################################
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1*1.  # spatial kernel
tau_i, sig_i = 15*0.001, 0.2*1.   ### important parameters!!
#### 5, 0.14  ### grid parameter
#### 15, 0.2  ### chaos parameter
#### 10, 0.11 ### waves/strips!!!
#### 8,  0.2  ### blinking
##################################

### balanced parameters
rescale = 3. ##(N*sig_e*np.pi*1)**0.5 #1
Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale
mu_i = .8*rescale

# %% functions
def phi(x):
    """
    rectified quardratic nonlinearity
    """
    # nl = np.where(x > 0, x**2, 0)
    # nl = np.where(x > 0, x*1, 0)  ### why a scaling factor needed!?????????????????????
    nl = np.where(x > 0, np.tanh(x)*1, 0)
    return nl

def g_kernel(sigma, size=N, skew_x=0.00, skew_y=.0, offset=0):
    """
    Generates a 2D Gaussian kernel.
    """
    sigma = sigma*size
    x0, y0 = (size - 1) / 2+offset, (size - 1) / 2+offset*0  # Center
    # kernel = np.fromfunction(
    #     lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
    #                  np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * (sigma*1.) ** 2) + (y-x)*0.), 
    #     (size, size)
    # )
    
    # Create the skewed Gaussian function
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - x0 + skew_x * (y - y0)) ** 2 / (2 * sigma ** 2) +
                              (y - y0 + skew_y * (x - x0)) ** 2 / (2 * sigma ** 2))),
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
    # power_spectrum = power_spectrum / np.sum(power_spectrum)  ### testing normalization ###
    
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

def spectrum_space(data, dxy):
    data_fft = np.fft.fftn(data)
    N,lt = data.shape
    data_fft_shifted = np.fft.fftshift(data_fft)
    magnitude_spectrum = np.abs(data_fft_shifted)**2
    
    magnitude_spectrum = magnitude_spectrum / np.sum(magnitude_spectrum)  ### testing normalization ###
    
    # plt.imshow(np.log1p(magnitude_spectrum[:,:,lt//2]), cmap='gray')
    frequencies = np.fft.fftfreq(N, d=dxy)
    # center = (N // 2, N // 2)
    # magnitude_spectrum[N//2, N//2,:] = 0
    return magnitude_spectrum, frequencies#[:len(frequencies)//2] ## space by time
    
def group_spectrum_space(data, dxy=1/N):
    N,_,T = data.shape
    pp,ff = spectrum_space(data[:,0,:], dxy)  # now in 1D space
    # spec_all,_ = spectrum_space(data, dxy)
    spec_all = np.zeros((N,len(ff), T))  
    for ii in range(N):
        spec_all[ii,:,:],_ = spectrum_space(data[:,ii,:], dxy)
        # for jj in range(T):
            # temp = data[ii,:,jj].squeeze()
            # spec_all[ii,:,jj],_ = spectrum_space(temp, dxy)
    return np.mean(np.mean(spec_all,2),0), ff

def autocorr1(x):
    '''numpy.corrcoef, partial'''
    x = x - np.mean(x)
    lags = np.arange(0, len(x)-int(0.1*len(x)))
    corr=[np.dot(x, x)/len(x) if l==0 else np.dot(x[l:],x[:-l])/len(x[l:]) for l in lags]
    return np.array(corr)

def measure_SNR_new(r_xyt, lags=int(lt//2), post=int(tau_i/dt*5)):
    autocorrs_sigs = np.zeros(N**2)
    autocorrs_nois = np.zeros(N**2)
    kk = 0
    for ii in range(N):
        for jj in range(N):
            temp_corr = autocorr1(r_xyt[ii,jj,:])[:lags] #- np.mean(r_xyt[ii,jj,:])**2
            autocorrs_nois[kk] = temp_corr[0] - np.mean((temp_corr[post:])**2)**0.5
            autocorrs_sigs[kk] = np.mean((temp_corr[post:])**2)**0.5
            kk += 1
    return autocorrs_sigs, autocorrs_nois

# %% setup spatiotemporal drive
space_f = 3 #3
time_f = .2
direction = +1 ### +/- for direction!
x = np.linspace(0, space_f*2 * np.pi, N)  # Create a range for sine wave input
Iamp = 2.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale *1
I_xyt = np.zeros((N, N, lt)) # Initialize the tensor
# Populate the tensor
for t in range(lt):
    temp = np.array([np.sin(x*1 + phase*0 + direction*t *time_f* (2 * np.pi / N)) 
                                for phase in np.linspace(0, 2 * np.pi, N)]).T
    I_xyt[:, :, t] = temp/np.linalg.norm(temp)
    
## %% visualization
# plt.figure()
# plt.imshow(I_xyt[:,:,0]); plt.colorbar()

# %% making the randon 2D spatial pattern
def make_2D_stim(time_f, space_f,N=N):
    x = np.linspace(0, space_f*2 * np.pi, N)  # Create a range for sine wave input
    I_xyt = np.zeros((N, N, lt)) # Initialize the tensor
    # Populate the tensor
    for t in range(lt):
        temp = np.array([np.sin(x + phase + t *time_f* (2 * np.pi / N)) 
                                    for phase in np.linspace(0, 2 * np.pi, N)]).T
        I_xyt[:, :, t] = temp/np.linalg.norm(temp)
        
    return I_xyt

# %% check stim correlation
# test = I_xyt.reshape(N**2, lt)
# cross_correlation_matrix = np.corrcoef(test)
# plt.figure(figsize=(8, 8))
# plt.imshow(cross_correlation_matrix)
# plt.colorbar()
    
# %%
def make_chaotic(I_xy, N=N, lt=lt):
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
        rescale = 3. ##(N*sig_e*np.pi*1)**0.5 #1
        Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
        Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
        Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
        Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
        mu_e = 1.*rescale
        mu_i = .8*rescale
        Iamp = 2.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale *1

        ### MF parameters ###
        # rescale = 3 #N/2  #8 20 30... linear with N
        # Wee = 1. *rescale  # recurrent weights
        # Wei = -1. *rescale
        # Wie = 1. *rescale
        # Wii = -1. *rescale
        # mu_e = .01 *1
        # mu_i = .01 *1
        # Iamp = 1* rescale  # he~0.5 10*median?
        
        ### modifying for 2D rate EI-RNN
        ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
        gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i, skew_y=.0, offset=-2))
        he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e \
                                                                       + I_xy[:,:,tt]*Iamp*1) )
        hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i \
                                                                       + I_xy[:,:,tt]*Iamp*0) )
        re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
    return re_xy

# %% run all sims
### chaotic
tau_i, sig_i = 15*0.001, 0.2*1.
spontaneous_r_chaos = make_chaotic(I_xyt*0)
driven_r_chaos = make_chaotic(I_xyt*1.2)

# tau_i, sig_i = 5*0.001, 0.14
# spontaneous_r_ctr = make_chaotic(I_xyt*0)
# driven_r_ctr = make_chaotic(I_xyt)

# %% visualize
data = driven_r_chaos[:,:,1:]*1
# data = spontaneous_r_chaos[:,:,1:]*1
# data = I_xyt*1
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

# %% analyze signal
signal, noise = measure_SNR_new(driven_r_chaos)

lags = lt//2
acf_pop = np.zeros(lags)
for ii in range(N):
    for jj in range(N):
        temp_corr = autocorr1(driven_r_chaos[ii,jj,:])[:lags] #- np.mean(r_xyt[ii,jj,:])**2
        acf_pop = acf_pop + temp_corr/N**2
        
plt.figure()
plt.plot(acf_pop)
plt.xlabel('lags', fontsize=20); plt.ylabel(r'C($\tau$)', fontsize=20)

# %% compare traces
# plt.figure()
# plt.subplot(211); plt.plot(I_xyt[5,5,:600], 'k',label='stime'); plt.xticks([]); plt.yticks([]); plt.ylabel('stimuli', fontsize=20)
# plt.subplot(212)
# plt.plot(re_sym[:600], label='symm')
# plt.plot(re_pd[:600], label='PD')
# plt.plot(re_nd[:600], label='ND')
# plt.legend()

# %% compare signal
# plt.figure()
# plt.bar(['symmetric','PD', 'ND'], \
#         [np.mean(signal_sym), np.mean(signal_pd), np.mean(signal_nd)], \
#         yerr =  [np.std(signal_sym), np.std(signal_pd), np.std(signal_nd)])
# plt.ylabel('response', fontsize=20); plt.xticks(rotation=45)
