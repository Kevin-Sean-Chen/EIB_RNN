# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:03:28 2024

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

import os
import pickle

# %% fixing initial condition to study chaos
np.random.seed(13)
N = 70 #200
init_2d_e = np.random.randn(N,N)
init_2d_i = np.random.randn(N,N)

# %% parameter setup
# N = 70  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
# tau_i, sig_i = 12*0.001, 0.2   ### imporat parameters!!
### moving dots 5ms, 0.2
### little coherence 5ms, 0.1
### chaotic waves 10ms, 0.1
### drifting changing blobs 10ms, 0.2

rescale = 1. ### seems to matter for finite size(?)

# %% time and space setup
### setting up space and time
dt = 0.001  # 1ms time steps
T = 2.0  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
space_vec = np.linspace(0,1,N)
kernel_size = 23  # pick this for numerical convolution
offset = 50  ### remove initial part to get rid of transient effects

# %% scanning choices
sig_ie = np.array([0.5, 1.0, 1.5, 2.0])  # ratio between i and e length scales
tau_ie = np.array([1, 2, 3, 4])  # ratio between i and e time scale

# %% functions
def phi(x):
    """
    rectified quardratic nonlinearity
    """
    # nl = np.where(x > 0, x*1, 0)
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
            

# %% 2D-conv network function
def chaotic_2d_net(sig_i, tau_i):
    
    #################################################
    ######## Mean field condition
    # Wee = 80*2  # recurrent weights
    # Wei = -160*2
    # Wie = 80*2
    # Wii = -150*2
    # mu_e = 0.48*10  # offset
    # mu_i = 0.32*10
    #################################################
    #################################################
    ######## Balancing condition 
    #################################################
    ### old param not balancing (ratio all the same)
    # Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
    # Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
    # Wie = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
    # Wii = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
    # mu_e = .7*rescale #*(N*sig_i*np.pi*1)**0.5 *rescale  #1e-8#.001*1  # offset
    # mu_i = .7*rescale #*(N*sig_i*np.pi*1)**0.5 *rescale  #1e-8#.001*1
    
    ### rescaling parameters for balance condition
    Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
    Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
    Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
    Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
    mu_e = 1.*rescale
    mu_i = .8*rescale
    #################################################
    
    ### initialization
    ### random initial conditions
    re_xy = np.zeros((N,N, lt))
    ri_xy = re_xy*1
    he_xy = re_xy*1
    hi_xy = re_xy*1
    re_xy[:,:,0] = init_2d_e*1
    ri_xy[:,:,0] = init_2d_i*1
    he_xy = re_xy*1
    hi_xy = ri_xy*1
    
    ### measure the field
    measure_mu = np.zeros((N,N,lt))
    measure_mu_ex = np.zeros((N,N,lt))
    measure_mu_ix = np.zeros((N,N,lt))
 
    ### neural dynamics
    for tt in range(lt-1):
        ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
        gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
        he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
        hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
        re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
        
        ### mean measurements
        measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
        measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
        measure_mu_ix[:,:,tt+1] = Wei*gi_conv_ri
        
        ### make E-I measurements
        # measure_e[tt+1] = (Wee*ge_conv_re + mu_e)[20,20]
        # measure_i[tt+1] = (Wei*gi_conv_ri)[20,20]
        
    return re_xy, ri_xy, measure_mu, measure_mu_ex, measure_mu_ix
    
# %% scanning

# Create a directory to save the files if it doesn't exist
output_dir = 'sims_balanced2'  ## the sims data folder
os.makedirs(output_dir, exist_ok=True)

for ii in range(len(sig_ie)):
    for jj in range(len(tau_ie)):
        print(ii)
        print(jj)
        
        ### get param changes
        sig_i = sig_ie[ii]*sig_e
        tau_i = tau_ie[jj]*tau_e
        
        ### simulate the 2D dynamics
        re_xy, ri_xy, measure_mu, measure_mu_ex, measure_mu_ix = chaotic_2d_net(sig_i, tau_i)
        
        ### measure beta in space and time
        beta_t = measure_mu / measure_mu_ex 
        
        ### spectral analysis
        pp,ff = group_spectrum(re_xy[:,:,offset:])
        
        ### record data and information
        # Combine the matrix and the extra information into a dictionary
        data_to_save = {
            're_xy': re_xy,
            'ri_xy': ri_xy,
            'beta_t': beta_t,
            'power': pp,
            'frequency': ff,
            'sigma_i': sig_i,
            'tau_i': tau_i,
            'sigma_e': sig_e,
            'tau_e': tau_e,
            'e_signal': measure_mu_ex,
            'i_signal': measure_mu_ix
        }
    
        # Define the filename with the iteration number
        filename = f"chaos_2d_sig{ii+1}_tau{jj+1}.pkl"
        
        # Define the full file path
        file_path = os.path.join(output_dir, filename)
        
        # Save the dictionary to the file as a pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

