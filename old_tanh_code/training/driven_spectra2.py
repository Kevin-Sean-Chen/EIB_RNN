# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:10:39 2024

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
### test different temporal drive and spatial patterns
### the hypothesis is that the moment matching should be related to the intrinsic length-scale

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

rescale = 3. ##(N*sig_e*np.pi*1)**0.5 #1
Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale
mu_i = .8*rescale

### setting up space and time
kernel_size = 23 #37  # pick this for numerical convolution

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

# %% setup spatiotemporal drive
### temporal signal
I_xy = np.sin(time*20*2) + 1*np.cos(time*30*2) ### 10-*100
I_xy = np.sin(time*60)
# I_xy = np.sin(time*20*7) + np.cos(time*30*8) ### out of 2-5 does not work!? ###... prob not!
plt.figure()
plt.plot(I_xy[:])
I_xy = I_xy/np.max(I_xy)  # 2D input video
Iamp = 2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale *1

# %% making the randon 2D spatial pattern
### spatial pattern
sigma_xy = 0.05 #0.4, 0.2, 0.1, 0.05
temp_space1 = np.random.randn(N,N)
temp_space2 = np.random.randn(N,N)
temp_k = g_kernel(sigma_xy, N)
pattern1 = spatial_convolution(temp_space1, temp_k) ########### TEST THE SPTAIAL FEATURE!!! ##############
pattern2 = temp_space2 #spatial_convolution(temp_space2, temp_k)

pattern1 = pattern1/np.linalg.norm(pattern1)
I_xyt = np.zeros((N,N, lt))
for tt in range(lt):
    I_xyt[:,:,tt] = I_xy[tt]*pattern1 # + z[tt]*pattern2

def make_2D_stim(sigma_xy):
    temp_space1 = np.random.randn(N,N)
    temp_k = g_kernel(sigma_xy, N)
    pattern1 = spatial_convolution(temp_space1, temp_k) ########### TEST THE SPTAIAL FEATURE!!! ##############
    pattern1 = pattern1/np.linalg.norm(pattern1)
    I_xyt = np.zeros((N,N, lt))
    for tt in range(lt):
        I_xyt[:,:,tt] = I_xy[tt]*pattern1 # + z[tt]*pattern2
    return I_xyt

# %% check stim correlation
# test = I_xyt.reshape(N**2, lt)
# cross_correlation_matrix = np.corrcoef(test)
# plt.figure(figsize=(8, 8))
# plt.imshow(cross_correlation_matrix)
# plt.colorbar()
    
# %%
def make_chaotic(I_xy):
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
        
        ### modifying for 2D rate EI-RNN
        ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
        gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
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
driven_r_chaos = make_chaotic(I_xyt)

tau_i, sig_i = 5*0.001, 0.14
spontaneous_r_ctr = make_chaotic(I_xyt*0)
driven_r_ctr = make_chaotic(I_xyt)

# %%
###############################################################################
### spectral analysis
###############################################################################
# %%
plt.figure()
plt.subplot(121)
test,ff = group_spectrum(spontaneous_r_chaos[:,:,50:])
plt.loglog(ff[2:],test[2:], label='spontaneous')
test,ff = group_spectrum(driven_r_chaos[:,:,50:])
plt.loglog(ff[2:],test[2:], label='driven')
test,ff = group_spectrum(I_xyt[:,:,50:])
plt.loglog(ff[2:],test[2:], label='stim')
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power', fontsize=20)
plt.title('chaotic regime', fontsize=20)
plt.legend(fontsize=20)
plt.ylim(1e-2, 1e4)
plt.subplot(122)
test,ff = group_spectrum(spontaneous_r_ctr[:,:,50:])
plt.loglog(ff[2:],test[2:], label='spontaneous')
test,ff = group_spectrum(driven_r_ctr[:,:,50:])
plt.loglog(ff[2:],test[2:], label='driven')
test,ff = group_spectrum(I_xyt[:,:,50:])
plt.loglog(ff[2:],test[2:], label='stim')
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power', fontsize=20)
plt.title('lattice regime', fontsize=20)
plt.legend(fontsize=20)
plt.ylim(1e-2, 1e4)

# %% spatial spectrum
def plot_spatial_spec(data):
    data_fft = np.fft.fftn(data)
    _,_,lt = data.shape
    data_fft_shifted = np.fft.fftshift(data_fft)
    magnitude_spectrum = np.abs(data_fft_shifted)
    plt.imshow(np.log1p(magnitude_spectrum[:,:,lt//2]), cmap='gray')
    return

spon, driven = spontaneous_r_chaos, driven_r_chaos
# spon, driven = spontaneous_r_ctr, driven_r_ctr
plt.figure()
plt.subplot(121)
plot_spatial_spec(spon)
plt.subplot(122)
plot_spatial_spec(driven)

# %% compare spatial input
sigs = np.array([0.05, 0.09, 0.1, 0.11, 0.2, 0.4, 0.8])
reps = 5
snrs = np.zeros((reps, len(sigs)))
tau_i, sig_i = 15*0.001, 0.2
plt.figure()
spon_data = make_chaotic(I_xyt*0)
test_spon,ff_spon = group_spectrum(spon_data[:,:,50:])
plt.loglog(ff_spon[1:],test_spon[1:], label='spontaneous') ### use loglog or plot
test_sig,ff_sig = group_spectrum(I_xyt[:,:,50:])
plt.plot(ff_sig[1:], test_sig[1:], label='stim')

for rr in range(reps):
    print('repeat: ', rr)
    for ii in range(len(sigs)):
        print(ii)
        ### compute driven spectrum
        temp_stim = make_2D_stim(sigs[ii])
        temp_stim = temp_stim.reshape(N**2, lt) ##### for shuffle control!!!
        temp_stim = temp_stim[np.random.permutation(N**2), :].reshape(N,N,lt) #### for shuffle control!!!
        driven_data = make_chaotic(temp_stim)
        test,ff = group_spectrum(driven_data[:,:,50:]) #temp_stim
        plt.plot(ff[1:],test[1:], label=rf"$\sigma = {sigs[ii]}$")
        
        
        ### measure SNR
        # signal_power, noise_power = test[1:], test_sig[1:]
        # noise_power = signal_power - noise_power  ##### probably wrong !!!!!!!!!!!!!!!!!!!!!!!!!!! visit RAJAN!!!
        # snrs[rr, ii] = 10 * np.log10(signal_power.sum() / noise_power.sum())
        
        ##hacky test
        signal_power, noise_power = test[1:], test_sig[1:]
        signal_band = np.where(test_sig[1:]>1)[0]
        noise_band = np.where(test_sig[1:]<1)[0]
        P_signal = np.sum(signal_power[signal_band])
        P_noise = np.sum(signal_power[noise_band])
        # Compute SNR in dB
        snrs[rr, ii] = 10 * np.log10(P_signal / P_noise)
        # pos = np.where(test_sig[1:]>1)[0]
        # snrs[rr, ii] = 10 * np.log10(signal_power[pos].sum())# / signal_power[5:].sum()) # given that we know the signal
    plt.legend(fontsize=10)
    plt.ylim(1e-2, 1e5)
# plt.xlim([0, 20])

# %%
from scipy.stats import ttest_ind
plt.figure()
# plt.plot(sigs,  snrs.T, 'k-o')
# plt.plot(sigs,  snrs_2d.T, 'k-o')
# plt.plot(sigs,  snrs_shuffle.T, '-o')
# plt.errorbar(sigs, np.mean(snrs_2d,0), yerr=np.std(snrs_2d,0), fmt='-o', capsize=5, label='2D')
# plt.errorbar(sigs, np.mean(snrs_shuffle,0), yerr=np.std(snrs_shuffle,0), fmt='-o', capsize=5, label='shuffled')
plt.xlabel(r'$\sigma_{sim}$', fontsize=20); plt.ylabel('signal (dB)', fontsize=20)
plt.xscale('log')
plt.legend()