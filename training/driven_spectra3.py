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
### test different temporal drive and spatiotemporal patterns
### here we are using a sine-wave patter drifting in time
### the hypothesis is that the moment matching should be related to the intrinsic length-scale and time scale

# %% init 2D chaotic network
N = 33
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

def measure_SNR(power_neural, power_sig, sig_threshold):
    power_neural, power_sig = power_neural[1:], power_sig[1:]
    signal_band = np.where(power_sig > sig_threshold)[0]
    noise_band = np.where(power_sig < sig_threshold)[0]
    P_signal = np.sum(power_neural[signal_band])
    P_noise = np.sum(power_neural[noise_band])
    return 10 * np.log10(P_signal / P_noise)

def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='same')
    return result[result.size//2:] / len(x)

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

def measure_SNR_group(r_xyt, lags=int(lt//2), post=int(tau_i/dt*5)):
    acf_pop = np.zeros(lags)
    for ii in range(N):
        for jj in range(N):
            temp_corr = autocorr1(r_xyt[ii,jj,:])[:lags] #- np.mean(r_xyt[ii,jj,:])**2
            acf_pop = acf_pop + temp_corr/N**2
    
    autocorrs_nois = acf_pop[0] - np.mean((acf_pop[post:])**2)**0.5
    autocorrs_sigs = np.mean((acf_pop[post:])**2)**0.5
    return autocorrs_sigs, autocorrs_nois

# %% setup spatiotemporal drive
space_f = 6
time_f = .2
x = np.linspace(0, space_f*2 * np.pi, N)  # Create a range for sine wave input
Iamp = 2.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale *1
I_xyt = np.zeros((N, N, lt)) # Initialize the tensor
# Populate the tensor
for t in range(lt):
    temp = np.array([np.sin(x + phase + t *time_f* (2 * np.pi / N)) 
                                for phase in np.linspace(0, 2 * np.pi, N)]).T
    I_xyt[:, :, t] = temp/np.linalg.norm(temp)
    
## %% visualization
plt.figure()
plt.imshow(I_xyt[:,:,0]); plt.colorbar()

# %% making the randon 2D spatial pattern
def make_2D_stim(time_f, space_f,N=N):
    x = np.linspace(0, space_f*2 * np.pi, N)  # Create a range for sine wave input
    I_xyt = np.zeros((N, N, lt)) # Initialize the tensor
    # Populate the tensor
    for t in range(lt):
        temp = np.array([np.sin(x + phase*0 + t *time_f* (2 * np.pi / N)) 
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
driven_r_chaos = make_chaotic(I_xyt*5)

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
# plt.ylim(1e-2, 1e4)
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
# plt.ylim(1e-2, 1e4)

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

# %%
###############################################################################
###################
###################
# check motion direction
# check ACF fucntion
###################
####################

# %% SNR measurement
# %% compare spatial input
# N=30 ################# check robustness with size!!
space_f = 4
time_f = 0.2 #.1 #0.05-.5
I_xyt = make_2D_stim(time_f, space_f)
sigs = np.arange(2,8,1) #np.array([3,6,9,12])
# sigs = np.array([0.05,0.1,0.15,0.2])
reps = 1
sig_threshold = 10 #0.001#10
snrs = np.zeros((reps, len(sigs)))
snrs_2d, snrs_shuffle = snrs*1, snrs*1
tau_i, sig_i = 15*0.001, 0.2
plt.figure()
spon_data = make_chaotic(I_xyt*0)
test_spon,ff_spon = group_spectrum(spon_data[:,:,50:])
plt.loglog(ff_spon[1:],test_spon[1:], label='spontaneous') ### use loglog or plot
power_sig, ff_sig = group_spectrum(I_xyt[:,:,50:])
plt.plot(ff_sig[1:], power_sig[1:], '-o',label='stim')
plt.legend()

# snrs = np.zeros((reps, len(sigs), N*N))
# snrs_2d, snrs_shuffle = snrs*1, snrs*1

# %%
for rr in range(reps):
    print('repeat: ', rr)
    for ii in range(len(sigs)):
        print(ii)
        
        ### compute driven spectrum
        temp_stim = make_2D_stim(time_f, sigs[ii],N)
        driven_data = make_chaotic(temp_stim,N)
        aa,bb = measure_SNR_group(driven_data)
        snrs_2d[rr,ii] = aa
        
        # power_neural, ff_neural = group_spectrum(driven_data[:,:,50:])
        # plt.figure(1)
        # plt.loglog(ff_neural[1:], power_neural[1:], label=rf"$\sigma = {sigs[ii]}$")
        # snrs_2d[rr, ii] = measure_SNR(power_neural, power_sig, sig_threshold)
        
        ### compute control, shuffle in space
        temp_stim = temp_stim.reshape(N**2, lt) ##### for shuffle control!!!
        temp_stim = temp_stim[np.random.permutation(N**2), :].reshape(N,N,lt) 
        driven_data_ctrl = make_chaotic(temp_stim,N)
        aa,bb = measure_SNR_group(driven_data_ctrl)
        snrs_shuffle[rr,ii] = aa
        # power_neural_ctrl, ff_neural_ctrl = group_spectrum(driven_data_ctrl[:,:,50:])
        # plt.figure(2)
        # plt.loglog(ff_neural_ctrl[1:], power_neural_ctrl[1:], label=rf"$\sigma = {sigs[ii]}$")
        # snrs_shuffle[rr, ii] = measure_SNR(power_neural_ctrl, power_sig, sig_threshold)
    
    # plt.figure(1)
    # plt.title('driven spectrum'); plt.legend(fontsize=10); #plt.ylim(0, 0.3)
    # plt.figure(2)
    # plt.title('shuffled in space'); plt.legend(fontsize=10); #plt.ylim(0, 0.3)#(1e-2, 1e5)

# plt.xlim([0, 20])

# %%
from scipy.stats import ttest_ind
plt.figure()
# plt.plot(sigs,  snrs.T, 'k-o')
# plt.plot(sigs,  snrs_2d.T, 'k-o')
# plt.plot(sigs,  snrs_shuffle.T, '-o')
plt.errorbar(sigs, np.nanmean(snrs_2d,0), yerr=np.nanstd(snrs_2d,0), fmt='-o', capsize=5, label='2D')
plt.errorbar(sigs, np.nanmean(snrs_shuffle,0), yerr=np.nanstd(snrs_shuffle,0), fmt='-o', capsize=5, label='shuffled')
plt.xlabel('spatial frequency', fontsize=20); plt.ylabel('signal (dB)', fontsize=20)
# plt.xscale('log')
plt.legend(fontsize=15)

# %% Intensity scan
###############################################################################
# %% setup
space_f = 3
time_f = 0.4 #.1 #0.05-.5
sigs = np.array([0.1, 0.25, 0.5, 1, 1.5, 2])
I_xyt = make_2D_stim(time_f, space_f)
reps = 1
sig_threshold = 10 #0.001#10
# snrs = np.zeros((reps, len(sigs)))
# snrs_2d, snrs_shuffle = snrs*1, snrs*1
tau_i, sig_i = 15*0.001, 0.2
plt.figure()
spon_data = make_chaotic(I_xyt*0)
test_spon,ff_spon = group_spectrum(spon_data[:,:,50:])
plt.loglog(ff_spon[1:],test_spon[1:], label='spontaneous') ### use loglog or plot
power_sig, ff_sig = group_spectrum(I_xyt[:,:,50:])
plt.plot(ff_sig[1:], power_sig[1:], '-o',label='stim')
plt.legend()

snrs = np.zeros((reps, len(sigs), N*N))
snrs_2d, snrs_shuffle = snrs*1, snrs*1

# %% scanning
for rr in range(reps):
    print('repeat: ', rr)
    for ii in range(len(sigs)):
        print(ii)
        
        ### compute driven spectrum
        temp_stim = make_2D_stim(time_f, space_f, N)* sigs[ii]
        driven_data = make_chaotic(temp_stim,N)
        aa,bb = measure_SNR_new(driven_data)
        snrs_2d[rr,ii] = aa
        # power_neural, ff_neural = group_spectrum(driven_data[:,:,50:])
        # plt.figure(1)
        # plt.loglog(ff_neural[1:], power_neural[1:], label=rf"$\sigma = {sigs[ii]}$")
        # snrs_2d[rr, ii] = measure_SNR(power_neural, power_sig, sig_threshold)
        
        ### compute control, shuffle in space
        temp_stim = temp_stim.reshape(N**2, lt) ##### for shuffle control!!!
        temp_stim = temp_stim[np.random.permutation(N**2), :].reshape(N,N,lt) 
        driven_data_ctrl = make_chaotic(temp_stim,N)
        aa,bb = measure_SNR_new(driven_data_ctrl)
        snrs_shuffle[rr,ii] = aa
        # power_neural_ctrl, ff_neural_ctrl = group_spectrum(driven_data_ctrl[:,:,50:])
        # plt.figure(2)
        # plt.loglog(ff_neural_ctrl[1:], power_neural_ctrl[1:], label=rf"$\sigma = {sigs[ii]}$")
        # snrs_shuffle[rr, ii] = measure_SNR(power_neural_ctrl, power_sig, sig_threshold)
    
    plt.figure(1)
    plt.title('driven spectrum'); plt.legend(fontsize=10); #plt.ylim(0, 0.3)
    plt.figure(2)
    plt.title('shuffled in space'); plt.legend(fontsize=10); #plt.ylim(0, 0.3)#(1e-2, 1e5)

# %% plotting for all
plt.figure()
plt.errorbar(sigs, np.mean(snrs_2d,2).squeeze(), yerr=np.std(snrs_2d,2).squeeze(), fmt='-o', capsize=5, label='2D')
plt.errorbar(sigs, np.mean(snrs_shuffle,2).squeeze(), yerr=np.std(snrs_shuffle,2).squeeze(), fmt='-o', capsize=5, label='shuffled')
plt.xlabel(r'intensity', fontsize=20); plt.ylabel('signal (dB)', fontsize=20)
plt.legend(); # plt.xscale('log')

# %%
#### HOW ABOUT INFO in SPACE!!!
###############################################################################
# %%
space_f = 2
time_f = .2 #0.05-.5
I_xyt = make_2D_stim(time_f, space_f)
# sigs = np.array([3,6,9,12])  ### spatial
sigs = np.array([0.05,0.1,0.15,0.2])   # temporal
reps = 1
sig_threshold = 0.0002#100
snrs = np.zeros((reps, len(sigs)))
snrs_2d, snrs_shuffle = snrs*1, snrs*1
tau_i, sig_i = 15*0.001, 0.2
plt.figure()
spon_data = make_chaotic(I_xyt*0)
test_spon,ff_spon = group_spectrum_space(spon_data[:,:,50:])
plt.plot(ff_spon[1:],test_spon[1:], label='spontaneous') ### use loglog or plot
test_sig,ff_sig = group_spectrum_space(I_xyt[:,:,50:])
plt.plot(ff_sig[1:], test_sig[1:], label='stim')

snrs = np.zeros((reps, len(sigs), N*N))
snrs_2d, snrs_shuffle = snrs*1, snrs*1

# %%
for rr in range(reps):
    print('repeat: ', rr)
    for ii in range(len(sigs)):
        print(ii)
        ### compute driven spectrum
        temp_stim = make_2D_stim(sigs[ii], space_f) ### scanning time
        # temp_stim = make_2D_stim(time_f, sigs[ii])   ### scanning space
        ####### CONTROL #######
        # temp_stim = temp_stim.reshape(N**2, lt) ##### for shuffle control!!!
        # temp_stim = temp_stim[np.random.permutation(N**2), :].reshape(N,N,lt) #### for shuffle control!!!
        # shuffle in time??
        # temp_stim = temp_stim.reshape(N**2, lt) ##### for shuffle control!!!
        # temp_stim = temp_stim[:, np.random.permutation(lt)].reshape(N,N,lt)
        #######################
        driven_data = make_chaotic(temp_stim)
        aa,bb = measure_SNR_new(driven_data)
        snrs_2d[rr,ii] = aa
        # test,ff = group_spectrum_space(driven_data[:,:,50:]) #temp_stim
        # plt.plot(ff[1:],test[1:], label=rf"$\sigma = {sigs[ii]}$")

        
        ### compute controled spectrum
        temp_stim = temp_stim.reshape(N**2, lt) ##### for shuffle control!!!
        temp_stim = temp_stim[np.random.permutation(N**2), :].reshape(N,N,lt) #### for shuffle control!!!
        driven_data_ctrl = make_chaotic(temp_stim,N)
        aa,bb = measure_SNR_new(driven_data_ctrl)
        snrs_shuffle[rr,ii] = aa
        
        ##hacky test
        # signal_power, noise_power = test[1:], test_sig[1:]
        # signal_band = np.where(test_sig[1:]>sig_threshold)[0]
        # noise_band = np.where(test_sig[1:]<sig_threshold)[0]
        # P_signal = np.sum(signal_power[signal_band])
        # P_noise = np.sum(signal_power[noise_band])
        # # Compute SNR in dB
        # snrs[rr, ii] = 10 * np.log10(P_signal / P_noise)
    
    plt.legend(fontsize=10)
    # plt.ylim(1e-2, 1e5)
# plt.xlim([0, 20])
plt.yscale('log')

# %% plotting
plt.figure()
# plt.plot(sigs,  snrs.T, 'k-o')
# plt.plot(sigs,  snrs_2d.T, 'k-o')
# plt.plot(sigs,  snrs_shuffle.T, '-o')
plt.errorbar(sigs, np.mean(snrs_2d,0), yerr=np.std(snrs_2d,0), fmt='-o', capsize=5, label='2D')
plt.errorbar(sigs, np.mean(snrs_shuffle,0), yerr=np.std(snrs_shuffle,0), fmt='-o', capsize=5, label='shuffled')
plt.xlabel(r'temporal frequency', fontsize=20); plt.ylabel('signal (dB)', fontsize=20)
# plt.xscale('log')
plt.legend()

# %%
# space_f = 7
# time_f = .06 #0.05-.5
# I_xyt = make_2D_stim(time_f, space_f)
# # Step 1: Define a 2D array
# N = 30
# data = I_xyt[:,:,0]
# # x = np.linspace(0, 2 * np.pi, N)
# # y = np.linspace(0, 2 * np.pi, N)
# # X, Y = np.meshgrid(x, y)

# # Example: 2D sine wave with specific frequencies
# # data = np.sin(10 * X) + np.cos(15 * Y)

# # Step 2: Compute the 2D FFT
# fft_result = np.fft.fft2(data)
# fft_shifted = np.fft.fftshift(fft_result)
# magnitude_spectrum = np.abs(fft_shifted)

# # Step 3: Find the peak frequency
# # Exclude the DC component at the center
# center = (N // 2, N // 2)
# magnitude_spectrum[center] = 0  # Set the center value to 0 to ignore DC

# # Find the indices of the peak
# peak_indices = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
# peak_freq_x, peak_freq_y = peak_indices

# # Convert to spatial frequencies
# freq_x = abs(peak_freq_x - N // 2)  # Offset from the center
# freq_y = abs(peak_freq_y - N // 2)

# # Step 4: Print results
# print(f"Dominating spatial frequency: freq_x = {freq_x}, freq_y = {freq_y}")

# # Optional: Visualize the magnitude spectrum
# plt.imshow(np.log1p(magnitude_spectrum), cmap='hot')
# plt.title("Magnitude Spectrum (Log-Scaled)")
# plt.colorbar()
# plt.scatter(peak_freq_y, peak_freq_x, color='blue', label='Dominating Frequency')
# plt.legend()
# plt.show()

