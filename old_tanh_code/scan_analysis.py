# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:14:03 2024

@author: kevin
"""

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)

import os
import pickle

# %% get all simulated pkl files
# Directory where the pickle files are saved
input_dir = 'sims_unbalance'
# Get a list of all files in the directory
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pkl')])

# %% data measured
# data_to_save = {
#     're_xy': re_xy,
#     'ri_xy': ri_xy,
#     'beta_t': beta_t,
#     'power': pp,
#     'frequency': ff,
#     'sigma_i': sig_i,
#     'tau_i': tau_i,
#     'sigma_e': sig_e,
#     'tau_e': tau_e,
#     'e_signal': measure_mu_ex,
#     'i_signal': measure_mu_ix
# }

# %% loop through files
ni = 4
nj = 4
fi = 0

### initalize measurements
beta_scan = np.zeros((ni, nj))  ### recording average beta values
sig_ratio = np.zeros(ni)
tau_ratio = np.zeros(nj)

fig, axs = plt.subplots(ni, nj, figsize=(15, 15))
ax = axs#.flatten()
row_change = np.arange(ni-1,-1,-1)
for ii in range(ni):
    for jj in range(nj):
        
        ### loading files
        file_path = os.path.join(input_dir, files[fi])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        ### loading data
        ff = data['frequency']
        pp = data['power']
        sig_ = round(data['sigma_i']/data['sigma_e'], 1)
        tau_ = round(data['tau_i']/data['tau_e'], 1)
        beta_t = data['beta_t']
        temp_re = data['re_xy']
        
        ### measuring beta
        sig_ratio[ii] = sig_
        tau_ratio[jj] = tau_
        beta_scan[ii,jj] = np.median(beta_t[:,:,50:])
        
        ### plotting
        ax = axs[row_change[ii],jj]
        #######################################################################
        ### for fourier in time
        #######################################################################
        ax.loglog(ff, pp)
        ax.set_xlim([0.1, 500])
        ax.set_ylim([0.01, 10**5])
        #######################################################################
        ### for fourier in space
        #######################################################################
        # data4fft = temp_re[:,:,50:]*1
        # _,_,lt = data4fft.shape
        # data_fft = np.fft.fftn(data4fft)
        # data_fft_shifted = np.fft.fftshift(data_fft)
        # magnitude_spectrum = np.abs(data_fft_shifted)
        # ax.imshow(np.log(magnitude_spectrum[:, :, 1*lt//2]), cmap='gray')
        # ax.set_xticks([])  # Remove x ticks
        # ax.set_yticks([]) 
        #######################################################################
        ### for activity histogram
        #######################################################################
        ### in time
        # temp = np.mean(np.mean(temp_re,0),0)   ### in time
        # ax.set_xlim([0, 1.])
        # temp = np.mean(temp_re,2).reshape(-1)  ### in space
        # ax.hist(temp,50)
        
        ax.set_title(rf'$\sigma$={sig_}, $\tau$={tau_}',fontsize=20)
        fi += 1
        
        if ii==0 and jj==3:
            ax.set_xlabel('Hz', fontsize=20)
            ax.set_ylabel('power', fontsize=20)
        ### for visualization
        # if fi-1 < (ni - 1) * nj:
            # ax.set_xticks([])
# ax.set_xlabel('Hz', fontsize=15)
# ax.set_ylabel('power', fontsize=15)
# ax.set_xlabel('firing', fontsize=20)
# ax.set_ylabel('count', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.4)
            
# %%
fig, ax = plt.subplots()
# cax = ax.imshow(beta_scan)
cax = ax.imshow(beta_scan, vmin=0, vmax=.5)  # Set color range here

# Set the ticks to correspond to the vectors
ax.invert_yaxis()
ax.set_xticks(np.arange(beta_scan.shape[0]))
ax.set_xticklabels(tau_ratio)
ax.set_yticks(np.arange(beta_scan.shape[1]))
ax.set_yticklabels(sig_ratio)
ax.set_xlabel(r'$\tau_i / \tau_e$', fontsize=20)
ax.set_ylabel(r'$\sigma_i / \sigma_e$', fontsize=20)
ax.set_title(r'<$\beta$> (balanced)', fontsize=20)
fig.colorbar(cax)
