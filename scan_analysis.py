# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:14:03 2024

@author: kevin
"""

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

import os
import pickle

# %% get all simulated pkl files
# Directory where the pickle files are saved
input_dir = 'sims_mf'
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
axs = axs.flatten()
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
        
        ### measuring beta
        sig_ratio[ii] = sig_
        tau_ratio[jj] = tau_
        beta_scan[ii,jj] = np.mean(beta_t[:,:,50:])
        
        ### plotting
        ax = axs[fi]
        ax.loglog(ff, pp)
        ax.set_title(f'sigma={sig_}, tau={tau_}')
        fi += 1
        
        ### for visualization
        if fi-1 < (ni - 1) * nj:
            ax.set_xticks([])
ax.set_xlabel('Hz', fontsize=20)
ax.set_ylabel('power', fontsize=20)
            
# %%
fig, ax = plt.subplots()
cax = ax.imshow(beta_scan)

# Set the ticks to correspond to the vectors
ax.set_xticks(np.arange(beta_scan.shape[0]))
ax.set_xticklabels(tau_ratio)
ax.set_yticks(np.arange(beta_scan.shape[1]))
ax.set_yticklabels(sig_ratio)
ax.set_xlabel('tau ratio', fontsize=20)
ax.set_ylabel('sigma ratio', fontsize=20)
ax.set_title('beta', fontsize=20)
ax.invert_yaxis()
fig.colorbar(cax)
