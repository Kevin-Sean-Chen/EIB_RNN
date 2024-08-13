# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:15:24 2024

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
input_dir = 'sims_init_'
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

# %% iteration across initial conditions
nfiles = len(files)

fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()
temp = []

for fi in range(nfiles):
    
    ### loading files
    file_path = os.path.join(input_dir, files[fi])
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    ### loading data
    ff = data['frequency']
    pp = data['power']
    beta_t = data['beta_t']
    re_space = data['re_xy']
    
    ### plotting
    ax1.loglog(ff, pp)
    ax2.plot(np.mean(np.mean(re_space,0),0))
    temp.append(re_space)
    
# %%
temp_rep = np.array(temp)
for rr in range(20):
    plt.figure()
    plt.imshow(temp_rep[rr,:,:,-1])

# plt.imshow(np.mean(temp_rep[15,:,:,:],2))

# %%
data4fft = temp_rep[9,:,:,50:]*1
_,_,lt = data4fft.shape

plt.figure()
# Compute the 3D Fourier Transform
data_fft = np.fft.fftn(data4fft)

# Shift the zero frequency component to the center
data_fft_shifted = np.fft.fftshift(data_fft)

# Compute the magnitude spectrum
magnitude_spectrum = np.abs(data_fft_shifted)

# Plot the magnitude spectrum for a particular time slice
plt.imshow(np.log(magnitude_spectrum[:, :, lt//2]), cmap='gray')
plt.title('Magnitude Spectrum (log scale) at t = t/2')
plt.colorbar()
plt.show()
