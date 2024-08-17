# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 01:01:19 2024

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

# %% analytic forms
def g_fft(n, sig, mu=0):
    temp = np.exp(-2*n**2*np.pi**2*sig**2 - 2*n*np.pi*mu*(-1)**0.5)
    return temp

def A_matrix(params, n):
    sig_e, sig_i, tau_e, tau_i, wee, wei, wie, wii = params
    A = np.array([[wee*g_fft(n, sig_e)/tau_e,  wei*g_fft(n, sig_i)/tau_e],
                  [wie*g_fft(n, sig_e)/tau_i,  wii*g_fft(n, sig_i)/tau_i]])
    return A
    
# %% scanning
nnn = 1
wee, wei, wie, wii = 1, -2, 0.99, -1.8
tau_e, sig_e = 0.005, 0.1

sig_ratios = np.arange(0.5,2,0.2)
tau_ratios = np.arange(0.5,5,0.5)

stabs = np.zeros((len(sig_ratios), len(tau_ratios)))

# for nnn in range(20):
for ii in range(len(sig_ratios)):
    for jj in range(len(tau_ratios)):
        sig_i = sig_e*sig_ratios[ii]
        tau_i = tau_e*tau_ratios[jj]
        params = sig_e, sig_i, tau_e, tau_i, wee, wei, wie, wii
        Atemp = A_matrix(params, nnn)
        ee,vv = np.linalg.eig(Atemp)
        
        ### given one n
        stabs[ii,jj] = np.max(ee)
        
        ### looping through n-s
        # stabs[ii,jj] += np.max(ee)
        
        
# %%
sig_ratios = np.round(sig_ratios, 1)
tau_ratios = np.round(tau_ratios, 1)
plt.figure()
plt.imshow(stabs)
plt.colorbar()
plt.yticks(ticks=np.arange(len(sig_ratios)), labels=sig_ratios)
plt.xticks(ticks=np.arange(len(tau_ratios)), labels=tau_ratios)
plt.xlabel('tau_i/tau_e', fontsize=20)
plt.ylabel('sig_i/sig_e', fontsize=20)
plt.gca().invert_yaxis()  # Flip the y-axis

# %% do it across frequencies
nl = 15
stabs_n = np.zeros((len(sig_ratios), len(tau_ratios), nl))  # sig x tau x n
for nnn in range(nl):
    for ii in range(len(sig_ratios)):
        for jj in range(len(tau_ratios)):
            sig_i = sig_e*sig_ratios[ii]
            tau_i = tau_e*tau_ratios[jj]
            params = sig_e, sig_i, tau_e, tau_i, wee, wei, wie, wii
            Atemp = A_matrix(params, nnn)
            ee,vv = np.linalg.eig(Atemp)
            
            ### given one n
            stabs_n[ii,jj,nnn] = np.max(ee)

# %%
unstab_n = stabs*np.nan

for ii in range(len(sig_ratios)):
    for jj in range(len(tau_ratios)):
        if np.max(stabs_n[ii,jj,:]) > 0:
            unstab_n[ii,jj] = np.argmax(stabs_n[ii,jj,:])

# %%
plt.figure()
plt.imshow(unstab_n)
plt.yticks(ticks=np.arange(len(sig_ratios)), labels=sig_ratios)
plt.xticks(ticks=np.arange(len(tau_ratios)), labels=tau_ratios)
plt.xlabel('tau_i/tau_e', fontsize=20)
plt.ylabel('sig_i/sig_e', fontsize=20)
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label('unstable spatial freq.', rotation=90, labelpad=15, fontsize=20)