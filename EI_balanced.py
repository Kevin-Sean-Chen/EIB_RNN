# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:10:17 2024

@author: kevin
"""

from scipy.signal import convolve2d
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# np.random.seed(13)

# %% EI-RNN setup
### network and weights
Ne = 1000  # E neurons
Ni = 1000  # I neurons

def dilute_net(size, pc):
    cij = np.zeros((size, size))
    randM = np.random.rand(size,size)
    pos = np.where(randM < pc)
    cij[pos] = 1
    ### normalization for <cij> = K/N
    # K = pc*size
    # cij = cij / K**0.5 *2
    ###
    return cij
def phi(x):
    # nl = np.where(x > 0, x, 0)  # ReLU nonlinearity
    nl = np.where(x > 0, np.tanh(x), 0)  # nonlinearity for rate equation
    return nl

finite_scale = 2. ### this is hand-tuned to correct for finite size
### N = total number of neurons.
### K = mean number of connections per neuron
pc = 0.5 # connection probability (fraction that are 0)
K = pc*(Ne)  # degree of connectivity
rescale_c = 1/(K**0.5)*finite_scale  # am not sure if this is correct... but might be for finite size network
c_ee, c_ei, c_ie, c_ii = dilute_net(Ne, pc), dilute_net(Ne, pc),\
                         dilute_net(Ne, pc), dilute_net(Ne, pc)

### weights and rescaling
Jee = 1.0*rescale_c  # recurrent weights
Jei = -2.0*rescale_c
Jie = 1.0*rescale_c
Jii = -1.8*rescale_c # -1.8
Je0 = 1.*1 #rescale_c   # does NOT scale with K if we are using a baseline!
Ji0 = .8*1 #rescale_c #0.8

### time scales
tau = 0.009  # in seconds
dt = 0.001  # time step
T = 2.
time = np.arange(0, T, dt)
lt = len(time)

# %% dynamics
vet = np.zeros((Ne, lt))
vet[:,0] = np.random.randn(Ne)
vit = np.zeros((Ni, lt))
vit[:,1] = np.random.randn(Ni)
ret = vet*1
rit = vit*1
measure_e = np.zeros(lt)
measure_i = np.zeros(lt)

### for stimulus scanning
stim = np.zeros(lt) + Je0
# stim[lt//2:] = 2  # lifting input

# amps = np.array([1,1.5,2, 2.5,3])
# ei_responses = np.zeros(len(amps))
# for ss in range(len(amps)):
#     stim = np.zeros(lt) + Je0
#     stim[lt//2:] = amps[ss]

for tt in range(lt-1):
    ### EI-RNN dynamics
    vet[:,tt+1] = vet[:,tt] + dt/tau*( -vet[:,tt] + Jee*c_ee@phi(vet[:,tt]) + Jei*c_ei@phi(vit[:,tt]) + Je0*0 + stim[tt])
    vit[:,tt+1] = vit[:,tt] + dt/tau*( -vit[:,tt] + Jie*c_ie@phi(vet[:,tt]) + Jii*c_ii@phi(vit[:,tt]) + Ji0)
    ret[:,tt+1] = phi(vet[:,tt+1])*1
    rit[:,tt+1] = phi(vit[:,tt+1])*1
    
    ### measuring the input current
    measure_e[tt+1] = (Jee*c_ee@phi(vet[:,tt+1]) + Je0)[30]
    measure_i[tt+1] = (Jei*c_ei@phi(vit[:,tt+1]))[30]
        
# %% for stimulus tuning
# plt.figure()
# plt.plot(amps-Je0, ei_responses,'-o')
# plt.xlabel('external drive', fontsize=20)
# plt.ylabel('network rate', fontsize=20)

# %% plotting
offset = 50  # to remove effects from initial condition
select_n = 20
plt.figure()
plt.plot(time[offset:], ret[:select_n, offset:].T/dt)
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('rate (Hz)', fontsize=20)
plt.title('firing rate', fontsize=20)

# %%
per_time = 1
plt.figure()
plt.plot(time[offset:], measure_e[offset:]/(per_time), label='excitation')  # input across a window
plt.plot(time[offset:], measure_i[offset:]/(per_time), label='inhibition')
plt.plot(time[offset:], (measure_e + measure_i)[offset:]/(per_time), '--', label='total')
plt.legend(fontsize=10)
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('input current', fontsize=20)
plt.title('balancing input', fontsize=20)
# plt.xlim([0,0.3])

# %% beta measurement
plt.figure()
plt.plot(time, np.abs(measure_e+measure_i)/measure_e)
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('beta', fontsize=20)

# %% plotting population
plt.figure(figsize=(7,15))
plt.imshow(ret[:,offset:], cmap='hot')   ######################### understand the value here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!???
plt.xlabel('time', fontsize=20)
plt.ylabel('cells', fontsize=20)
# plt.colorbar()

plt.figure()
# plt.hist(ret[:,offset:].reshape(-1)/1, 20)
plt.hist(np.mean(ret[:,offset:],1)/dt, 30)
# plt.yscale('log')
plt.xlabel('rate', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.title('firing rate distribution', fontsize=20)
