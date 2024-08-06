# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:52:02 2024

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

from scipy.stats import expon
import scipy as sp

# %% Part 1 Linear algbra
N = 50  # input dim
M = 10  # output dim
W = np.random.uniform(-1,1,size=(N,M))  # uniform weights

I = np.random.randn(N)  # input pattern

y = W.T@I  #output

# %% Part 2 Random walk
def GenerateVoltage(p,T,Vreset,Vthresh,V0):
    V = np.zeros(T)
    nspk = 0
    spk_t = []
    V[0] = V0
    for t in range(1,T):
        if np.random.rand()<p:  # probablistic steps
            dV = +1
        else:
            dV = -1
        V[t] = V[t-1] + dV  # random walk
        
        if V[t]>Vthresh:  # passing threshold
            V[t] = Vreset  # reset value
            nspk += 1
            spk_t.append(t)
    return V, nspk, spk_t

V0 = -65
Vthresh = -45
Vreset = -70
dt = 0.001
T = int(1/dt)

# %% scanning p
ps = np.arange(0.1,1,.1)
spk_rate = np.zeros(len(ps))
for ii in range(len(ps)):
    _, nspk,_ = GenerateVoltage(ps[ii], T, Vreset, Vthresh, V0)
    spk_rate[ii] = nspk

plt.figure()
plt.plot(ps, spk_rate)
plt.xlabel('p')
plt.ylabel('firing rate (Hz)')

# %% example trace
p = 0.6
Vout, nspk, spk_t = GenerateVoltage(p,T,Vreset,Vthresh,V0)
time_vec = np.arange(0,1,dt)
plt.figure()
plt.plot(time_vec, Vout)
plt.plot(time_vec[spk_t], np.ones(len(spk_t))+Vthresh, 'ro')
plt.xlabel('time (s)')
plt.ylabel('voltage')

# %% Part 3 Convlute spike trains
T = 3  # 3 seconds
dt = 0.001  # 1ms step
time_vec = np.arange(0,T,dt)
N = len(time_vec)  # length of time vector
target_rage = 20  # Hz we want for spike rate
p = 20*dt
spiketrain = np.random.rand(N)>(1-p)
spiketrain = spiketrain.astype(float)
print('firing rate is:',sum(spiketrain)/T,' Hz')

# %% kernel
times = np.arange(-0.05, 0.05, dt)
mu = 5*dt
k = 1/mu*expon.pdf(times/mu)  # kernel form
plt.figure()
plt.plot(times,k)
plt.ylabel('kernel weight')
plt.xlabel('time (s)')

# %% conv-spk
conv_spk = np.convolve(spiketrain, k, mode='same')
plt.figure()
# plt.subplot(211)
# plt.plot(time_vec, spiketrain)
# plt.subplot(212)
# plt.plot(time_vec, conv_spk)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(time_vec, spiketrain)
ax2.plot(time_vec, conv_spk)
ax2.set_xlabel('time (s)')

# %% Part 4
k = -0.125*np.ones((3,3))
k[1][1] = 1.0
octopus = plt.imread('C:/Users/kevin/Documents/github/octopus_1.png') 
plt.figure()
plt.imshow(np.mean(octopus,2), cmap='gray')


mean_oct = np.mean(octopus,2)

# %%
conved_oct = mean_oct*0

# for ii in range(4):
#     conved_oct[:,:,ii] = np.abs(sp.signal.convolve2d(octopus[:,:,ii].squeeze(), k, mode='same', fillvalue=0))

conved_oct = np.abs(sp.signal.convolve2d(mean_oct, k, mode='same', fillvalue=0))
plt.figure()
plt.imshow(conved_oct,  cmap='gray')
