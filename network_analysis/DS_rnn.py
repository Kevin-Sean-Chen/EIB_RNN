# -*- coding: utf-8 -*-
"""
Created on Wed May  7 01:18:40 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from scipy.signal import correlate

# np.random.seed(42)
# %% network setting
# Parameters
N = 300              # number of neurons
g = 1.5*1            # gain (scaling strength)
T = 2000             # total time steps
dt = 0.1             # Euler integration step
tau = 1.0            # time constant
time = np.arange(0, T*dt, dt)

# Initialize recurrent weights: J_ij ~ N(0, g^2 / N)
J = np.random.randn(N, N) * (g / np.sqrt(N))
w1 = np.random.rand(N)
w2 = np.random.rand(N)

# Initialize neural activity
x = np.random.randn(N)
r = np.tanh(x)  # initial output (nonlinearity)

# stim
amp = 1
stim_LR = np.cos(time/5)
stim_RL = -np.sin(time/5 - 10)
threshold = 0.8
stim_LR[stim_LR>threshold] = amp; stim_LR[stim_LR<threshold] = 0;
stim_RL[stim_RL>threshold] = amp; stim_RL[stim_RL<threshold] = 0;
plt.plot(time, stim_LR)
plt.plot(time, stim_RL)

# %% measurments
# Store activity of a few neurons
num_to_plot = 5
activity_lr = np.zeros((N*1, T))
activity_rl = activity_lr*1
activity_sp = activity_lr*1

# Run RNN with Euler integration
for t in range(T):
    dx = (-x/tau + J @ r + w1*stim_LR[t] + w2*stim_RL[t])
    x += dt * dx
    r = np.tanh(x)
    activity_lr[:, t] = r#[:num_to_plot]
    
for t in range(T):
    dx = (-x/tau + J @ r + w2*stim_LR[t] + w1*stim_RL[t])
    x += dt * dx
    r = np.tanh(x)
    activity_rl[:, t] = r#[:num_to_plot]

for t in range(T):
    dx = (-x/tau + J @ r + 0)
    x += dt * dx
    r = np.tanh(x)
    activity_sp[:, t] = r#[:num_to_plot]

# Plot dynamics
plt.figure(figsize=(10, 6))
for i in range(num_to_plot):
    plt.plot(time, activity_lr[i,:], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Activity')
plt.title(f'Chaotic RNN dynamics (g={g}, N={N})')
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure()
plt.imshow(activity_lr, aspect='auto')

# %%
nn = 2
plt.figure()
plt.plot(activity_lr[nn,:])
plt.plot(activity_rl[nn,:])

# %% population analysis
rect_lr = activity_lr*1
rect_lr[rect_lr<-0.] = 0
rect_rl = activity_rl*1
rect_rl[rect_rl<-0.] = 0

pos_lr = np.where(stim_LR>0)[0]
pos_rl = np.where(stim_RL>0)[0]
dsi = np.zeros(N)
for ii in range(N):
    nlr = np.mean(rect_lr[ii,pos_lr])
    nrl = np.mean(rect_rl[ii,pos_rl])
    dsi[ii] = (nlr-nrl)/(nlr+nrl)
    
plt.figure()
plt.hist(dsi,30)

# %% detect spontaneous direction
def find_delay(x,y):
    T = len(x)
    lags = np.arange(-T + 1, T)
    xcorr = correlate(y - np.mean(y), x - np.mean(x), mode='full')
    xcorr /= (np.std(x) * np.std(y) * T)  # normalize
    max_lag_index = np.argmax(xcorr)
    lag_to_peak = lags[max_lag_index]
    return lag_to_peak

# %% iterate
cause_ij = np.zeros((N, N))
for ii in range(N):
    for jj in range(N):
        if ii < jj:
            print(ii)
            cause_ij[ii,jj] = find_delay(activity_sp[ii,:], activity_sp[jj,:])
        
# %% spotaneous lags
plt.figure()
plt.imshow(cause_ij)

upper_right = np.triu(cause_ij, k=1)
values = upper_right[upper_right != 0]
plt.figure()
plt.hist(values,100)

# %% next...
# compare this with pair-wise DSI??