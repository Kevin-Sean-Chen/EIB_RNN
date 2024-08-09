# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 21:11:50 2024

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# Set the seed for demo
np.random.seed(42)

# %%
# %% WTA network
###############################################################################
# %%
### What is the condition for one of the neurons to be driven to zero firing rate? What additional condition \\
### (on the synaptic weights) will lead the winner to exhibit runaway growth?
# %%
### network
ws = .2#0.2*1
wo = -0.7*5
W = np.array([[ws, wo],[wo, ws]])

### time
dt = 0.001  #1ms step
tau = 0.018  #ms
T = 5  # in seconds
time = np.arange(0,T,dt)
lt = len(time)

### nonlinearity
def NL(x):
    r_max, theta = 100, 5
    nl = np.where(x > 0, x, 0)   ### ReLU
    # nl = np.clip(nl, a_min=None, a_max=r_max)
    # nl = r_max/(1+np.exp(-x - theta))  ### sigmoid version
    return nl 

### initialize and make stimuli
b1,b2 = 0,1
v1,v2 = np.array([1,1]), np.array([1,-1])
It = np.zeros((2, lt)) + (b1*v1+b2*v2)[:,None]
# It = np.zeros((2,lt)) + np.array([63, 57])[:,None]
It = np.zeros((2,lt)) + np.array([30, 30])[:,None]*2  ### according to 2e setting
# It = np.zeros((2,lt)) + np.array([57, 63])[:,None]
vt = np.zeros((2, lt))
rt = vt*1

### dynamics
for tt in range(lt-1):
    vt[:,tt+1] = vt[:,tt] + dt/tau*(-vt[:,tt] + W@NL(vt[:,tt]) + It[:,tt])
    rt[:,tt+1] = NL(vt[:,tt+1])
    
### analysis
plt.figure()
plt.plot(time, rt.T)
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('firing (Hz)', fontsize=20)
plt.legend(['r1', 'r2'], fontsize=20)

# %%
Jac = np.array([[-1+ws, wo],
                [wo, -1+ws]])
### trace is less than

# %%
###############################################################################
# %% (II) perceptual rivalry
###############################################################################
ws = 0.2*1
wo = -0.7*10
W = np.array([[ws, wo],[wo, ws]])
T = 100  # in seconds
time = np.arange(0,T,dt)
lt = len(time)
def NL(x):
    # nl = np.where(x > 0, x, 0)   ### ReLU
    rmax, theta, beta = 10, -0, 1/4
    nl = rmax/(1+np.exp(-beta*(x - theta)))  ### sigmoid version
    return nl

### new input and noist parameter
It = np.zeros((2,lt)) + np.array([20, 20])[:,None] 
vt = np.zeros((2,lt))
rt = vt*1
D = 100  # noise level

### dynamics
for tt in range(lt-1):
    vt[:,tt+1] = vt[:,tt] + dt/tau*(-vt[:,tt] + W@NL(vt[:,tt]) + It[:,tt]) + np.random.randn(2)*D*dt**0.5
    rt[:,tt+1] = NL(vt[:,tt+1])

plt.figure()
plt.plot(time, rt.T)
plt.xlabel('time', fontsize=20)
plt.ylabel('activity', fontsize=20)
# plt.xlim([0,50])

# %% bistable proof
plt.figure()
plt.hist(rt[0,:], 50, density=True)
plt.xlabel('r1', fontsize=20)
plt.ylabel('density', fontsize=20)

# %% play with dwell time...
temp = rt[1,:]*1
rt_analysis = temp*0
rt_analysis[temp>np.mean(temp)] = 1

def calculate_binary_dwell_times(binary_vector):
    dwell_times = []
    current_state = binary_vector[0]
    start_index = 0

    for i in range(1, len(binary_vector)):
        if binary_vector[i] != current_state:
            dwell_time = i - start_index
            dwell_times.append((current_state, dwell_time))
            current_state = binary_vector[i]
            start_index = i

    # Append the dwell time for the last state
    dwell_time = len(binary_vector) - start_index
    # dwell_times.append((current_state, dwell_time))
    # dwell_times.append((dwell_time))
    return np.array(dwell_times)[:,1]

dwell_time = calculate_binary_dwell_times(rt_analysis)*dt

# %%
plt.figure()
plt.hist(dwell_time,15)
plt.xlabel('dwell time', fontsize=20)
plt.yscale('log')
plt.xlim([0,3])



# %%
###############################################################################
# %%
# %% BCM modeling
###############################################################################
# %% setup
# T = 10 s, η = 1 × 10−6 ms−1, y0 = 10, τ = 50 ms. Use the Euler method with a timestep of 1 ms. You should also put a hard
# bound for the weights at 0 (when wi < 0, set it to 0)
T = 10  # 10 s
dt = 0.001  # 1ms steps
eta = 10**-6 * 1000 # per second learning rate
tau = 0.050 *1  # 50 ms
y0 = 10  # 10

time = np.arange(0,T,dt)
lt = len(time)
# %% input
# two input patterns x1 = (20, 0) and x2 = (0, 20). At each timestep, one of the two patterns is presented to the neuron,
# the pattern is chosen randomly with a probability 0.5 of being pattern x1 and 0.5 of being pattern x2
x1 = np.array([20, 0])
x2 = np.array([0, 20])
p_x = 0.5

# %% dynanmics
wt = np.zeros((2,lt))
wt[:,0] = -np.random.rand(2)*1  # random initial condition
yt = np.zeros(lt)
thetat = yt*1

def cap_weights(x):
    xx = x*1
    xx[x>0] = 0
    return xx

for tt in range(lt-1):
    if np.random.rand()>p_x:
        xt = x1*1
    else:
        xt = x2*1
    yt[tt] = wt[:,tt]@xt
    thetat[tt+1] = thetat[tt] + dt/tau*(-thetat[tt] + yt[tt]**2/y0)  # adaptive threshold
    ### BCM model
    wt[:,tt+1] = wt[:,tt] + dt*eta*xt*yt[tt]*(yt[tt] - thetat[tt])  # weight dynamics
    ### Hebbian
    # wt[:,tt+1] = wt[:,tt] + dt*eta*xt*yt[tt]
    wt[:,tt+1] = cap_weights(wt[:,tt+1])
    
# %%  plotting
plt.figure(figsize=(7,7))
plt.subplot(311)
plt.plot(time, wt.T)
plt.ylabel('weights', fontsize=20)
plt.subplot(312)
plt.plot(time, thetat)
plt.ylabel('theta', fontsize=20)
plt.subplot(313)
plt.plot(time, yt)
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('y', fontsize=20)
