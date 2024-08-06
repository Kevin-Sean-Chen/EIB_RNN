# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:58:14 2024

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

from scipy.linalg import hankel
from mpl_toolkits.mplot3d import Axes3D

# Set the seed for demo
np.random.seed(42)

# %% 
# 3.1 Design your stimulus
###############################################################################
# %% generate stimuli
T = 30  # seconds
dt = 0.002  # 2ms
time_full = np.arange(0, T, dt)  # time vector
lt = len(time_full)
s_t = np.random.randn(lt)  # Gaussian stimuli

# %% 
# 3.2 Simulate data
###############################################################################
# %% setup linear filter
tau = 0.010  # 10ms
ww = 0.3*1000 #0.3 rad/ms

time_filter = np.arange(0, 0.05, dt)
linear_filter = np.exp(-time_filter/tau)*np.sin(ww*time_filter)  # filter form

plt.figure()
plt.plot(-time_filter, linear_filter)
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('weights', fontsize=20)
plt.title('temporal kernel', fontsize=20)

# %% convolution
kernel = np.fliplr(linear_filter[None,:]).squeeze()  # change to make it causal in time
u_f_t = np.convolve(s_t, kernel, mode='same')  # linear signal

# %% firing nonlinearity
theta = 5
delta = 1
r_max = 1*4  # tune to have ~ 20Hz
def NL(ut, r_max=r_max):
    """
    spiking nonlinearity
    """
    nl = r_max/(1+np.exp((theta-ut)/delta))  # firing nonlinearity
    return nl

firing_rate = NL(u_f_t)

# %% rate to spikes
spike_t = np.zeros(lt)
pos = np.where(firing_rate*1 > np.random.rand(lt))[0]  # generating spikes from firing rate
spike_t[pos] = 1

plt.figure()
plt.plot(time_full, spike_t)
plt.ylabel('spikes', fontsize=20)
plt.xlabel('time (s)', fontsize=20)
plt.title('simulated spike trains', fontsize=20)
# plt.xlim([0,1]); plt.yticks([])

# %% full LN model, and find r_max
def LN_spk_model(st, r_max=r_max):
    """
    linear-nonlinear spiking model
    function takes in st as the stimulus vector and produces spike_t spike train
    an optional parameter r_max to scan through parameters
    """
    u_f_t = np.convolve(st, kernel, mode='same')  # linear part
    firing_rate = NL(u_f_t, r_max)  # nonlinear part
    lt = len(st)
    spike_t = np.zeros(lt)
    pos = np.where(firing_rate*1 > np.random.rand(lt))[0]  # spiking process
    spike_t[pos] = 1
    return spike_t

rs = np.arange(1,10,0.5)
spk_counts = np.zeros(len(rs))
for rr in range(len(rs)):
    spk_t = LN_spk_model(s_t, rs[rr])
    spk_counts[rr] = np.sum(spk_t)/T

plt.figure()
plt.plot(rs, spk_counts)
plt.xlabel('r_max', fontsize=20)
plt.ylabel('firing (Hz)', fontsize=20)
plt.title('finding the right r_max', fontsize=20)

# %% 
# 3.3 STA
###############################################################################
# %% computing STA
def compute_STA(st, spk):
    """
    given stimulus st and spike spk, compute spike triggered average sta
    """
    pos = np.where(spk>0)[0]
    acs_l = len(time_filter)  # use the kernel window length
    sta = np.zeros(acs_l)  # sta form
    offset = acs_l//2+1  # to account for index mismatch in convolution
    for pp in range(len(pos)):
        if pos[pp]>acs_l and pos[pp]+offset<len(st):
            sta += st[pos[pp]-acs_l + offset:pos[pp]+offset]
    sta /= len(pos)
    return sta

sta_est = compute_STA(s_t, spike_t)
plt.figure()
plt.plot(-time_filter,linear_filter, label='true filter')
plt.plot(-time_filter,sta_est, label='STA')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('weights', fontsize=20)
plt.legend(fontsize=20)
plt.title('STA (30s, ~700 spikes)', fontsize=20)

# %% scaling
Ts = np.array([1,5,10,20,50,100])  # length of spike train
MSEs = np.zeros(len(Ts))  # measuring recovery error
n_spk = MSEs*0
for tt in range(len(Ts)):
    T = Ts[tt]
    time_full = np.arange(0, T, dt)
    lt = len(time_full)
    s_t = np.random.randn(lt)
    spike_t = LN_spk_model(s_t)
    sta_est = compute_STA(s_t, spike_t)
    MSEs[tt] = np.sum((sta_est - linear_filter)**2)
    n_spk[tt] = np.sum(spike_t)
    
plt.figure()
plt.plot(n_spk, MSEs, '-o')
plt.xlabel('number of spikes', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.title('inference performance with spikes', fontsize=20)

# %%
# 3.4 STC: one-dimensional case
###############################################################################
# %% STC
def QNL(ut, r_max=r_max):
    """
    Quadratic spiking nonlinearity
    """
    nl = r_max/(1+np.exp(((theta-ut**2))/delta))  # firing nonlinearity ###!!??
    return nl
def QLN_spk_model(st, r_max=r_max):
    """
    Quadratic version of the linear-nonlinear model
    takes in st stimululi and returns spike train spike_t
    """
    # u_f_t = np.convolve(st, kernel, mode='same')  # linear
    padded_time_series = np.pad(st, (len(kernel) - 1, 0), mode='constant')
    convolved = np.convolve(st, kernel, mode='full')  # linear part
    u_f_t = convolved[:len(st)]
    firing_rate = QNL(u_f_t, r_max)  # nonlinear part
    lt = len(st)
    spike_t = np.zeros(lt)
    pos = np.where(firing_rate*1 > np.random.rand(lt))[0]  # spiking process
    spike_t[pos] = 1
    return spike_t
T = 1000
time_full = np.arange(0, T, dt)
lt = len(time_full)
s_t = np.random.randn(lt)+0
spike_t = QLN_spk_model(s_t)
sta_est = compute_STA(s_t, spike_t)

plt.figure()
plt.plot(-time_filter,linear_filter, label='true filter')
plt.plot(-time_filter,sta_est, label='STA')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('weights', fontsize=20)
plt.legend(fontsize=20)
plt.title('STA for quadratic nonlinear', fontsize=20)

# %% compute STC
D = len(time_filter)
xx = s_t*1
X = hankel(np.append(np.zeros(D-1),xx[:lt-D+1]),xx[lt-D+0:])  ### building design matrix with a fancy trick!
# X = hankel(np.append(np.zeros(D-2),xx[:lt-D+2]),xx[lt-D+1:]).shape
# X = np.concatenate((np.ones([lt,1]),X),axis=1)  ### if you want to model offset...

X_demean = X - sta_est
C_stc = (spike_t*X_demean.T) @ X_demean - 1/X.shape[0]*X.T@X  # compute spike triggered covariance

# %% finding the feature
uu,ss,vv = np.linalg.svd(C_stc)  # decompose the STC matrix
stc_est = uu[:,0]*-1  # correct up to a sign flip
plt.figure()
plt.plot(-time_filter, linear_filter, label='true filter')
plt.plot(-time_filter, stc_est, label='STC')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('weights', fontsize=20)
plt.legend(fontsize=20)
plt.title('STC for quadratic nonlinear', fontsize=20)

# %%
# 3.5 STC: two-dimensional case
###############################################################################
# %% 2D case!
theta = 15
delta = 2
tau_g, ww_g = 0.03, 0.2*1000
### make filter g
linear_filter_g = -np.exp(-time_filter/tau_g)*np.cos(ww_g*time_filter)

def NL_2d(uf, ug):
    """
    2D nonlinearity
    """
    # nl = r_max/( (1+np.exp((theta-uf)**2/delta))*(1+np.exp((theta-ug)**2/delta)) ) ##### modified !!??
    nl = r_max/( (1+np.exp((theta-uf**2)/delta))*(1+np.exp((theta-ug**2)/delta)) )
    return nl

# %% generating spikes from 2D model
def causal_filtering(st, kernel):
    padded_time_series = np.pad(st, (len(kernel) - 1, 0), mode='constant')
    convolved = np.convolve(st, kernel, mode='full')
    u = convolved[:len(st)]
    return u

lt = 1000000  # when this is large do not plot the scattering 3D one

### build linear part for f
s_t_f = np.random.randn(lt)*3
kernel_f = kernel*1 
uf = causal_filtering(s_t_f, kernel_f)

### build linear part for g
s_t_g = np.random.randn(lt)*3
kernel_g = np.fliplr(linear_filter_g[None,:]).squeeze()  # change to make it causal in time
ug = causal_filtering(s_t_g, kernel_g) 
rate_2d = NL_2d(uf, ug)
spike_2d = np.zeros(lt)
pos = np.where(rate_2d*1 > np.random.rand(lt))[0]  # spiking
spike_2d[pos] = 1

print('total spike count: ', np.sum(spike_2d))

# %% plotting 3D nonlinear function (use this when lt is short otherwise it takes time...)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(uf, ug, rate_2d, c='b', marker='o')
# ax.set_xlabel('uf')
# ax.set_ylabel('ug')
# ax.set_zlabel('rate')

# %% compute STC for 2D
def compute_STC(st, spk):
    """
    function to compute feature from STC given stimuli st and spike train spk
    """
    D = len(time_filter)
    xx = st*1
    X = hankel(np.append(np.zeros(D-1),xx[:lt-D+1]),xx[lt-D+0:])  # design matrix
    C_stc = (spk*X.T) @ X
    uu,ss,vv = np.linalg.svd(C_stc)
    return uu[:,0]

stc_est_f = compute_STC(s_t_f, spike_2d)
stc_est_g = compute_STC(s_t_g, spike_2d)

plt.figure()
plt.plot(-time_filter, linear_filter, 'b',label='true filter f')
plt.plot(-time_filter, linear_filter_g, 'g',label='true filter g')
plt.plot(-time_filter, -stc_est_f, 'b--', label='STC f')
plt.plot(-time_filter, stc_est_g*2, 'g--', label='STC g')  # I need to check if rescaling is ok...
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('weights', fontsize=20)
plt.legend(fontsize=7)
plt.title('2D STC result', fontsize=20)


# %% Sim fun for 2.2
###############################################################################
C = 1
gL = 0.1
EL = -65
VR = -65
VT = -50
ES = -40  # larger than VT

def FI_curve(gs):
    fac = (gL*EL-gs*ES)/(gs+gL)
    f = -gL/C* np.log((VT-fac)/(VR-fac))
    return f

gss = np.arange(0.1,2,0.2)
fs = np.zeros(len(gss))
for ss in range(len(gss)):
    fs[ss] = FI_curve(gss[ss])

plt.figure()
plt.plot(gss, fs*1000,'-o')
plt.xlabel('g_s', fontsize=20)
plt.ylabel('firing rate (Hz)', fontsize=20)
