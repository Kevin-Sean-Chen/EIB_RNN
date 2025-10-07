# -*- coding: utf-8 -*-
"""
Created on Thu May 22 01:37:18 2025

@author: kevin
"""

from scipy.signal import convolve2d
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from scipy.signal import correlate
import random

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
import cv2

# %% comparison of 1/N vs. 1/sqrt(N) network
### testing the form with rate iteration, rather than current

# %% parameter settings
N = 40  # neurons
tau_e = 0.005*1.  # time constant ( 5ms in seconds )
sig_e = 0.1    # spatial kernel
tau_i = 0.015
sig_i = 0.2*1.    ### important parameters!!

tau_i, sig_i = 15*0.001*1., 0.20    ### important parameters!!
#### 5, 0.14  ### grid parameter
#### 15, 0.2  ### chaos parameter
#### 10, 0.11 ### waves/strips!!!
#### 8,  0.2  ### blinking
#### 15, 0.14 ### switching waves!!

amp = 0

# %% balanced condition
rescale = 4. #.25,.5,1,2,4,8,16
base = 1/11 ### larger->regular(1/20), smaller->chaotic(1/40)

Ke_sqrt = 1.*(N**2*sig_e**2*np.pi*1)**0.5
Ki_sqrt = 1.*(N**2*sig_i**2*np.pi*1)**0.5

### chosen
Wee = 1. *Ke_sqrt*rescale  # recurrent weights
Wei = -2. *Ki_sqrt*rescale
Wie = .99 *Ke_sqrt*rescale
Wii = -1.8 *Ki_sqrt*rescale
mu_e = 1.*rescale *N*base #*Ke_sqrt/2 *1. # .8: chaos, .9: complex, 1.:flow (?)
mu_i = .8*rescale *N*base #*Ki_sqrt/4 #

### Huck's
# Wee = .94*Ke_sqrt  *rescale  # recurrent weights
# Wei = -3.9*Ki_sqrt *rescale
# Wie = 2.03*Ke_sqrt *rescale
# Wii = -2.1*Ki_sqrt *rescale
# mu_e = .18*Ke_sqrt  ###(N**2*sig_e**2*np.pi*1)**0.5
# mu_i = .0005*Ki_sqrt

stim_scal = amp #/ N**0.5

# %% unbalanced condition
###########################
### find a way to confirm 1/N connection!!
### find MF chaos that is NOT balanced...
###########################
# rescale = N/2. #7 #N/2  #8 20 30... linear with N

# # Wee = 1. *rescale  # recurrent weights
# # Wei = -1. *rescale
# # Wie = 1. *rescale
# # Wii = -1. *rescale
# # mu_e = .01 *1
# # mu_i = .01 *1

# Wee = 1. *rescale  # recurrent weights
# Wei = -2. *rescale
# Wie = .99 *rescale
# Wii = -1.8 *rescale
# mu_e = 1 *1
# mu_i = .8 *1

# stim_scal = amp #/ N

# %% network setup
### setting up space and time
dt = 0.001  # 1ms time steps
T = 1.0  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
space_vec = np.linspace(0,1,N)
kernel_size = 37#N-1 #37  # pick this for numerical convolution

### stim params
pattern = np.zeros((N,N))
pattern[10:15,10:15] = 1
stim_amp = mu_e*2
stim_dur = 10

### random initial conditions
re_xy[:,:,0] = np.random.rand(N,N)*.1
ri_xy[:,:,0] = np.random.rand(N,N)*.1
he_xy = re_xy*1
hi_xy = ri_xy*1

### measure one cell
measure_e = np.zeros(lt)
measure_i = np.zeros(lt)

### measure the field
measure_mu = np.zeros((N,N,lt))
measure_mu_ex = np.zeros((N,N,lt))

def phi(x):
    """
    rectified quardratic nonlinearity
    """
    # nl = np.where(x > 0, x**2, 0)
    # nl = np.where(x > 0, x*1, 0)  ### why a scaling factor needed!?????????????????????
    # nl = 1/(1+np.exp(-x))
    nl = np.where(x > 0, np.tanh(x)*1, 0)
    return nl

def g_kernel(sigma, size=kernel_size):
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

# %% dynamics
### neural dynamics
for tt in range(lt-1):
    ### stim
    if tt>lt//2 and tt<lt//2+stim_dur:
        stim = stim_amp
    else:
        stim = 0
        # mu_e = 5
    
    ### phi(W*r)
    ## modifying for 2D rate EI-RNN
    ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    re_xy[:,:,tt+1] = re_xy[:,:,tt] + dt/tau_e*( -re_xy[:,:,tt] + phi(Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) + stim*pattern)
    ri_xy[:,:,tt+1] = ri_xy[:,:,tt] + dt/tau_i*( -ri_xy[:,:,tt] + phi(Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
    
    ### W*r
    # ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    # gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    # he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e))                  
    # hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i))
    # re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
    # ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
    
    ### make E-I measurements
    measure_e[tt+1] = (Wee*ge_conv_re + mu_e)[20,20]
    measure_i[tt+1] = (Wei*gi_conv_ri)[20,20]
    
    ### mean measurements
    measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    # measure_mu[:,:,tt+1] = (  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
    he_xy[:,:,tt+1] = Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e 

# %% stim response
plt.figure()
plt.plot(np.mean(np.mean(re_xy[:,:,:],0),0))
stim_vec = np.zeros((lt))
stim_vec[lt//2+1:lt//2+10+1] = 1
plt.plot(stim_vec)
plt.xlim([450, 550])

# %%
offset = 50
plt.figure()
plt.plot(time[offset:], measure_e[offset:], label='E')
plt.plot(time[offset:], measure_i[offset:], label='I')
plt.plot(time[offset:], (measure_e+measure_i)[offset:],label='total')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('current', fontsize=20)
# plt.title('spatial balancing', fontsize=20)
plt.legend(fontsize=15)

# %% plot dynamics
offset = 1
plt.figure()
plt.plot(time[offset:], re_xy[20,20,offset:].squeeze(), label='E cell1')
plt.plot(time[offset:], re_xy[15,15,offset:].squeeze(), label='E cell2')
plt.plot(time[offset:], ri_xy[15,15,offset:].squeeze(), label='I cell2')
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('activity', fontsize=20)
plt.legend(loc='lower center', bbox_to_anchor=(1, -.3), fontsize=7)
# Adjust the layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.75, 1])
# plt.xlim([0.1,0.14])

# %% pot beta for a single cell
beta_i = np.abs(measure_e + measure_i)/measure_e

plt.figure()
plt.plot(time, beta_i)
plt.xlabel('time', fontsize=20)
plt.ylabel('beta', fontsize=20)
print(np.abs(np.mean(measure_e + measure_i)) / np.mean(measure_e))

# %% net flow analysis
def get_optical_flow(prev, next_):
    ### cv2 function for optical flow
    flow = cv2.calcOpticalFlowFarneback(prev.astype(np.float32), next_.astype(np.float32), 
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def make255(A):
    A = A-np.min(A)
    A_log = np.log1p(A)  
    A_prepped = np.zeros_like(A_log)
    for t in range(A.shape[2]):
        frame = A_log[:, :, t]
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-12)
        A_prepped[:, :, t] = (frame_norm * 255).astype(np.uint8)
    return A_prepped

def tensor2speed(A, dt=1):
    ### compute net speed
    flows = [get_optical_flow(re_xy[:, :, t], A[:, :, t+dt]) for t in range(500, lt - dt)]
    avg_flows = np.array([flow.reshape(-1, 2).mean(axis=0) for flow in flows])
    net_motion = avg_flows.sum(axis=0)
    net_speed = np.linalg.norm(net_motion) / ((lt-dt)/2)
    return net_speed, net_motion

def tensor2speed2(A, dt=1):
    T = A.shape[2]
    frame_speeds = []

    for t in range(500, T - dt):  # assumes t starts at 500 for some reason
        flow = get_optical_flow(A[:, :, t], A[:, :, t + dt])
        speed = np.linalg.norm(flow.reshape(-1, 2), axis=1).mean()
        frame_speeds.append(speed)

    avg_speed = np.mean(frame_speeds)
    return avg_speed, frame_speeds


net_speed, net_vec = tensor2speed2(make255(re_xy), 1)
print('motion speed: ', net_speed)

# %% scanning
###############################################################################
# %% functional
def sim_2D_EI(N, lt=lt, sigs=(sig_e, sig_i), rescale=1):
    
    ### change parameters
    sig_e, sig_i = sigs
    # rescale = 1 #.25,.5,1,2,4,8

    Ke_sqrt = 1.*(N**2*sig_e**2*np.pi*1)**0.5
    Ki_sqrt = 1.*(N**2*sig_i**2*np.pi*1)**0.5

    ### chosen
    Wee = 1. *Ke_sqrt*rescale  # recurrent weights
    Wei = -2. *Ki_sqrt*rescale
    Wie = .99 *Ke_sqrt*rescale
    Wii = -1.8 *Ki_sqrt*rescale
    mu_e = 1.*rescale *N/11 #50 #*Ke_sqrt/2 *1. # .8: chaos, .9: complex, 1.:flow (?)
    mu_i = .8*rescale *N/11 #50 #*Ki_sqrt/4
    
    ### prep
    re_xy = np.zeros((N,N, lt))
    ri_xy = re_xy*1

    ### random initial conditions
    re_xy[:,:,0] = np.random.rand(N,N)*.1
    ri_xy[:,:,0] = np.random.rand(N,N)*.1
    he_xy = re_xy*1
    hi_xy = ri_xy*1

    ### measure the field
    measure_mu = np.zeros((N,N,lt))
    measure_mu_ex = np.zeros((N,N,lt))
    
    ### dynamics
    for tt in range(lt-1):
        
        ## modifying for 2D rate EI-RNN
        ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
        gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
        re_xy[:,:,tt+1] = re_xy[:,:,tt] + dt/tau_e*( -re_xy[:,:,tt] + phi(Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
        ri_xy[:,:,tt+1] = ri_xy[:,:,tt] + dt/tau_i*( -ri_xy[:,:,tt] + phi(Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
    
        ### modifying for 2D rate EI-RNN
        # ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
        # gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
        # he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
                        
        # hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
        # re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
        # ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])       
        
        ### mean measurements
        measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
        measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
        he_xy[:,:,tt+1] = (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e)
    
    beta_t = measure_mu / measure_mu_ex 
    
    ### semi-beta
    pos = np.where(re_xy>0)[0]
    semi_beta = measure_mu.reshape(-1)[pos]/measure_mu_ex.reshape(-1)[pos]
    return he_xy, beta_t, np.nanmean(semi_beta)
    # return re_xy, beta_t, np.nanmean(semi_beta)

def pca_dim(re_xy, dim=0.9):
    N = re_xy.shape[0]
    flatten_r = re_xy.reshape(N**2, lt)
    uu,ss,vv = np.linalg.svd(np.cov(flatten_r))
    cums = np.cumsum(ss/np.sum(ss))
    if len(np.where(cums<dim)[0])>0:
        dim = np.where(cums<dim)[0][-1] + 1
    else: dim=0
    return dim, cums

def sparse_theta(data):
    # Compute non-zero fraction along the T dimension for each (i, j)
    data[data<0.5] = 0
    non_zero_fraction = 1-np.count_nonzero(data, axis=2) / data.shape[2]

    # Average this fraction over all (i, j)
    average_fraction = np.mean(non_zero_fraction)

    return average_fraction

def autocorr1(x):
    '''numpy.corrcoef, partial'''
    x = x - np.mean(x)
    lags = np.arange(0, len(x)-int(0.1*len(x)))
    corr=[np.dot(x, x)/len(x) if l==0 else np.dot(x[l:],x[:-l])/len(x[l:]) for l in lags]
    return np.array(corr)

def crosscorr1(x,y):
    '''numpy.corrcoef, partial'''
    x = x - np.mean(x)
    y = y - np.mean(y)
    lags = np.arange(0, len(x)-int(0.1*len(x)))
    corr=[np.dot(x, y)/len(x) if l==0 else np.dot(x[l:],y[:-l])/len(x[l:]) for l in lags]
    return np.array(corr)

def measure_SNR_group(r_xyt, lags=int(lt//2), post=int(tau_i/dt*5)):
    acf_pop = np.zeros(lags)
    for ii in range(N):
        for jj in range(N):
            temp_corr = autocorr1(r_xyt[ii,jj,:])[:lags] #- np.mean(r_xyt[ii,jj,:])**2
            acf_pop = acf_pop + temp_corr/N**2
    
    autocorrs_nois = acf_pop[0] - np.mean((acf_pop[post:])**2)**0.5
    autocorrs_sigs = np.mean((acf_pop[post:])**2)**0.5
    return acf_pop #autocorrs_sigs, autocorrs_nois

def unique_pairs(N, k):
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]  # Generate all unique pairs (i, j) with i < j
    selected_pairs = random.sample(all_pairs, min(k, len(all_pairs)))  # Randomly pick k pairs
    return selected_pairs

def measure_SNR_cross(r_xyt, lags=int(lt//2), post=int(tau_i/dt*5), k_pairs=100):
    pairs = unique_pairs(N**2, k_pairs)
    acf_pop = np.zeros(lags)
    look_up = np.arange(N**2, ).reshape(N,N)
    autocorrs_nois, autocorrs_sigs = np.zeros(k_pairs), np.zeros(k_pairs)
    for kk in range(k_pairs):
        pair_k = pairs[kk]
        a_neuron = np.concatenate(np.where(look_up==pair_k[0]))
        b_neuron = np.concatenate(np.where(look_up==pair_k[1]))
        temp_corr = crosscorr1(r_xyt[a_neuron[0],a_neuron[1],:], r_xyt[b_neuron[0],b_neuron[1],:])[:lags] #- np.mean(r_xyt[ii,jj,:])**2
        # autocorrs_nois[kk] = temp_corr[0] - np.mean((temp_corr[post:])**2)**0.5
        # autocorrs_sigs[kk] = np.mean((temp_corr[post:])**2)**0.5
        acf_pop = acf_pop + temp_corr/k_pairs
    # autocorrs_nois = acf_pop[0] - np.mean((acf_pop[post:])**2)**0.5
    # autocorrs_sigs = np.mean((acf_pop[post:])**2)**0.5
    return acf_pop #autocorrs_sigs, autocorrs_nois

# %% scanning
Ks = np.array([0.5, 1, 2, 4, 8, 16])
Ks = np.array([1,2,3,4,5]) ### can probably work in linear scale given it is sqrt(K)?
reps = 1
var_r = np.zeros(len(Ks))
mea_r = var_r*1
mea_beta = var_r*1
var_beta = var_r*1
beta_store = []
dims, cums = np.zeros((len(Ks), reps)), []
sparsity = np.zeros((len(Ks), reps))
semi_beta = np.zeros((len(Ks), reps))
x_corr = np.zeros((len(Ks), int(lt//2)))
a_corr = np.zeros((len(Ks), int(lt//2)))
speeds = np.zeros((len(Ks), reps))

for ki in range(len(Ks)):
    temp_mean = []
    temp_beta = []
    temp_beta_raw = []
    temp_dim = []
    for rr in range(reps):
        print(ki)
        ### sim network
        re_xyi, beta_it, semi_b = sim_2D_EI(N, lt, rescale=Ks[ki])#Ns[ni]-1)
        # re_xyi, beta_it = sim_2D_EI(Ns[2], lt, k_size=37, iptI=Ss[0], stim=stim, sigs=(sig_e, 0.2)) ############ N-1 #############
        ### activity
        temp_mean.append(np.mean(re_xyi,2).reshape(-1))
        ### beta
        temp_beta.append(np.nanmedian(beta_it[:,:,50:], 2).reshape(-1))
        temp_beta_raw.append(beta_it[:,:,lt-1].reshape(-1))
        temp_beta_raw.append(beta_it[:,:,lt//2].reshape(-1))
        dim_, cum_ = pca_dim(re_xyi)
        dims[ki, rr] = dim_
        sparsity[ki, rr] = sparse_theta(re_xyi)
        semi_beta[ki, rr] = semi_b
        ### correlation
        x_corr[ki,:] = measure_SNR_cross(re_xyi)
        a_corr[ki,:] = measure_SNR_group(re_xyi)
        ### net speed measurement
        net_speed, net_vec = tensor2speed(make255(re_xyi), 1)
        speeds[ki, rr] = net_speed
        
    ### record
    var_r[ki] = np.var(np.array(temp_mean))
    mea_r[ki] = np.mean(np.array(temp_mean))  
    mea_beta[ki] = np.mean(np.array(temp_beta))
    var_beta[ki] = np.var(np.array(temp_beta))
    
    beta_store.append(np.concatenate(temp_beta_raw)) ### store all beta_t
    cums.append(cum_)
    
# %% plotting
plt.figure()
plt.plot(Ks, var_r, '-o')
plt.xlabel('scaled sqrt(K)', fontsize=20)
plt.ylabel('population var', fontsize=20)

plt.figure()
plt.errorbar(Ks, mea_r, yerr=var_r**0.5, fmt='-o')
plt.xlabel('scaled sqrt(K)', fontsize=20)
plt.ylabel('population mean', fontsize=20)

plt.figure()
plt.plot(Ks, var_beta, '-o')
plt.xlabel('scaled sqrt(K)', fontsize=20)
plt.ylabel('beta var', fontsize=20)

plt.figure()
plt.plot(Ks, semi_beta, '-o')
plt.xlabel('scaled sqrt(K)', fontsize=20)
plt.ylabel('semi-beta', fontsize=20)

plt.figure()
plt.errorbar(Ks, mea_beta, yerr=var_beta**0.5, fmt='-o')
plt.xlabel('scaled sqrt(K)', fontsize=20)
plt.ylabel('beta mean', fontsize=20)

plt.figure()
plt.plot(Ks, sparsity, '-o')
plt.xlabel('scaled sqrt(K)', fontsize=20)
plt.ylabel('sparsity', fontsize=20)

plt.figure()
plt.plot(Ks, speeds, '-o')
plt.xlabel('scaled sqrt(K)', fontsize=20)
plt.ylabel('net drift speed', fontsize=20)

plt.figure()
plt.plot(Ks, dims,'-o') #/Ns[:,None]**2,'-o')
plt.xlabel('scaled sqrt(K)', fontsize=20)
plt.ylabel('dimension', fontsize=20)

# %% corrs
for ii in range(0,len(Ks),2):
    plt.figure(1)
    plt.plot(x_corr[ii,:], label=str(Ks[ii])); plt.legend(); plt.ylabel('cross-corr', fontsize=20)
    plt.figure(2)
    plt.plot(a_corr[ii,:], label=str(Ks[ii])); plt.legend(); plt.ylabel('auto-corr', fontsize=20)
    
# %% make video
# gif_name = 'K2_scale10'
# data = re_xy[:,:,200:]*1
# # data = beta_t[:,:,100:]*1
# # data = shifted_images*1
# # data = I_xy*1

# # Function to create a frame with the iteration number in the title
# def create_frame(data, frame):
#     fig, ax = plt.subplots()
#     ax.imshow(data[:, :, frame], cmap='viridis')
#     ax.set_title(f'Iteration: {frame}')
#     plt.colorbar(ax.images[0], ax=ax)
#     plt.close(fig)  # Close the figure to avoid displaying it
#     return fig

# # Create and save frames as images
# frames = []
# for frame in range(data.shape[2]):
#     fig = create_frame(data, frame)
#     fig.canvas.draw()
#     # Convert the canvas to an image
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     frames.append(Image.fromarray(image))

# # Save frames as a GIF
# frames[0].save(gif_name+'.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)

# # Check if the GIF plays correctly
# from IPython.display import display, Image as IPImage
# display(IPImage(filename=gif_name+'.gif'))