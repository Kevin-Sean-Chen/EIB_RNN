# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:45:27 2024

@author: kevin
"""

import torch
from scipy.signal import convolve2d
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from scipy.signal import correlate

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% proposal: Exploring computation through spatiotemporal neural dynamics
###############################################################################
# Simple implementation of rate model from Noga Mosheiff, Bard Ermentrout, Chengcheng Huang 2023
# which is a rate-based model for ChengCheng et al. 2019 (with spiking population)
# The plan is to implement 2D chaotic neural model and explore its dynamics/computation
# then study the input-output computation of this model, as well as effects from random connections (is there a transition?)
# further exporations: 
#   1. possible optimization for tasks
#   2. effects of short-term synaptic plasiticity in space
#   3. fitting to data??

### Haim's suggestions: (here till Sunday and back on Thursday)
    # use ReLu, order one units, and check equations and response timescales!!
    # is it really balanced??
    
    # if yes, is it fluctuating around coherence? how does the dilution affect?
    # even if not, anaylize initial condition effects and find the attractor through spatiotemporal analysis
    # further think about encoding and working memory

###############################################################################
# %% network parameters
# N = 70  # neurons
# Wee = 80*1  # recurrent weights
# Wei = -160*1
# Wie = 80*1
# Wii = -150*1
# mu_e = 0.48*1  # offset
# mu_i = 0.32*1
# tau_e = 0.005  # time constant ( 5ms in seconds )
# sig_e = 0.1  # spatial kernel
# tau_i, sig_i = 10*0.001, 0.07
# ### A, traveling waves solution (τi = 8, σi = 0.1). 
# ### B, alternating bumps solution (τi = 9, σi = 0.06). 
# ### C, alternating stripes solution (τi = 9, σi = 0.1). 
# ### D, chaotic solution (τi = 12.8, σi = 0.096).

# %% test to simplify
N = 100  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  # spatial kernel
tau_i, sig_i = 15*0.001, 0.20    ### important parameters!!
#### 5, 0.14  ### grid parameter
#### 15, 0.2  ### chaos parameter
#### 10, 0.11 ### waves/strips!!!
#### 8,  0.2  ### blinking
##################################
### moving dots 5ms, 0.2
### little coherence 5ms, 0.1
### chaotic waves 10ms, 0.1
### drifting changing blobs 10ms, 0.2

rescale = 3. ##(N*sig_e*np.pi*1)**0.5 #1
# Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
# Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# Wie = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
# Wii = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# mu_e = .7*rescale #*(N*sig_i*np.pi*1)**0.5 *rescale  #1e-8#.001*1  # offset
# mu_i = .7*rescale #*(N*sig_i*np.pi*1)**0.5 *rescale  #1e-8#.001*1

Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale
mu_i = .8*rescale

# %%
# ### Haim's parameters
# sig_e = 0.1 #% Define as per your setup
# sigm_i = 0.15
# tau_e, tau_i = 0.005, 0.005 
# Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
# Wei = -4.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# Wie = 2.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
# Wii = -2*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# mu_e = 10.*rescale
# mu_i = 1*rescale

# %% network setup
### setting up space and time
dt = 0.001  # 1ms time steps
T = .5  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
space_vec = np.linspace(0,1,N)
kernel_size = 23 #37  # pick this for numerical convolution

### random initial conditions
re_xy[:,:,0] = np.random.rand(N,N)*.1
ri_xy[:,:,0] = np.random.rand(N,N)*.1
# re_xy[0,:,50] = np.ones(N)
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

# %% dynamics
def spatial_convolution(r,k):
    """
    2D spatial convolution given kernel k and neural field r
    """
    gr = sp.signal.convolve2d(r.squeeze(), k, mode='same',  boundary='wrap') #, fillvalue=0,
    # gr = sp.signal.convolve2d(r.squeeze(), k, mode='same', boundary='symm')
    ###########################################################################
    #### double check on setting boundary conditions!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ###########################################################################
    return gr

### neural dynamics
for tt in range(lt-1):
    ### stim
    # if tt>lt//2:
        # mu_e = 5
    ### Noga Mosheiff paper
    # ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    # gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    # re_xy[:,:,tt+1] = re_xy[:,:,tt] + dt/tau_e*( -re_xy[:,:,tt] + phi(Wee*ge_conv_re + Wei*gi_conv_ri + mu_e) )
    # ri_xy[:,:,tt+1] = ri_xy[:,:,tt] + dt/tau_i*( -ri_xy[:,:,tt] + phi(Wie*ge_conv_re + Wii*gi_conv_ri + mu_i) )
    
    ### modifying for 2D rate EI-RNN
    ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + dt/tau_i*( -hi_xy[:,:,tt] + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
    re_xy[:,:,tt+1] = phi(he_xy[:,:,tt+1])
    ri_xy[:,:,tt+1] = phi(hi_xy[:,:,tt+1])
    
    # if tt>100 and tt<110:
    #     re_xy[60:,60:,tt+1] = 1
    
    ### make E-I measurements
    measure_e[tt+1] = (Wee*ge_conv_re + mu_e)[20,20]
    measure_i[tt+1] = (Wei*gi_conv_ri)[20,20]
    
    
    ### mean measurements
    measure_mu[:,:,tt+1] = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    # measure_mu[:,:,tt+1] = (  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    measure_mu_ex[:,:,tt+1] = (Wee*ge_conv_re + mu_e)
    
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
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
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

# %%
# plt.figure()
# plt.plot(time[offset:]-.25, np.mean(np.mean(re_xy[:,:,offset:],0),0), label='sigma=.1')
# plt.plot(time[offset:]-.25, temp, label='sigma=.15')
# plt.legend(fontsize=20)
# plt.xlim([-0.04, 0.04])

# %% visualize
### for rate
data = re_xy[:,:,1:]*1
### for beta
# data = measure_mu / measure_mu_ex
fig, ax = plt.subplots()
cax = ax.matshow(data[:, :, 0], cmap='gray')
fig.colorbar(cax)

def update(frame):
    ax.clear()
    cax = ax.matshow(data[:, :, frame], cmap='gray')
    ax.set_title(f"Iteration {frame+1}")
    return cax,

# Create the animation
ani = FuncAnimation(fig, update, frames=data.shape[-1], blit=False)

plt.show()

# %% for beta comparison

data_r = re_xy*1  # excitatroy network
beta_t = measure_mu / measure_mu_ex  # beta dynamics

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Initial plots for the first frame
cax1 = ax1.matshow(data_r[:, :, 0], cmap='gray')
cax2 = ax2.matshow(beta_t[:, :, 0], cmap='gray')

# Add colorbars
fig.colorbar(cax1, ax=ax1)
fig.colorbar(cax2, ax=ax2)

def update(frame):
    ax1.clear()
    ax2.clear()
    cax1 = ax1.matshow(data_r[:, :, frame], cmap='gray')
    cax2 = ax2.matshow(beta_t[:, :, frame], cmap='gray')
    ax1.set_title(f"Iteration {frame+1} - Subplot 1")
    ax2.set_title(f"Iteration {frame+1} - Subplot 2")
    return cax1, cax2

# Create the animation
ani = FuncAnimation(fig, update, frames=data_r.shape[-1], blit=False)

plt.show()

# %% compute beta assuming stationary
mu_total = np.mean(measure_mu, 2)
mu_ex = np.mean(measure_mu_ex, 2)
beta_ns = np.abs(mu_total)/mu_ex

plt.figure()
plt.hist(beta_ns.reshape(-1),20)
plt.title(f'N= {N}', fontsize=20)
plt.xlabel('beta per neuron', fontsize=20)


# %% study beta in detail
plt.figure()
plt.hist(np.nanmean(beta_t,2).reshape(-1),100)
plt.xlabel(r'$<\beta>_t$', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.title(f'N= {N}', fontsize=20)

plt.figure()
plt.hist(beta_t.reshape(-1),100)
plt.xlabel(r'$\beta_t$', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.title(f'N= {N}', fontsize=20)
plt.xlim([0,6])

# %% make video
##Generate example data (random 10x10x100 tensor)
# gif_name = 'beta_t3s2'
# # data = re_xy[:,:,100:]*1
# data = beta_t[:,:,100:]*1
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

# %% functions
### write function for spatial x-corr
### write for MLE calculation
### power-spectrum calculation

def spectrum_analysis(time_series, dt=dt):
    fft_result = np.fft.fft(time_series)
    
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    
    # Get the corresponding frequencies
    frequencies = np.fft.fftfreq(len(time_series), d=dt)
    
    # Plot the power spectrum
    # plt.figure(figsize=(10, 6))
    # plt.plot(frequencies[:len(frequencies)//2], power_spectrum[:len(frequencies)//2])
    # plt.title('Power Spectrum', fontsize=20)
    # plt.xlabel('Frequency (Hz)', fontsize=20)
    # plt.ylabel('Power', fontsize=20)
    return power_spectrum[:len(frequencies)//2], frequencies[:len(frequencies)//2]
    
def group_spectrum(data, dt=dt):
    N,_,T = data.shape
    pp,ff = spectrum_analysis(data[0,0,:])
    spec_all = np.zeros((N,N, len(ff)))  
    for ii in range(N):
        for jj in range(N):
            temp = data[ii,jj,:].squeeze()
            spec_all[ii,jj,:],_ = spectrum_analysis(temp)
    return np.mean(np.mean(spec_all,0),0), ff
            

plt.figure()
test,ff = group_spectrum(re_xy[:,:,50:])
# plt.loglog(ff,test)
plt.plot(ff[2:],test[2:])
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power', fontsize=20)
# plt.xlim([1,200])

# %% spatial
plt.figure()
data4fft = re_xy[:,:,50:]*1
_,_,lt = data4fft.shape
data_fft = np.fft.fftn(data4fft)
data_fft_shifted = np.fft.fftshift(data_fft)
magnitude_spectrum = np.abs(data_fft_shifted)
plt.imshow(np.log(magnitude_spectrum[:, :, lt//2]), cmap='gray')

# %%
# def spatial_cross_correlation(data, window):
#     N, _, T = data.shape
#     r = data*1
    
#     size_bound = N-window*2
#     temp_d = np.zeros((size_bound, size_bound, T))
#     temp_c = np.zeros((size_bound, size_bound, T))
#     for tt in range(T):
#         for ii in range(window, N-window):
#             for jj in range(window, N-window):
#                 remp_d[ii-window, jj-window] = r[ii,jj,tt] - np.mean(r[ii,jj,:],2)
#                 remp_c[ii-window, jj-window] = r[ii,jj,tt] - np.mean(r[ii,jj,:],2)
            
#     var_xyt = np.var(data[window:N-window, window:N-window, :])  # var(x,y,t)_full
    
#     # Reshape each nxn spatial snapshot into a 1D vector of length n*n
#     reshaped_matrix = matrix.reshape(n*n, t)
    
#     # Initialize a matrix to store the cross-correlation results
#     cross_corr_matrix = np.zeros((n*n, n*n))
    
#     # Compute cross-correlation for each pair of spatial points
#     for i in range(n*n):
#         for j in range(i, n*n):
#             # Correlate the time series of each pair of spatial points
#             cross_corr = correlate(reshaped_matrix[i], reshaped_matrix[j], mode='full')
#             # Store the peak (zero-lag) of the cross-correlation
#             cross_corr_matrix[i, j] = cross_corr[t-1]
#             cross_corr_matrix[j, i] = cross_corr[t-1]
    
#     return cross_corr_matrix

# %% show time series
time_points = np.array([90, 105, 160, 210, 260, 310])
titls = ['pre-stim', 'stim', 'post-stim 50steps', 'post-stim 100steps', 'post-stim 150 steps', 'post-stim 200steps']

fig, axs = plt.subplots(1, len(time_points), figsize=(20, 10))
for ss in range(len(time_points)):
    cax = axs[ss].imshow(re_xy[:,:,time_points[ss]])
    axs[ss].set_title(titls[ss], fontsize=15)
    axs[ss].set_xticks([])  # Remove x ticks
    axs[ss].set_yticks([])
    
    # if ss==len(time_points)-1:
        # fig.colorbar(cax, ax=axs[ss])
# cbar = fig.colorbar(cax, ax=axs[ss], location='right', shrink=0.6)

# Adjust the spacing of the subplots to make room for the colorbar
# fig.subplots_adjust(wspace=0.3)

# %%
# plt.figure()
# plt.plot([1,2,3,4], [np.nanstd(beta_50.reshape(-1)), np.nanstd(beta_100.reshape(-1)), \
#                        np.nanstd(beta_150.reshape(-1)), np.nanstd(beta_200.reshape(-1))], '-o')