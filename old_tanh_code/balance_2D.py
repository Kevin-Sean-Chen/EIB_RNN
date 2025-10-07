# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:14:44 2024

@author: kevin
"""

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

# %% comparison of 1/N vs. 1/sqrt(N) network
### weak connection on 1/N order, which potentially generates unbalanced 
### strong connection on 1/sqrt(N) order, with balanced conditions

# %% parameter settings
N = 41  # neurons
tau_e = 0.005*1.  # time constant ( 5ms in seconds )
sig_e = 0.1    # spatial kernel
tau_i = 0.015
sig_i = 0.2*1.    ### important parameters!!

tau_i, sig_i = 15*0.001*1., 0.14    ### important parameters!!
#### 5, 0.14  ### grid parameter
#### 15, 0.2  ### chaos parameter
#### 10, 0.11 ### waves/strips!!!
#### 8,  0.2  ### blinking
#### 15, 0.14 ### switching waves!!

sig_ei = 0.2
sig_ie = 0.1

amp = 0

# %% balanced condition
rescale = 1.

Wee = 1* 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = 1*  .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale* N/15
mu_i = .8*rescale* N/15
# mu_e = .2*rescale*1 *(N**2*sig_e**2*np.pi*1)**0.5 #*N/15 ### ...think about scaling of this!!! ### 
# mu_i = .05*rescale*1 *(N**2*sig_i**2*np.pi*1)**0.5 #*N/15

# Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
# Wei = -1.5*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# Wie = .9*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
# Wii = -1.6*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# mu_e = 1.*rescale*1
# mu_i = .6*rescale*1

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
kernel_size = 23 #37  # pick this for numerical convolution

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

# %% stimuli
### spatial pattern
sigma_xy = 0.1
temp_space = np.random.randn(N,N)
temp_k = g_kernel(sigma_xy, N)
pattern = spatial_convolution(temp_space, temp_k)
pattern = pattern/np.max(pattern)
# pattern = 1
### time series
stim = ((np.sin(time/np.pi*50)+1)/2)*stim_scal
# plt.plot(stim)

# %% dynamics

### neural dynamics
for tt in range(lt-1):
    ### stim
    # if tt>lt//2:
        # mu_e = 5
    
    ### modifying for 2D rate EI-RNN
    ge_conv_re = spatial_convolution(re_xy[:,:,tt], g_kernel(sig_e))
    gi_conv_ri = spatial_convolution(ri_xy[:,:,tt], g_kernel(sig_i))
    he_xy[:,:,tt+1] = he_xy[:,:,tt] + dt/tau_e*( -he_xy[:,:,tt] + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e)  + stim[tt]*pattern )
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

# %%
###############################################################################
# %% ideas for analysis:
    ### spectrum
    ### perturbation
    ### "computation":
        ### reliability for readout!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ### diversity for high-D
        ### noise tolerance???
    
    ### idea:
        ### scan through spectrum for balanced vs. unbalanced
        ### plot beta vs. MSE, with dots across parameters
        
# %% spectral analysis
def spectrum_analysis(time_series, dt=dt):
    fft_result = np.fft.fft(time_series)
    
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    
    # Get the corresponding frequencies
    frequencies = np.fft.fftfreq(len(time_series), d=dt)

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

###
plt.figure()
test,ff = group_spectrum(re_xy[:,:,offset:])
plt.loglog(ff,test)
# plt.plot(ff[2:],test[2:])
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power', fontsize=20)

# %% 'beta' parameter and rate distribution
plt.figure()
# plt.hist(re_xy.reshape(-1),100)
plt.hist(np.mean(re_xy[:,:,offset:],2).reshape(-1),50)
plt.xlabel(r'$<r>_t$', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.title(f'N= {N}', fontsize=20)

beta_t = measure_mu[:,:,:] / measure_mu_ex[:,:,:]  # beta dynamics
plt.figure()
plt.hist(beta_t.reshape(-1),1000)
plt.xlabel(r'$\beta_t$', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.title(f'N= {N}', fontsize=20)
plt.xlim([0,6])
print('median of beta: ', np.nanmedian(beta_t))  #nanmedian?
plt.axvline(x=np.nanmedian(beta_t), color='r', linestyle='--')

# %% spectral tests
cmap = plt.get_cmap('viridis')
def cut_beta(thre_beta):
    temp_beta = beta_t*0+.0
    temp_beta[beta_t<thre_beta] = re_xy[beta_t<thre_beta]
    test_beta,ff_beta = group_spectrum(temp_beta[:,:,offset:])
    plt.loglog(ff_beta,test_beta, '--', label=r'$\beta_{below}=$'+str(thre_beta))
    return 
thre_beta = .9
plt.figure()
test,ff = group_spectrum(re_xy[:,:,offset:])
plt.loglog(ff,test, label='full sepctrum')
cut_beta(.5)
cut_beta(.7)
cut_beta(.9)
# plt.loglog(ff_beta,test_beta)
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power', fontsize=20)
plt.legend(fontsize=13)

# %%
plt.figure()
plt.hist(beta_t.reshape(-1),100)
plt.xlabel(r'$\beta_t$', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.title(r'$\beta$ distribution and cutoff', fontsize=20)
plt.xlim([0,5])
x_locations = [-1, .5, .7, .9]
cols = ['blue','orange', 'green','red']
# Plot vertical lines at the specified x locations
for ii,x in enumerate(x_locations):
    plt.axvline(x=x, color=cols[ii], linestyle='--', linewidth=2)

# %% check space
thre_beta = 0#1.05 # 1.04
temp_beta = beta_t*0+.0
temp_beta[beta_t>thre_beta] = re_xy[beta_t>thre_beta]
plt.figure()
data4fft = temp_beta[:,:,50:]*1
_,_,lt = data4fft.shape
data_fft = np.fft.fftn(data4fft)
data_fft_shifted = np.fft.fftshift(data_fft)
magnitude_spectrum = np.abs(data_fft_shifted)
plt.imshow(np.log(magnitude_spectrum[:, :, int(lt/2)]), cmap='gray')
plt.title(r'spatial spectrum for $\beta_{above}=$'+f'{thre_beta}')

# %%
# spectral analysis
from scipy.fft import fftn, fftshift, fftfreq

# Compute 3D FFT
u = re_xy*1
u_demeaned = u*1 - np.mean(u, axis=(0, 1), keepdims=True)
U_hat = fftn(u_demeaned[:,:,100:], axes=(0, 1, 2))
U_hat_shifted = fftshift(U_hat)
power_spectrum = np.abs(U_hat_shifted)**2

# Parameters
nx, ny, nt = N*1, N*1, U_hat.shape[-1] #len(time)
# Lx, Ly, T = 1.0, 1.0, 1.0
Lx, Ly, T = 1.0, 1.0, 10.0
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)

# Get frequency axes
kx = fftshift(fftfreq(nx, d=Lx/nx)) * 2 * np.pi
ky = fftshift(fftfreq(ny, d=Ly/ny)) * 2 * np.pi
omega = fftshift(fftfreq(nt, d=T/nt)) * 2 * np.pi

# Extract central slice through ky=0 to show omega vs kx
mid_ky = ny // 2
spectrum_slice = power_spectrum[mid_ky, :, :]

# Plot dispersion relation
# KX, OMEGA = np.meshgrid(kx, omega, indexing='ij')
plt.figure(figsize=(8, 5))
plt.pcolormesh(kx, omega, np.log(spectrum_slice.T), cmap='viridis')
plt.xlabel('Wave number $k_x$')
plt.ylabel('Frequency $\\omega$')
plt.title('Dispersion Relation from 2D Wave Field')
plt.colorbar(label='Power')
plt.tight_layout()
plt.show()

# Find peak in power spectrum
# power_spectrum[N//2,N//2,:] = 0
max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
k_peak = np.sqrt(kx[max_idx[0]]**2 + ky[max_idx[1]]**2)
omega_peak = omega[max_idx[2]]

# Compute wave speed
c_estimated = omega_peak / k_peak if k_peak != 0 else 0
c_estimated

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fftn, fftshift, fftfreq

# # Parameters for spatiotemporal data
# X, Y, T_grid = np.meshgrid(x, y, t, indexing='ij')

# # Create a traveling wave: u(x, y, t) = sin(kx*x + ky*y - omega*t)
# kx_true, ky_true, omega_true = 2*np.pi/5, 2*np.pi/5, 2*np.pi/4
# u = np.sin(kx_true * X + ky_true * Y - omega_true * T_grid)

# U_hat = fftn(u, axes=(0, 1, 2))
# # U_hat = fftn(re_xy, axes=(0, 1, 2))
# U_hat_shifted = fftshift(U_hat)
# power_spectrum = np.abs(U_hat_shifted)**2

# # Frequency axes
# kx = fftshift(fftfreq(nx, d=Lx/nx)) * 2 * np.pi
# ky = fftshift(fftfreq(ny, d=Ly/ny)) * 2 * np.pi
# omega = fftshift(fftfreq(nt, d=T/nt)) * 2 * np.pi

# # Find peak in power spectrum
# max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
# k_peak = np.sqrt(kx[max_idx[0]]**2 + ky[max_idx[1]]**2)
# omega_peak = omega[max_idx[2]]

# # Compute wave speed
# c_estimated = omega_peak / k_peak if k_peak != 0 else 0

# # Plot a slice of power spectrum at fixed ky
# plt.figure(figsize=(8, 5))
# plt.imshow(np.log(power_spectrum[:, ny//2, :].T), extent=[kx[0], kx[-1], omega[0], omega[-1]],
#             aspect='auto', origin='lower', cmap='magma')
# plt.colorbar(label='Power')
# plt.xlabel('Wave number $k_x$')
# plt.ylabel('Frequency $\\omega$')
# plt.title('Spectral Slice at $k_y=0$')
# plt.tight_layout()
# plt.show()

# c_estimated


# %% activity
plt.figure()
thre_beta = .0
temp_beta = beta_t*0+np.nan
temp_beta[beta_t>thre_beta] = re_xy[beta_t>thre_beta]
plt.hist(np.nanmean(temp_beta,2).reshape(-1),50, label=r'$\beta_{above}=$'+f'{thre_beta}')
thre_beta = .9
temp_beta = beta_t*0+np.nan
temp_beta[beta_t>thre_beta] = re_xy[beta_t>thre_beta]
plt.hist(np.nanmean(temp_beta,2).reshape(-1),50, label=r'$\beta_{above}=$'+f'{thre_beta}')
thre_beta = .99
temp_beta = beta_t*0+np.nan
temp_beta[beta_t>thre_beta] = re_xy[beta_t>thre_beta]
plt.hist(np.nanmean(temp_beta,2).reshape(-1),50, label=r'$\beta_{above}=$'+f'{thre_beta}')
plt.yscale('log')
plt.legend(fontsize=11)
plt.xlabel(r'$<r_t>$', fontsize=20)
plt.ylabel('count', fontsize=20)

# %%
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
    cax2 = ax2.matshow(temp_beta[:, :, frame], cmap='gray')
    ax1.set_title(f"Iteration {frame+1} - Subplot 1")
    ax2.set_title(f"Iteration {frame+1} - Subplot 2")
    return cax1, cax2

# Create the animation
ani = FuncAnimation(fig, update, frames=data_r.shape[-1], blit=False)

plt.show()

# %% do RLS
###############################################################################
# plt.figure()
# plt.hist(beta_mf2.reshape(-1), bins=100, alpha=0.5, label='mean-field 1')
# plt.hist(beta_mf.reshape(-1), bins=500, alpha=0.5, label='mean-field 2')
# plt.hist(beta_ei.reshape(-1), bins=100, alpha=0.5, label='EI')
# plt.xlim([0,5])
# plt.legend(fontsize=20)
# plt.xlabel('beta', fontsize=20); plt.ylabel('counts', fontsize=20)
