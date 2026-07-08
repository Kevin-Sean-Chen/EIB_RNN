from ctypes.wintypes import RGB
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from datetime import datetime
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

### borrow functions from disorder script
import sys
from pathlib import Path
# add repo root to sys.path so `scripts` files can import each other
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from scripts.relu2D_driven import make_2D_stim_with_drift, relu2D_driven, make_2D_stim_moving_dot
from scripts.relu2D_disorder import gabor2d

# %% setup network
L = 31
time_f, space_f, drift_rate, device = 0.1*1, 2.5*2, 0.0, 'cpu'
N = L
dt = 0.0001
Nstep_init = 1 * 10 ** 3 //3
Nstep = 1 * 10 ** 3 //3  ### 150 for speed-accuracy comparison; 1*10**3//3 for visualizing tracking dynamics
npf = 1

### network parameters
ntype = 'relu_gaussian'
J0 = np.array([[1, -4], [2, -2]])
K = 10 ** 2
tau = np.array([.01, .01])
u = np.array([10, 0.0])
sigma = 0.05 * np.array([1, np.sqrt(2)])

# Initial state
r0 = -np.linalg.inv(J0) @ u
re0 = r0[0] + 0.05 * np.random.rand(L, L)
ri0 = r0[1] + 0.08 * np.random.rand(L, L)

# Time and space axes
tt = dt * npf * np.arange(1, Nstep // npf + 1)
xx = np.arange(1, L + 1) / L
yy = np.arange(1, L + 1) / L


# %% setup stimulus
I_xyt, drift_series = make_2D_stim_with_drift(N, Nstep, time_f, space_f, drift_rate, device=device)

I_xyt = I_xyt*1#*0

### visualizations
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(I_xyt[:, :, 0].cpu().numpy(), cmap='gray')
plt.subplot(1, 3, 2)
plt.plot(I_xyt[0, 0, :].cpu().numpy())
plt.subplot(1, 3, 3)
plt.plot(drift_series.cpu().numpy())
plt.show()

# %% run simulation trials
### trial settings
n_repeats = 1#10
w_readout = np.random.randn(L**2)
G = gabor2d(N, f=5*.5, theta=np.deg2rad(30), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5
G = G.reshape(-1)
# w_readout = G*1  ### use gabor as nv
readouts = np.zeros((n_repeats, Nstep // npf))

# Run simulation
for rr in range(n_repeats):
    print(f"Running trial {rr+1}/{n_repeats}...")
    ### initialize and run
    re0 = r0[0] + 0.05 * np.random.rand(L, L)
    ri0 = r0[1] + 0.08 * np.random.rand(L, L)
    re_all, ri_all = relu2D_driven(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, I_xyt, re0, ri0)
    # reshape re_all to (Nstep//npf, L*L)
    re_all_reshaped = re_all.reshape(Nstep // npf, L * L)
    readouts[rr, :] = re_all_reshaped @ w_readout

# %% plot results
plt.figure(figsize=(10, 6))
for rr in range(n_repeats):
    plt.plot(tt, readouts[rr, :], alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Readout Activity')
plt.title('Network Readout Activity Over Time Across Trials')
plt.show()

# %%
######################################################################################################################
######################################################################################################################
# %% testing object tracking vs. K strength
Ks = [0.1, 1, 10, 100, 1000, 10000]
rep = 5
corr_coeffs = np.zeros((len(Ks), rep))  # scalar Pearson correlation coefficients per (K, rep)
cross_corrs_first = []  # cross-corr for the first rep only (for plotting)
trackings_first = []  # COM traces for the first rep only (for plotting)
I_xyt = make_2D_stim_moving_dot(N, Nstep, dot_size=0.2, drift_rate=0.7, device=device)  ###0.05; 1.5 ## 0.11, 0.7
I_xyt = I_xyt*10

from scipy.signal import correlate

### for speed-accuaracy comparison
# reps = 10
# SAs = np.zeros((len(Ks), 2, reps)) ### speed and accuracy arrays

# COM_y for input stimulus (same for all K and reps)
y_coords = torch.linspace(0, 1, N, device=device, dtype=I_xyt.dtype).view(1, N, 1)  # shape (1, N, 1)
num_input = (I_xyt * y_coords).sum(dim=(0, 1))          # shape (T,)
den_input = I_xyt.sum(dim=(0, 1))                       # shape (T,)
com_input = num_input / (den_input + 1e-8)              # shape (T,)

eps = 1e-8
com_input = 2 * (com_input - com_input.min()) / (com_input.max() - com_input.min() + eps) - 1
x = com_input.detach().cpu().numpy().astype(float)
lags = np.arange(-len(x) + 1, len(x))
lags_subset = np.where((lags >= -20) & (lags <= 20))[0]

peak_lags_all = np.zeros((len(Ks), rep))
peak_heights_all = np.zeros((len(Ks), rep))

for kk in range(len(Ks)):
    K = Ks[kk]
    for rr in range(rep):
        print(f"Running simulation for K={K}, rep {rr+1}/{rep}...")
        re0 = r0[0] + 0.05 * np.random.rand(L, L)
        ri0 = r0[1] + 0.08 * np.random.rand(L, L)
        re_all, ri_all = relu2D_driven(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, I_xyt, re0, ri0)
        re_all_reshaped = re_all.reshape(Nstep // npf, L * L)
        readout = re_all_reshaped @ w_readout
        corr_coeffs[kk, rr] = np.corrcoef(readout, drift_series.cpu().numpy())[0, 1]

        # COM_y for response tensor
        re_all_t = torch.tensor(re_all, device=device, dtype=torch.float32)
        y_coords_re = y_coords.to(re_all_t.dtype)
        num_re = (re_all_t * y_coords_re).sum(dim=(0, 1))       # shape (T,)
        den_re = re_all_t.sum(dim=(0, 1))                       # shape (T,)
        com_re = num_re / (den_re + 1e-8)                       # shape (T,)
        com_re = 2 * (com_re - com_re.min()) / (com_re.max() - com_re.min() + eps) - 1

        y = com_re.detach().cpu().numpy().astype(float)
        x0 = x - x.mean()
        y0 = y - y.mean()
        raw_corr = correlate(x0, y0, mode='full')
        overlap = len(x) - np.abs(lags)
        corr_unbiased = raw_corr / overlap

        if rr == 0:
            trackings_first.append(y)
            cross_corrs_first.append(corr_unbiased)

        com_re_window = corr_unbiased[lags_subset]
        peak_idx = np.argmax(com_re_window)
        peak_lags_all[kk, rr] = lags[lags_subset][peak_idx]
        peak_heights_all[kk, rr] = com_re_window[peak_idx]

    # for kk in range(len(Ks)):
    #     ### find lag values between -20 and 20
    #     # get integer indices into `lags` for the window [-20, 20]
    #     lags_subset = np.where((lags >= -20) & (lags <= 20))[0]
    #     com_re = cross_corrs[kk][lags_subset]
    #     peak_idx = np.argmax(com_re)
    #     peak_value = com_re[peak_idx]
    #     SAs[kk, 0, rr] = lags[lags_subset][peak_idx]  ### speed: time to peak
    #     SAs[kk, 1, rr] = peak_value  ### accuracy: height of the peak

# %% plotting
plt.figure(figsize=(12, 6))
### plot responses labeled by K
for kk in range(len(Ks)):
    plt.plot(trackings_first[kk], label=f"K={Ks[kk]} (rep 1)")
plt.plot(com_input.detach().cpu().numpy(), label='Input COM', color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('COM of Network Response')
plt.title('COM of Network Response Over Time for Different K (rep 1)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for kk in range(len(Ks)):
    plt.plot(lags, cross_corrs_first[kk], label=f"K={Ks[kk]} (rep 1)")
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.xlim([-20, 20])
plt.title('Cross-Correlation of COM Time Series for Different K (rep 1)')
plt.legend()
plt.show()

# # %% plot speed and accuracy
# # print(SAs)
# plt.figure(figsize=(8, 6))
# for kk in range(len(Ks)):
#     plt.scatter(SAs[kk, 0, :], SAs[kk, 1, :], label=f"K={Ks[kk]}")
# plt.xlabel('Time to Peak (s)')
# plt.ylabel('Peak COM Value')
# plt.title('Speed and Accuracy of COM Tracking for Different K')
# plt.legend()
# plt.show()

# %% plot peak location and height vs K
peak_lags_mean = peak_lags_all.mean(axis=1)
peak_lags_std = peak_lags_all.std(axis=1)
peak_heights_mean = peak_heights_all.mean(axis=1)
peak_heights_std = peak_heights_all.std(axis=1)
    
plt.figure(figsize=(12, 6))
plt.errorbar(Ks, peak_lags_mean, yerr=peak_lags_std, fmt='o-', capsize=4, label='Time to Peak (mean ± std)')
plt.xlabel('Reservoir Strength K')
plt.ylabel('Lag')
plt.xscale('log')
plt.title('Peak Location vs K (mean ± std over reps)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.errorbar(Ks, peak_heights_mean, yerr=peak_heights_std, fmt='o-', capsize=4, label='Peak Height (mean ± std)')
plt.xlabel('Reservoir Strength K')
plt.ylabel('Cross-Correlation')
plt.xscale('log')
plt.title('Peak Height vs K (mean ± std over reps)')
plt.legend()
plt.show()