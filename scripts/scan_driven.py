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
Nstep = 1 * 10 ** 3 //3
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
Ks = [0.1, 1, 10, 100, 1000]
corr_coeffs = []  # scalar Pearson correlation coefficients (one per K)
cross_corrs = []  ### lagged cross-corr arrays between COM and the signal (one array per K)
trackings = [] ### raw COM time series
I_xyt = make_2D_stim_moving_dot(N, Nstep, dot_size=0.1, drift_rate=.7, device=device)  ###0.05; 1.5 ## 0.11, 0.7
I_xyt = I_xyt*10

from scipy.signal import correlate

for kk in range(len(Ks)):
    ### neural dynamics
    K = Ks[kk]    
    print(f"Running simulation for K={K}...")
    re_all, ri_all = relu2D_driven(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, I_xyt, re0, ri0)
    re_all_reshaped = re_all.reshape(Nstep // npf, L * L)
    readout = re_all_reshaped @ w_readout
    corr = np.corrcoef(readout, drift_series.cpu().numpy())[0, 1]
    corr_coeffs.append(corr)

    ### analyae tracking with COM
    y_coords = torch.linspace(0, 1, N, device=device, dtype=I_xyt.dtype).view(1, N, 1)  # shape (1, N, 1)

    # COM_y for input stimulus
    num_input = (I_xyt * y_coords).sum(dim=(0, 1))          # shape (T,)
    den_input = I_xyt.sum(dim=(0, 1))                       # shape (T,)
    com_input = num_input / (den_input + 1e-8)              # shape (T,)

    # COM_y for response tensor
    re_all_t = torch.tensor(re_all, device=device, dtype=torch.float32)
    y_coords_re = y_coords.to(re_all_t.dtype)

    num_re = (re_all_t * y_coords_re).sum(dim=(0, 1))       # shape (T,)
    den_re = re_all_t.sum(dim=(0, 1))                       # shape (T,)
    com_re = num_re / (den_re + 1e-8)                       # shape (T,)

    ### normalize COM to be between 0 and 1 (optional, since we already used physical y coordinates)
    eps = 1e-8
    com_input = 2 * (com_input - com_input.min()) / (com_input.max() - com_input.min() + eps) - 1
    com_re = 2 * (com_re - com_re.min()) / (com_re.max() - com_re.min() + eps) - 1
    
    trackings.append(com_re.detach().cpu().numpy())

    ### plot the cross correlation of the two COM time series
        
    x = com_input.detach().cpu().numpy().astype(float)
    y = com_re.detach().cpu().numpy().astype(float)

    x0 = x - x.mean()
    y0 = y - y.mean()

    raw_corr = correlate(x0, y0, mode='full')
    lags = np.arange(-len(x) + 1, len(x))

    # number of overlapping points at each lag
    overlap = len(x) - np.abs(lags)

    # unbiased-by-overlap normalization
    corr_unbiased = raw_corr / overlap

    cross_corrs.append(corr_unbiased)

# %% plotting
plt.figure(figsize=(12, 6))
### plot responses labeled by K
for kk in range(len(Ks)):
    plt.plot(trackings[kk], label=f"K={Ks[kk]}")
plt.plot(com_input.detach().cpu().numpy(), label='Input COM', color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('COM of Network Response')
plt.title('COM of Network Response Over Time for Different K')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for kk in range(len(Ks)):
    plt.plot(lags,cross_corrs[kk], label=f"K={Ks[kk]}")
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.xlim([-20, 20])
plt.title('Cross-Correlation of COM Time Series for Different K')
plt.legend()
plt.show()