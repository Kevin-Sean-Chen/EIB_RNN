import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from datetime import datetime
import os
import matplotlib.pyplot as plt

from relu2D_disorder import *

# %% parameter setup
Ks = np.array([10, 10**2, 10**3, 10**4, 10**5])  ### random strength
gs = np.array([0, 0.5, 1.0, 1.5, 2.0])  ### disorder strength
phis = np.array([0., .2, .4, .6, .8])*np.pi ### frequency of disorder pattern
scans = np.zeros((len(Ks), len(gs)))  # store (mean, std) of coherence metric

L = 31
N = L
dt = 0.0001
Nstep_init = 1 * 10 ** 3
Nstep = 1 * 10 ** 3 *2
npf = 1
g0 = 5.  ### disorder strength, if not scanned
freq = 0.15  ### frequency for gabor
phi = 0.1 ### phase for gabor, if not scanned
K = 1e2  ### random strength, if not scanned
mv, nv = torch.randint(0, 2, (N**2, 1), dtype=torch.float32)*2-1, torch.randn(N**2, 1, dtype=torch.float32)  ### if not defined
# Network type
ntype = 'relu_gaussian'

# Network parameters
J0 = np.array([[1, -4], [2, -2]])
tau = np.array([.01, .01])
u = np.array([10, 0])
sigma = 0.05 * np.array([1, np.sqrt(2)])
J2 = np.array([[0.454, -0.825], [0.908, -0.412]])
J3 = np.array([[12, -23], [19, -9]])

# Initial state
r0 = -np.linalg.inv(J0) @ u
re0 = r0[0] + 0.05 * np.random.rand(L, L)
ri0 = r0[1] + 0.08 * np.random.rand(L, L)

# Time and space axes
tt = dt * npf * np.arange(1, Nstep // npf + 1)
xx = np.arange(1, L + 1) / L
yy = np.arange(1, L + 1) / L

# record kappas
kappas = np.zeros((len(Ks), len(gs), Nstep // npf))

for kk in range(len(Ks)):
    for gg in range(len(gs)):
        ### scan K strength
        K = Ks[kk] ##10
        ### scan phi phase difference
        # phi = phis[gg]
        ### scan g disorder strength
        g0 = gs[gg]
        G = gabor2d(N, f=freq, theta=np.deg2rad(30), gamma=0.5, phi=0.0, normalize=True)
        G = torch.tensor(G, dtype=mv.dtype, device=mv.device).reshape(-1, 1)
        mv = G*1  ### use gabor as mv
        G = gabor2d(N, f=freq, theta=np.deg2rad(30), gamma=0.5, phi=phi, normalize=True) ### 0.5,1,1.5
        G = torch.tensor(G, dtype=mv.dtype, device=mv.device).reshape(-1, 1)
        nv = G*1  ### use gabor as nv
        g = (mv, nv*g0)#, mv2, nv2*5., G)  ### pass in as a tuple
        # mv_continuous = makes_fourier_m(L, fi)
        # mv = torch.sign(mv_continuous)
        # g = (mv, nv*gs[gg]) #
        # g = (mv, nv*gs[gg], mv2, nv2*gs[gg])  ### pass in as a tuple
        print(f"Running simulation for K={Ks[kk]}, g={gs[gg]}")
        
        # Run simulation
        re_all, ri_all, mue_all, mui_all = relu2D(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g, r_and_mu=True)
        # Compute coherence metric at the midpoint of the simulation
        # coherence = coherence_metric(re_all[:, :, Nstep // 2])
        # coherence = coherence_chi(re_all.reshape(N*N, -1), mv.numpy())
        coherence = linear_dimention(mue_all)#(re_all)
        ### compute the fraction of mue_all that is large than zero
        fraction_large = (mue_all > 0).astype(float).mean()
        ### balance: |mue-mui|/mue
        balance = np.abs(mue_all - mui_all) / np.abs(mue_all)
        balance_avg = balance[~np.isnan(balance)].mean()
        ### 2nd peak of acf of kappa
        hist_re = re_all.reshape(L*L, -1)
        mv_t = (mv.T @ hist_re / N).squeeze()  # shape
        peak_val, acf = second_acf_peak_latent(mv_t)
        ### 2nd peak of acf of activity patterns
        avg_peak, peaks = avg_second_acf_peak(mue_all, p=100)
        
        ### store measurements
        # scans[kk, gg] = coherence
        # scans[kk, gg] = fraction_large #coherence
        # scans[kk, gg] = balance_avg
        # scans[kk, gg] = peak_val
        scans[kk, gg] = avg_peak
        print(f"Coherence metric: {coherence}")

        ### record kappas
        hist_re = re_all.reshape(L*L, -1)
        kappai =  (mv.T @ hist_re / N).T
        kappas[kk, gg, :] = kappai.numpy().squeeze()

# %% plotting
plt.figure()
plt.imshow(scans, origin='lower', extent=(phis[0]/np.pi, phis[-1]/np.pi, Ks[0], Ks[-1]), aspect='auto')
plt.colorbar(label='2nd acf peak')
plt.xlabel('structure strength') #('Phase (phi)')
plt.ylabel('Random Strength K')
# plt.yscale('log')  # Since K values span multiple orders of magnitude
plt.show()

# %% plot kappas with many tight subplots
fig, axs = plt.subplots(len(Ks), len(gs), figsize=(15, 10), sharex=True)#, sharey=True)
for kk in range(len(Ks)):
    for gg in range(len(gs)):
        axs[kk, gg].plot(tt, kappas[kk, gg, :])
        axs[kk, gg].set_title(f'K={Ks[kk]}, phi={phis[gg]}')
        if kk == len(Ks) - 1:
            axs[kk, gg].set_xlabel('Time')
        if gg == 0:
            axs[kk, gg].set_ylabel('Kappa')
plt.tight_layout()
plt.show()