import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from datetime import datetime
import os
import matplotlib.pyplot as plt

from relu2D_disorder import *
from relu2D_dense import relu2D_dense, build_dense_operators
from scipy.sparse.linalg import eigs

# %% debug gabor to check it is scale invariant
Ns = [21,31,41,51,61]
plt.figure()
for N in Ns:
    G = gabor2d(N, f=8., theta=np.deg2rad(30), gamma=0.5, phi=0.1, normalize=True)
    plt.subplot(2,3,Ns.index(N)+1)
    plt.imshow(G, cmap='gray')
    plt.title(f'N={N}')
plt.show()

# %% parameter setup
Ks = np.array([10, 10**2, 10**3, 10**4, 10**5])  ### random strength
gs = np.array([0, 0.5, 1.0, 1.5, 2.0])*1  ### disorder strength
phis = np.array([0., .2, .4, .6, .8])*np.pi ### frequency of disorder pattern
Ns = np.array([21, 31, 41, 51, 61])  ### scan size of the network
scans = np.zeros((len(Ks), len(gs)))  # store (mean, std) of coherence metric

L = 31
N = L
dt = 0.0001
Nstep_init = 1 * 10 ** 3
Nstep = 1 * 10 ** 3 *2
npf = 1
g0 = .5  ### disorder strength, if not scanned
freq = 5. #0.15  ### frequency for gabor
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

# record input alignment
vu_alignments = np.zeros((len(Ks), len(gs)))

for kk in range(len(Ks)):
    for gg in range(len(gs)):
        ### scan K strength
        K = Ks[kk] ##10
        ### scan phi phase difference
        phi = phis[gg]
        ### scan g disorder strength
        # g0 = gs[gg]
        ### scan network size
        # N = Ns[kk]; L=N
        # Initial state
        re0 = r0[0] + 0.05 * np.random.rand(L, L)
        ri0 = r0[1] + 0.08 * np.random.rand(L, L)

        ### make rank-one structure
        G = gabor2d(N, f=freq, theta=np.deg2rad(30), gamma=0.1, phi=0.0, normalize=True)
        G = torch.tensor(G, dtype=mv.dtype, device=mv.device).reshape(-1, 1)
        mv = G*1  ### use gabor as mv
        G = gabor2d(N, f=freq, theta=np.deg2rad(30), gamma=0.1, phi=phi, normalize=True) ### 0.5,1,1.5
        G = torch.tensor(G, dtype=mv.dtype, device=mv.device).reshape(-1, 1)
        nv = G*1  ### use gabor as nv
        g = (mv, nv*g0)
        # g = (mv, nv*g0, mv, mv)  #, G)  ### pass in as a tuple
        # mv_continuous = makes_fourier_m(L, fi)
        # mv = torch.sign(mv_continuous)
        # g = (mv, nv*gs[gg]) #
        # g = (mv, nv*gs[gg], mv2, nv2*gs[gg])  ### pass in as a tuple

        print(f"Running simulation for K={Ks[kk]}, g={gs[gg]}")
        
        # Run simulation
        ### conventional relu2D
        # re_all, ri_all, mue_all, mui_all = relu2D(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g, r_and_mu=True)
        ### dense version
        re_all, ri_all, mue_all, mui_all = relu2D_dense(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g, r_and_mu=True)
        
        ### Compute coherence metric at the midpoint of the simulation
        # coherence = coherence_metric(re_all[:, :, Nstep // 2])
        coherence = coherence_chi(re_all.reshape(N*N, -1), mv.numpy())
        # coherence = linear_dimention(mue_all)#(re_all)
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
        scans[kk, gg] = peak_val
        # scans[kk, gg] = avg_peak
        print(f"Coherence metric: {coherence}")

        ### record kappas
        hist_re = re_all.reshape(L*L, -1)
        kappai =  (mv.T @ hist_re / N).T
        kappas[kk, gg, :] = kappai.numpy().squeeze()

        ### quantify alignment between v and u
        We, Wi = build_dense_operators(N, sigma[0], sigma[1])
        chi = (mv @ nv.T + -mv @ mv.T*0) / N  ### rank-one disorder

        ### make block J matrices with ([[We, Wi],[Wi, We]])
        # build a 2x2 block-diagonal matrix (no cross-coupling between blocks)
        J_block = np.block([
            [We.numpy()/1 + chi.numpy(), Wi.numpy()/1],
            [We.numpy()/1 + chi.numpy(), Wi.numpy()/1]
        ])
        J = (J_block) * np.sqrt(K) - np.eye(J_block.shape[0])  ### scale disorder by sqrt(K/N)
        ### do spectral analysis of J
        vals, vecs = eigs(J, k=10, which='LM')
        leading_eigvec = vecs[:, np.argmax(vals.real)]
        u_vec = leading_eigvec[:N*N]
        vu_alignment = np.abs((nv.T @ u_vec).squeeze()) / (np.linalg.norm(nv.numpy()) * np.linalg.norm(u_vec))
        vu_alignments[kk, gg] = vu_alignment

# %% plotting (remember to change labels accordingly!)
plt.figure()
# scans rows correspond to Ks, columns to phis
Ks_log = np.log10(Ks)
im = plt.imshow(scans, origin='lower',
                extent=(phis[0], phis[-1], Ks_log[0], Ks_log[-1]),
                aspect='auto', cmap='viridis')
# show original K values as y-labels but positioned at their log10 locations
plt.yticks(Ks_log, [str(int(k)) for k in Ks])
# prevent the later plt.yticks(Ks) call from overwriting our log ticks
plt.yticks = lambda *args, **kwargs: None
cbar = plt.colorbar(im, label='metric')
plt.xlabel('Phase φ')
# show phi ticks as multiples of π
xticks = phis
plt.xticks(xticks, [f"{p/np.pi:.1f}π" for p in phis])
plt.ylabel('Strength K')
plt.yticks(Ks)
plt.title('Scan over K and φ')
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

# %% plot vu_alignments
plt.figure()
plt.plot(vu_alignments.reshape(-1), scans.reshape(-1), 'o' )
plt.xlabel('Alignment |v·u|/(||v||||u||)')
plt.ylabel('Metric (peak of acf)')
plt.show()