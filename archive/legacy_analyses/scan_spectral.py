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

#############################################################################
# Analytic scan for matrix spectrum with gain sqrt(K) and disorder g
#############################################################################

# %% parameter setup
Ks = np.array([10, 10**2, 10**3, 10**4, 10**5])  ### random strength
gs = np.array([0, 0.5, 1.0, 1.5, 2.0])*1  ### disorder strength
phis = np.array([0., .2, .4, .6, .8])*np.pi ### frequency of disorder pattern
Ns = np.array([21, 31, 41, 51, 61])  ### scan size of the network
leading = 500
scans = np.zeros((len(Ks), len(gs), leading, 2))  # store (mean, std) of coherence metric

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

# %% test visualization for giant matrix
We, Wi = build_dense_operators(N, sigma[0], sigma[1])
chi = (mv @ nv.T + -mv @ mv.T*1) / N  ### rank-one disorder

### make block J matrices with ([[We, Wi],[Wi, We]])
# build a 2x2 block-diagonal matrix (no cross-coupling between blocks)
J_block = np.block([
    [We.numpy()/1 + chi.numpy(), Wi.numpy()/1],
    [We.numpy()/1 + chi.numpy(), Wi.numpy()/1]
])
J = (J_block) * np.sqrt(K) - np.eye(J_block.shape[0]) 

plt.figure(figsize=(8,8))
plt.imshow(J)
plt.colorbar()
plt.title('Giant Matrix J Visualization')
plt.show()

measure_K = []
measure_lamb = []
# %% scanning
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
        
        ### construct full matrix
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
        vals, vecs = eigs(J, k=leading, which='LM')
        ### store measurements
        scans[kk, gg, :, 0] = vals.real #coherence
        scans[kk, gg, :, 1] = vals.imag

        measure_K.append(K)
        measure_lamb.append(np.max(vals))

# %% plotting (remember to change labels accordingly!)
### make subplost for K along the rows, g along the columns
plt.figure(figsize=(15, 12))
for kk in range(len(Ks)):
    for gg in range(len(gs)):
        plt.subplot(len(Ks), len(gs), kk*len(gs)+gg+1)
        plt.scatter(scans[kk, gg, :, 0], scans[kk, gg, :, 1], color='blue')
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')
        plt.title(f'K={Ks[kk]}, g={gs[gg]}')
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(measure_K, [lamb.real for lamb in measure_lamb], color='red')
plt.xlabel('K')
plt.ylabel('Leading Eigenvalue Real Part')
plt.xscale('log'); plt.yscale('log')
plt.show()