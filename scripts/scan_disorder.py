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
fs = np.array([0.5, 1, 2, 4, 8])*1 ### frequency of disorder pattern
scans = np.zeros((len(Ks), len(gs)))  # store (mean, std) of coherence metric

L = 31
N = L
dt = 0.0001
Nstep_init = 1 * 10 ** 3
Nstep = 1 * 10 ** 3
npf = 1
g = .0  ### disorder strength
### test with low-rank structure
mv, nv = torch.randint(0, 2, (N**2, 1), dtype=torch.float32)*2-1, torch.randn(N**2, 1, dtype=torch.float32)
### test with low-rank structure
mv = torch.randint(0, 2, (N**2, 1), dtype=torch.float32) * 2 - 1

# Create orthogonal binary vector for mv2
# Use Gram-Schmidt to find orthogonal direction, then binarize
random_vec = torch.randn(N**2, 1, dtype=torch.float32)
# Remove component parallel to mv
orthogonal_component = random_vec - (mv.T @ random_vec) / (mv.T @ mv) * mv
# Normalize and binarize
mv2 = torch.sign(orthogonal_component)
# Ensure exactly +1/-1 (no zeros)
mv2[mv2 == 0] = 1
# Create 2 orthogonal n vectors (can be continuous)
random_n = torch.randn(N**2, 2, dtype=torch.float32)
Q_n, _ = torch.linalg.qr(random_n)
nv, nv2 = Q_n[:, 0:1], Q_n[:, 1:2]


def make_fourier_m(N, freq):
    x_coords = torch.arange(N, dtype=torch.float32)
    y_coords = torch.arange(N, dtype=torch.float32)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    fourier_m = torch.sin(freq * X.flatten()).unsqueeze(1)
    return fourier_m
# Create sine and cosine vectors for nv and nv2
# Create spatial coordinates for the N x N grid
x_coords = torch.arange(N, dtype=torch.float32)
y_coords = torch.arange(N, dtype=torch.float32)
X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
freq = 2 * torch.pi / N  # One full period across the grid
mv_continuous = torch.sin(freq * X.flatten()).unsqueeze(1)
mv2_continuous = torch.cos(freq * X.flatten()).unsqueeze(1)  # Use X for both to maintain orthogonality
# # Binarize to +1/-1
mv = torch.sign(mv_continuous)
mv2 = torch.sign(mv2_continuous)
# # Handle any zeros (though unlikely with sine/cosine)
# mv[mv == 0] = 1
# mv2[mv2 == 0] = 1
#### now for nv and nv2
# x_coords = torch.arange(N, dtype=torch.float32)
# y_coords = torch.arange(N, dtype=torch.float32)
# X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
# # Flatten to match N^2 x 1 shape and create sine/cosine patterns
# freq = 5*2 * torch.pi / N  # One full period across the grid
# nv = torch.sin(freq * X.flatten()).unsqueeze(1)
# nv2 = torch.cos(freq * Y.flatten()).unsqueeze(1)

print((mv @ nv.T).shape)

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


for kk in range(len(Ks)):
    for gg in range(len(gs)):
        K = 10 #Ks[kk]
        fi = fs[kk]*freq
        mv_continuous = make_fourier_m(L, fi)
        mv = torch.sign(mv_continuous)
        g = (mv, nv*gs[gg]) #, mv2, nv2*gs[gg])  ### pass in as a tuple
        print(f"Running simulation for K={Ks[kk]}, g={gs[gg]}")
        # Run simulation
        re_all, ri_all = relu2D(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g)
        # Compute coherence metric at the midpoint of the simulation
        # coherence = coherence_metric(re_all[:, :, Nstep // 2])
        # coherence = coherence_chi(re_all.reshape(N*N, -1), mv.numpy())
        coherence = linear_dimention(re_all)
        scans[kk, gg] = coherence
        print(f"Coherence metric: {coherence}")

# %% plotting
plt.figure()
plt.imshow(scans, origin='lower', extent=(gs[0], gs[-1], fs[0], fs[-1]), aspect='auto')
plt.colorbar(label='linear dimension') #('Coherence Metric')
plt.xlabel('Disorder Strength g')
plt.ylabel('Frequency of Disorder Pattern f') #('Random Strength K')
plt.show()