import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from datetime import datetime
import os
import matplotlib.pyplot as plt

from relu2D_asym import relu2D_bias
from relu2D_dense import relu2D_dense, build_dense_operators

from scripts.relu2D_disorder import coherence_metric, coherence_chi, \
                                    linear_dimention, avg_second_acf_peak, second_acf_peak_latent

# %% network parameters
L = 31
N = L
dt = 0.0001
Nstep_init = 1 * 10 ** 3
Nstep = 1 * 10 ** 3
npf = 2

I_xyt = torch.zeros((N, N, Nstep))  ### no input

### network parameters
ntype = 'relu_gaussian'
J0 = np.array([[1, -4], [2, -2]])
K = 10 ** 1
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

# %% test with SVD to extract wave spead
import numpy as np

def phase_corr_shift(a, b, eps=1e-12):
    """Return (dy, dx) that best shifts a -> b (integer-pixel, robust)."""
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fb * np.conj(Fa)
    R /= (np.abs(R) + eps)
    c = np.fft.ifft2(R)
    c = np.abs(c)

    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    ny, nx = a.shape

    # wrap to signed shift
    dy = iy if iy < ny//2 else iy - ny
    dx = ix if ix < nx//2 else ix - nx
    return dy, dx

def estimate_speed_phasecorr(U, x, y, t):
    """
    U: array (Nt, Ny, Nx) real or complex (use np.abs(U) if complex wavefield)
    x,y: 1D coordinate arrays
    t: 1D time array
    """
    Nt, Ny, Nx = U.shape
    dx_phys = float(np.mean(np.diff(x)))
    dy_phys = float(np.mean(np.diff(y)))

    vxs, vys, ts = [], [], []
    for k in range(Nt-1):
        a = np.abs(U[k])      # or U[k].real / U[k] depending on what you track
        b = np.abs(U[k+1])
        dy, dx = phase_corr_shift(a, b)
        dt = t[k+1] - t[k]
        vxs.append((dx * dx_phys) / dt)
        vys.append((dy * dy_phys) / dt)
        ts.append(0.5*(t[k]+t[k+1]))

    vxs = np.array(vxs); vys = np.array(vys)
    speed = np.sqrt(vxs**2 + vys**2)
    return np.array(ts), vxs, vys, speed

# %% scanning parameters
plt_logic = False #True
biass = np.linspace(0.0, 2.0, 15)/N
biass = np.array([0, 4, 8, 12, 16, 20])/N
# biass = np.array([0, 8, 16])/N
samps = 20
acfs_allb = []
wv_measure = np.zeros((len(biass), samps))
spd_measure = np.zeros(len(biass))
dim_measure = np.zeros(len(biass))

for bb in range(len(biass)):
    bias = biass[bb]
    re_all, ri_all = relu2D_bias(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, I_xyt, re0, ri0, bias)
    ### randomly sample samp from 1:N*N
    samp = np.random.randint(1, L * L, samps)
    ### sample cells
    hist_re = re_all.reshape(L*L, -1)[samp, :]
    acfs = []
    for ss in range(samps):
        peak_val, acf = second_acf_peak_latent(hist_re[ss, :])
        acfs.append(acf)
        wv_measure[bb, ss] = peak_val
    acfs_allb.append(np.array(acfs))

    ### test wave speed calculation
    _, vxs, vys, speed = estimate_speed_phasecorr(re_all, xx, yy, tt)
    spd_measure[bb] = np.mean(speed)

    ### linear dimension
    dim_measure[bb] = linear_dimention(re_all)

    ### PLOT logic
    if plt_logic==True:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        time_indices = [10, len(tt) // 2, -1]
        titles = ['Beginning', 'Middle', 'End']

        for i, idx in enumerate(time_indices):
            im = axs[i].imshow(re_all[:, :, idx], aspect='auto', origin='lower', extent=[xx[0], xx[-1], yy[0], yy[-1]])
            axs[i].set_title(f"{titles[i]} (t={tt[idx]:.3f})")
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

        plt.suptitle('re_all at Three Time Points')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# %% plotting
plt.figure()
plt.plot(biass, wv_measure)
plt.xlabel('Bias')
plt.ylabel('Max ACF Value')
plt.title('Bias vs. Max ACF Value')
plt.show()

plt.figure()
plt.plot(biass, spd_measure)
plt.xlabel('Bias')
plt.ylabel('Wave Speed')
plt.title('Bias vs. Wave Speed')
plt.show()

plt.figure()
plt.plot(biass, dim_measure)
plt.xlabel('Bias')
plt.ylabel('Linear Dimension')
plt.title('Bias vs. Linear Dimension')
plt.show()

# %% plot ACF for all bias values
fig, axs = plt.subplots(len(biass), 1, figsize=(10, 3*len(biass)))
for bb in range(len(biass)):
    for ss in range(samps):
        axs[bb].plot(tt, acfs_allb[bb][ss])
    axs[bb].set_xlabel('Time')
    axs[bb].set_ylabel('ACF')
    axs[bb].set_title(f'ACF for bias = {biass[bb]:.4f}')
plt.tight_layout()
plt.show()
