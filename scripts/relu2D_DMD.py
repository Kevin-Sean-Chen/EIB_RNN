###################################################
# 2D EI network with disorder non-local connections
# testin with DMD to analyze spatial-temporal patterns
###################################################

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
from numpy.fft import fft2, fftfreq

### import function from relu2D_asym
### import WM task for space
import sys
from pathlib import Path
# add repo root to sys.path so `scripts` files can import each other
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

# from scripts.WM_task import make_wm_trial
from scripts.relu2D_asym import relu2D_bias


# relu2D_step function to perform one step of the ReLU 2D simulation
# This function uses PyTorch for efficient computation, especially on GPU.
# %% functional

def dmd_spatial_modes(
    r,
    dt=1.0,
    dx=1.0,
    dy=1.0,
    rank=30,
    n_show=8,
    subtract_mean=True,
    lag=1
):
    """
    DMD analysis for r(x,y,t).

    Input
    -----
    r : array
        Shape (T, Nx, Ny) or (Nx, Ny, T).
    dt : float
        Time step.
    dx, dy : float
        Spatial grid spacing.
    rank : int
        DMD truncation rank.
    n_show : int
        Number of DMD modes to plot.
    subtract_mean : bool
        Subtract temporal mean at each pixel/neuron.

    Returns
    -------
    results : dict
        DMD eigenvalues, modes, frequencies, growth rates, dominant k.
    """

    r = np.asarray(r)

    # Convert to (T, Nx, Ny) if needed
    if r.shape[-1] > r.shape[0]:
        r = np.moveaxis(r, -1, 0)

    T, Nx, Ny = r.shape

    if subtract_mean:
        r0 = r - r.mean(axis=0, keepdims=True)
    else:
        r0 = r.copy()

    # -----------------------------
    # Build data matrices
    # -----------------------------
    X = r0.reshape(T, Nx * Ny).T
    X1 = X[:, :-lag]
    X2 = X[:, lag:]

    # -----------------------------
    # SVD truncation
    # -----------------------------
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)

    rank = min(rank, len(S))
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh.conj().T[:, :rank]

    # -----------------------------
    # DMD operator
    # -----------------------------
    A_tilde = U_r.conj().T @ X2 @ V_r @ np.diag(1 / S_r)

    eigvals, W = np.linalg.eig(A_tilde)

    Phi = X2 @ V_r @ np.diag(1 / S_r) @ W
    modes = Phi.T.reshape(rank, Nx, Ny)

    # Continuous-time eigenvalues
    mu = np.log(eigvals) / dt

    growth = np.real(mu)
    omega = np.imag(mu)

    # -----------------------------
    # Dominant spatial wavenumber
    # -----------------------------
    kx = 2 * np.pi * fftfreq(Nx, d=dx)
    ky = 2 * np.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Kmag = np.sqrt(KX**2 + KY**2)

    dominant_k = np.zeros(rank)

    # for j in range(rank):
    #     F = np.abs(fft2(modes[j])) ** 2
    #     idx = np.unravel_index(np.argmax(F), F.shape)
    #     dominant_k[j] = Kmag[idx]

    for j in range(rank):
        P = np.abs(fft2(modes[j]))**2

        dominant_k[j] = np.sum(Kmag * P) / np.sum(P)

    # Sort modes by oscillation frequency
    order = np.argsort(np.abs(omega))[::-1]

    eigvals = eigvals[order]
    modes = modes[order]
    growth = growth[order]
    omega = omega[order]
    dominant_k = dominant_k[order]

    # -----------------------------
    # Plot 1: DMD eigenvalues
    # -----------------------------
    plt.figure(figsize=(5, 5))

    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5)
    plt.scatter(np.real(eigvals), np.imag(eigvals), s=40)

    plt.xlabel(r"$\mathrm{Re}(\lambda)$")
    plt.ylabel(r"$\mathrm{Im}(\lambda)$")
    plt.title("DMD eigenvalues")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 2: DMD spatial modes
    # -----------------------------
    n_show = min(n_show, rank)

    plt.figure(figsize=(10, 2.5 * n_show))

    for i in range(n_show):
        plt.subplot(n_show, 2, 2 * i + 1)
        plt.imshow(np.real(modes[i]), origin="lower")
        plt.colorbar()
        plt.title(
            f"Mode {i}: real part, "
            + rf"$k={dominant_k[i]:.2f}$, $\omega={omega[i]:.2f}$"
        )

        plt.subplot(n_show, 2, 2 * i + 2)
        plt.imshow(np.angle(modes[i]), origin="lower", cmap="twilight")
        plt.colorbar()
        plt.title("phase")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 3: dispersion relation
    # -----------------------------
    plt.figure(figsize=(6, 5))

    sc = plt.scatter(
        dominant_k,
        np.abs(omega),
        c=growth,
        s=80,
        edgecolor="k",
    )

    plt.xlabel(r"dominant spatial wavenumber $k$")
    plt.ylabel(r"oscillation frequency $|\omega|$")
    plt.title("DMD dispersion relation")
    plt.colorbar(sc, label=r"growth rate $\sigma$")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 4: growth rate vs k
    # -----------------------------
    plt.figure(figsize=(6, 5))

    plt.scatter(dominant_k, growth, s=80, edgecolor="k")
    plt.axhline(0, color="k", linestyle="--", alpha=0.5)

    plt.xlabel(r"dominant spatial wavenumber $k$")
    plt.ylabel(r"growth rate $\sigma$")
    plt.title("DMD growth rate vs spatial scale")
    plt.tight_layout()
    plt.show()

    return {
        "eigvals": eigvals,
        "modes": modes,
        "growth": growth,
        "omega": omega,
        "dominant_k": dominant_k,
        "dt": dt,
        "dx": dx,
        "dy": dy,
    }

def relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3, chi, r_and_mu=False):
    if ntype != 'relu_gaussian':
        raise ValueError("Only 'relu_gaussian' ntype is supported in this version.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    re = torch.tensor(re, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape (1,1,N,N)
    ri = torch.tensor(ri, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    dx = 1 / N
    k = np.arange(-int(np.ceil(10 * np.max(sigma))), int(np.ceil(10 * np.max(sigma))) + 1)
    x = np.arange(-(N-1)//2, (N-1)//2 + 1)
    we = np.sum(dx * (2 * np.pi * sigma[0]**2)**-0.5 *
                np.exp(-0.5 * (dx * (x[:, None] + k))**2 / sigma[0]**2), axis=1)
    wi = np.sum(dx * (2 * np.pi * sigma[1]**2)**-0.5 *
                np.exp(-0.5 * (dx * (x[:, None] + k))**2 / sigma[1]**2), axis=1)
    we_kernel = torch.tensor(np.outer(we, we), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    wi_kernel = torch.tensor(np.outer(wi, wi), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    pad = (we_kernel.shape[-1] // 2, we_kernel.shape[-2] // 2)

    ### learning rule parameters
    tau_J = 0.02
    eta = 0.05
    # if len(chi) == 2:
    #     chi, chi_rowb = chi
    # else:
    #     chi_rowb = chi*0
    for _ in range(npf):

        ### if there is local learnig rule (use Oja's rule here)
        # re_flat = re.squeeze().reshape(-1)  # flatten to 1D tensor
        # chi += dt/tau_J * (eta * (torch.outer(re_flat, re_flat)) - chi)# - torch.outer(re_flat, re_flat) @ chi

        # periodic boundary padding
        reP = F.pad(re, (pad[0], pad[0], pad[1], pad[1]), mode='circular')
        riP = F.pad(ri, (pad[0], pad[0], pad[1], pad[1]), mode='circular')

        conv_re = F.conv2d(reP, we_kernel) + (chi @ re.flatten()).reshape(N,N) #- (chi_rowb @ re.flatten()).reshape(N,N)
        conv_ri = F.conv2d(riP, wi_kernel)

        mue = K**0.5 * (u[0] + J0[0, 0] * conv_re + J0[0, 1] * conv_ri)
        mui = K**0.5 * (u[1] + J0[1, 0] * conv_re + J0[1, 1] * conv_ri)

        re = re + (dt / tau[0]) * (-re + torch.relu(mue))
        ri = ri + (dt / tau[1]) * (-ri + torch.relu(mui))

    re = re.squeeze().cpu().numpy()
    ri = ri.squeeze().cpu().numpy()
    if r_and_mu is True:
        return re, ri, mue.squeeze().cpu().numpy(), mui.squeeze().cpu().numpy()
    else:
        return re, ri

def relu2D(N, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g, r_and_mu=False):
    if N % 2 != 1:
        raise ValueError('N must be an odd integer')

    print('initializing simulation... ', end='', flush=True)
    str_temp = ''

    # Initialization steps
    # chi = torch.randn(N**2, N**2) * (g / N) ### random connectivity matrix
    # d = 1
    # mv, nv = torch.randn(N**2, d), torch.randn(N**2, d) ### low-rank vectors
    # chi = (mv @ mv.T) / N* g  #### MM or MN ###

    ### if g is two vectors to form a rank-2 connectivity
    if len(g) != 4:
        mv, nv = g
        chi = (mv @ nv.T) / N  #### MN ###
    else:
        mv, nv, mv2, nv2 = g
        # chi = (mv @ nv.T + mv2 @ nv2.T) / N  #### MN ###
        chi = (mv @ nv.T + -mv @ mv.T) / N #### row balanced condition
        # chi = ((mv @ nv.T) / N, -(mv @ mv.T)/N)  #### if we hand over two components, the other for row balance

    re = re0.copy()
    ri = ri0.copy()
    for n1 in range(1, int(np.floor(Nstep_init / npf)) + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 * npf / Nstep_init, 5))
        print(str_temp, end='', flush=True)

        if r_and_mu is True:
            re, ri, mue, mui = relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3, chi, r_and_mu=True)
        else:
            re, ri = relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3, chi)

    print('\nrunning simulation... ', end='', flush=True)
    str_temp = ''

    n_record = int(np.floor(Nstep / npf))
    re_all = np.full((N, N, n_record), np.nan)
    ri_all = np.full((N, N, n_record), np.nan)
    mue_all = np.full((N, N, n_record), np.nan)
    mui_all = np.full((N, N, n_record), np.nan)

    for n1 in range(1, n_record + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 / n_record, 5))
        print(str_temp, end='', flush=True)

        if r_and_mu is True:
            re, ri, mue, mui = relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3, chi, r_and_mu=r_and_mu)
        else:
            re, ri = relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3, chi)

        re_all[:, :, n1 - 1] = re
        ri_all[:, :, n1 - 1] = ri
        if r_and_mu is True:
            mue_all[:, :, n1 - 1] = mue
            mui_all[:, :, n1 - 1] = mui

    print('\n')
    if r_and_mu is True:
        return re_all, ri_all, mue_all, mui_all
    else:
        return re_all, ri_all


def gabor2d(N, f=0.1, theta=0.0, sigma_x=None, sigma_y=None, gamma=1.0, phi=0.0,
            center=None, normalize=False):
    """
    Make an N x N real Gabor patch, with parameters expressed in resolution-invariant units.

    Scaling rules (to be invariant to N):
    - f : spatial frequency in cycles per image. (e.g. f=1 -> one cycle across image width)
          Internally converted to cycles-per-pixel by dividing by N: (f/N).
    - sigma_x, sigma_y : if <= 1 interpreted as fraction of N (e.g. 0.1 -> 10% of N).
                         if > 1 interpreted as pixels.
                         If None, defaults to 0.1 * N (10% of image).
    - center : (x0, y0). If provided values are in [0,1], they are interpreted as fractions
               of the image (0..1) relative to width/height; otherwise interpreted as pixel coords.
    - gamma : aspect ratio (sigma_x / sigma_y) as before.
    - theta, phi : radians, unchanged.

    Returns
    -------
    G : (N, N) ndarray
        Real-valued Gabor patch.
    """
    # center: allow normalized coords in [0,1]
    if center is None:
        x0 = y0 = (N - 1) / 2.0
    else:
        x0, y0 = center
        # if center values look like fractions, scale to pixel coords
        if 0.0 <= x0 <= 1.0:
            x0 = x0 * (N - 1)
        if 0.0 <= y0 <= 1.0:
            y0 = y0 * (N - 1)

    # sigma defaults and interpretation: fraction or pixels
    if sigma_x is None:
        sigma_x = 0.2 * N
    elif sigma_x <= 1.0:
        sigma_x = float(sigma_x) * N

    if sigma_y is None:
        sigma_y = sigma_x / gamma
    elif sigma_y <= 1.0:
        sigma_y = float(sigma_y) * N

    # frequency: interpret f as cycles per image (resolution invariant).
    # convert to cycles-per-pixel by dividing by N -> then multiply xr to get phase.
    # i.e., cos(2*pi * (f / N) * xr + phi) = cos(2*pi * f * xr / N + phi)
    f_cycles_per_image = float(f)
    # coordinate grid (x = columns, y = rows)
    y, x = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

    # rotate coordinates
    xr = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    yr = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

    gauss = np.exp(-0.5 * ((xr / sigma_x)**2 + (yr / sigma_y)**2))
    carrier = np.cos(2.0 * np.pi * f_cycles_per_image * xr / float(N) + phi)
    G = gauss * carrier

    if normalize:
        m = np.max(np.abs(G))
        if m > 0:
            G = G / m
    return G


# %% Main function to run the simulation and visualize results
if __name__ == "__main__":
    # Simulation parameters
    SAVE = False  # Whether to save the animation
    L = 31
    N = L
    dt = 0.0001
    Nstep_init = 1 * 10 ** 3
    Nstep = 2 * 10 ** 3
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

    # Create sine and cosine vectors for nv and nv2
    # Create spatial coordinates for the N x N grid
    # x_coords = torch.arange(N, dtype=torch.float32)
    # y_coords = torch.arange(N, dtype=torch.float32)
    # X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    # freq = 2 * torch.pi / N  # One full period across the grid
    # mv_continuous = torch.sin(freq * X.flatten()).unsqueeze(1)
    # mv2_continuous = torch.cos(freq * X.flatten()).unsqueeze(1)  # Use X for both to maintain orthogonality
    # # Binarize to +1/-1
    # mv = torch.sign(mv_continuous)
    # mv2 = torch.sign(mv2_continuous)
    # # Handle any zeros (though unlikely with sine/cosine)
    # mv[mv == 0] = 1
    # mv2[mv2 == 0] = 1
    #### now for nv and nv2
    x_coords = torch.arange(N, dtype=torch.float32)
    y_coords = torch.arange(N, dtype=torch.float32)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    # Flatten to match N^2 x 1 shape and create sine/cosine patterns
    freq = 5*2 * torch.pi / N  # One full period across the grid
    nv = torch.sin(freq * X.flatten()).unsqueeze(1)
    nv2 = torch.cos(freq * Y.flatten()).unsqueeze(1)

    print((mv @ nv.T).shape)

    ### test spatial patterns
    G = gabor2d(N, f=2, theta=np.deg2rad(30), gamma=0.1, phi=0.0, normalize=True)
    G = torch.tensor(G, dtype=mv.dtype, device=mv.device).reshape(-1, 1)
    mv = G*1  ### use gabor as mv
    G = gabor2d(N, f=2, theta=np.deg2rad(30), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5
    G = torch.tensor(G, dtype=mv.dtype, device=mv.device).reshape(-1, 1)
    nv = G*1  ### use gabor as nv
    g = (mv*0, nv*.0)
    # g = (mv, nv*.5, mv2, nv2*5.)#, G)  ### pass in as a tuple


    plt.figure()
    plt.imshow(G.reshape(N, N).cpu(), cmap='bwr')
    plt.colorbar()
    plt.title('Gabor Pattern used for nv')
    plt.show()

    # Network type
    ntype = 'relu_gaussian'

    # Network parameters
    J0 = np.array([[1, -4], [2, -2]])
    K = 10 ** 2
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

    # Run simulation
    re_all, ri_all, mue_all, mui_all = relu2D(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g, r_and_mu=True)

    # bias = 20*1/N #2.0
    # I_xyt = torch.zeros((N, N, Nstep))  ### no input
    # re_all, ri_all = relu2D_bias(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, I_xyt, re0, ri0, bias)



    ### DMD analysis
    res = dmd_spatial_modes(
                            mue_all, #mue_all, #re_all,
                            dt=1,
                            dx=1.0,
                            dy=1.0,
                            rank=40,
                            n_show=8,
                            lag=1, ##50
                        )
    plt.figure()
    plt.scatter(res["dominant_k"], np.abs(res["omega"]), c=res["growth"])
    plt.xlabel("k")
    plt.ylabel("|omega|")
    plt.colorbar(label="growth rate")
    plt.title("DMD dispersion relation")
    plt.show()


