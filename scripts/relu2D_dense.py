###########################################################################
# 2D EI network with dense implementation of disorder non-local connections
###########################################################################

import scipy.io as sio
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

### borrow functions from disorder script
import sys
from pathlib import Path
# add repo root to sys.path so `scripts` files can import each other
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from scripts.relu2D_disorder import gabor2d
from src.metrics import (
    avg_second_acf_peak,
    coherence_chi,
    coherence_metric,
    linear_dimention,
    second_acf_peak_latent,
)

# %% functional
# --- helper: build Gaussian 2D kernels identical to your current code ---
def _make_1d_kernel(N, sigma):
    # reproduce your periodic Gaussian "wrap" construction
    dx = 1.0 / N
    k = np.arange(-int(np.ceil(10 * sigma)), int(np.ceil(10 * sigma)) + 1)
    x = np.arange(-(N-1)//2, (N-1)//2 + 1)
    w = np.sum(
        dx * (2 * np.pi * sigma**2)**-0.5 *
        np.exp(-0.5 * (dx * (x[:, None] + k))**2 / sigma**2),
        axis=1
    )
    return w

def _make_2d_kernel(N, sigma):
    w = _make_1d_kernel(N, sigma)
    K2 = np.outer(w, w)  # separable 2D kernel
    return torch.tensor(K2, dtype=torch.float32)

# --- helper: turn a 2D circular convolution by `kernel_2d` into a dense (N^2 x N^2) matrix ---
def build_conv_matrix_from_kernel(kernel_2d: torch.Tensor, N: int, device=None):
    """
    Returns W (N^2 x N^2) such that vec_out = W @ vec_in
    where out = conv2d_circular(in, kernel_2d).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kH, kW = kernel_2d.shape
    pad = (kW // 2, kH // 2)

    # basis images: each is one-hot at a pixel -> yields one column of W
    eye = torch.eye(N*N, device=device)
    basis = eye.view(N*N, 1, N, N)                        # (B=N^2, 1, N, N)
    kernel = kernel_2d.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,kH,kW)

    out = F.conv2d(F.pad(basis, (pad[0], pad[0], pad[1], pad[1]), mode='circular'), kernel)
    # out: (N^2, 1, N, N). Each batch element is the conv result for one basis vector.
    # Flatten results and stack as COLUMNS to form the operator:
    W = out.view(N*N, N*N).T  # (N^2, N^2); columns are L(e_i), so W @ x = L(x)
    return W

# --- public helper: build E and I dense operators once ---
def build_dense_operators(N, sigma_e, sigma_i, device=None):
    """
    Build dense conv matrices W_e and W_i (both N^2 x N^2)
    from Gaussian widths sigma_e, sigma_i (scalars).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Ke = _make_2d_kernel(N, float(sigma_e))
    Ki = _make_2d_kernel(N, float(sigma_i))
    W_e = build_conv_matrix_from_kernel(Ke, N, device=device)
    W_i = build_conv_matrix_from_kernel(Ki, N, device=device)
    return W_e, W_i

# --- main: dense version of your stepper (matrix form like chi) ---
def relu2D_step_dense(
    re, ri, N, dt, npf, ntype, K, tau, u, J0,
    W_e, W_i,                         # (N^2 x N^2) dense (or sparse) operators
    chi=None,                         # optional (N^2 x N^2) structured term added on E-path
    r_and_mu=False, device=None
):
    """
    Same computation as your kernel/conv version, but using NxN large matrices.
    We assume:
      - mue uses conv over E activity plus optional 'chi @ re'
      - mui uses conv over I activity
      - u has shape (2,) and J0 is 2x2 like before
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # to tensors
    re = torch.as_tensor(re, dtype=torch.float32, device=device).view(N*N)
    ri = torch.as_tensor(ri, dtype=torch.float32, device=device).view(N*N)

    # parameters
    K = float(K)
    tau_e, tau_i = float(tau[0]), float(tau[1])
    u_e, u_i = float(u[0]), float(u[1])
    J00, J01 = float(J0[0,0]), float(J0[0,1])
    J10, J11 = float(J0[1,0]), float(J0[1,1])

    # ensure operators on device
    if not isinstance(W_e, torch.Tensor): W_e = torch.as_tensor(W_e, dtype=torch.float32, device=device)
    if not isinstance(W_i, torch.Tensor): W_i = torch.as_tensor(W_i, dtype=torch.float32, device=device)
    if chi is not None and not isinstance(chi, torch.Tensor):
        chi = torch.as_tensor(chi, dtype=torch.float32, device=device)

    # Time stepping
    for _ in range(npf):
        # linear drives
        conv_e = W_e @ re
        if chi is not None:
            conv_e = conv_e + (chi @ re) ### additive structure

        conv_i = W_i @ ri

        # mean inputs
        mue = (K**0.5) * (u_e + J00 * conv_e + J01 * conv_i)
        mui = (K**0.5) * (u_i + J10 * conv_e + J11 * conv_i)

        # Euler update with ReLU nonlinearity
        re = re + (dt / tau_e) * (-re + torch.relu(mue))
        ri = ri + (dt / tau_i) * (-ri + torch.relu(mui))

    # return numpy in original shapes
    re_np = re.view(N, N).detach().cpu().numpy()
    ri_np = ri.view(N, N).detach().cpu().numpy()

    if r_and_mu:
        mue_np = mue.view(N, N).detach().cpu().numpy()
        mui_np = mui.view(N, N).detach().cpu().numpy()
        return re_np, ri_np, mue_np, mui_np
    else:
        return re_np, ri_np

def relu2D_dense(N, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g, r_and_mu=False):
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
        chi = (mv @ nv.T + -mv @ mv.T*0) / N #### row balanced condition
        # chi = ((mv @ nv.T) / N, -(mv @ mv.T)/N)  #### if we hand over two components, the other for row balance

    ### make dense operators
    W_e, W_i = build_dense_operators(N, sigma[0], sigma[1])
    W_e, W_i = build_dense_operators(N, sigma[0], sigma[1])

    # build row-balance as a torch tensor on the same device/dtype as W_e
    device = W_e.device
    dtype = W_e.dtype
    mv_dev = mv.to(device=device, dtype=dtype)
    row_balance = torch.eye(N**2, device=device, dtype=dtype)*0 + (mv_dev @ mv_dev.T) / float(N)
    W_e = W_e - W_e * row_balance  # apply row balance on E path

    re = re0.copy()
    ri = ri0.copy()

    ### dynamics
    for n1 in range(1, int(np.floor(Nstep_init / npf)) + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 * npf / Nstep_init, 5))
        print(str_temp, end='', flush=True)

        if r_and_mu is True:
            re, ri, mue, mui = relu2D_step_dense(re, ri, N, dt, npf, ntype, K, tau, u, J0, 
                                                 W_e, W_i,
                                                 chi, r_and_mu=True)
        else:
            re, ri = relu2D_step_dense(re, ri, N, dt, npf, ntype, K, tau, u, J0, 
                                       W_e, W_i,
                                       chi)

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
            re, ri, mue, mui = relu2D_step_dense(re, ri, N, dt, npf, ntype, K, tau, u, J0, 
                                                 W_e, W_i,
                                                 chi, r_and_mu=r_and_mu)
        else:
            re, ri = relu2D_step_dense(re, ri, N, dt, npf, ntype, K, tau, u, J0, 
                                       W_e, W_i,
                                       chi)
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
    

# %% Main function to run the simulation and visualize results
if __name__ == "__main__":
    # Simulation parameters
    SAVE = False  # Whether to save the animation
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
    G = gabor2d(N, f=5, theta=np.deg2rad(30), gamma=0.1, phi=0.0, normalize=True)
    G = torch.tensor(G, dtype=mv.dtype, device=mv.device).reshape(-1, 1)
    mv = G*1  ### use gabor as mv
    G = gabor2d(N, f=5, theta=np.deg2rad(30), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5
    G = torch.tensor(G, dtype=mv.dtype, device=mv.device).reshape(-1, 1)
    nv = G*1  ### use gabor as nv
    g = (mv, nv*.5)
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
    re_all, ri_all = relu2D_dense(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g)

    ### Save results
    # Ensure the data directory exists
    # data_dir = os.path.join(os.path.dirname(__file__), 'data')
    # os.makedirs(data_dir, exist_ok=True)
    # filename = f"2Drelu_test_{datetime.now().strftime('%m-%d-%Y %H-%M-%S')}"
    # filepath = os.path.join(data_dir, f"{filename}.mat")
    # sio.savemat(filepath, {
    #     're_all': re_all,
    #     'ri_all': ri_all,
    #     'J0': J0,
    #     'K': K,
    #     'tau': tau,
    #     'u': u,
    #     'sigma': sigma
    # })

    space_coh = coherence_metric(re_all[:,:, Nstep//2])
    time_coh = coherence_chi(re_all.reshape(N*N, -1), mv.numpy())
    lin_dim = linear_dimention(re_all)
    print(f"Linear dimension of re_all: {lin_dim}")
    print(f"Space coherence at mid time point: {space_coh}")
    print(f"Time coherence: {time_coh}")

    # Plot simulation
    plt.figure(figsize=(8, 4))
    plt.plot(tt, re_all.mean(axis=(0, 1)), label='re_all mean')
    plt.plot(tt, ri_all.mean(axis=(0, 1)), label='ri_all mean')
    plt.xlabel('Time')
    plt.ylabel('Mean firing rate')
    plt.legend()
    plt.title('Simulation Results')
    plt.show()

    plt.figure()
    ### randomly select 10 neurons to plot
    neuron_indices = np.random.choice(L*L, size=10, replace=False)
    hist_re = re_all.reshape(L*L, -1)
    plt.plot(tt, hist_re[neuron_indices, :].T)
    plt.plot(tt, (mv.T @ hist_re / N).T, 'k--', label='mv@r/N', linewidth=2.5)
    # plt.plot(tt, (mv2.T @ hist_re / N).T, 'k-*', label='mv2@r/N', linewidth=2.5)
    plt.xlabel('Time')
    plt.ylabel('Firing rate of selected neurons')
    plt.title('Firing Rates of Selected Neurons')
    plt.legend()
    plt.show()

    ### show 2nd acf peak
    avg_peak, peaks = avg_second_acf_peak(re_all, p=100)
    plt.figure()
    plt.hist(peaks, bins=20, edgecolor='black')
    plt.axvline(avg_peak, color='r', linestyle='dashed', linewidth=1, label=f'Avg 2nd peak: {avg_peak:.3f}')
    plt.xlabel('2nd ACF Peak Value')
    plt.ylabel('Count')
    plt.title('Histogram of 2nd ACF Peak Values')
    plt.legend()
    plt.show()

    ### show the latent acf for mv projection
    hist_re = re_all.reshape(L*L, -1)
    mv_t = (mv.T @ hist_re / N).squeeze()  # shape
    peak_val, acf = second_acf_peak_latent(mv_t)
    plt.figure()
    plt.plot(acf, label='ACF of mv projection')
    plt.axhline(0, color='k', linestyle='dashed', linewidth=1)
    if not np.isnan(peak_val):
        plt.axhline(peak_val, color='r', linestyle='dashed', linewidth=1, label=f'2nd peak: {peak_val:.3f}')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('ACF of Latent Projection onto mv')
    plt.legend()
    plt.show()

    ### test with plotting re_heatmaps
    # reshape to (N*N, T)
    hist_re = re_all.reshape(-1, re_all.shape[2])  # shape (neurons, time)

    # sort rows so that the time-of-max (peak time) is ordered across rows.
    # tie-breaker: larger peak amplitude comes earlier for equal peak times
    peak_times = np.argmax(hist_re, axis=1)
    peak_vals = np.max(hist_re, axis=1)
    order = np.lexsort(( -peak_vals, peak_times ))  # primary: peak_times asc, secondary: peak_vals desc

    hist_re_sorted = hist_re[order]

    # optional: if you instead want to sort by peak amplitude (descending), uncomment:
    # order = np.argsort(-peak_vals)
    # hist_re_sorted = hist_re[order]

    # visualize as heatmap (neurons on y, time on x)
    plt.figure(figsize=(6, 8))
    plt.imshow(hist_re_sorted, aspect='auto', cmap='viridis',
               extent=[tt[0], tt[-1], 0, hist_re_sorted.shape[0]])
    plt.xlabel('Time')
    plt.ylabel('Neuron (sorted by peak time)')
    plt.title('Sorted Neural Activity (rows sorted by peak time)')
    plt.colorbar(label='Activity (re)')
    plt.tight_layout()
    plt.show()

    
    # Plot three time points of re_all
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


    # Create animation for re_all
    fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
    im_anim = ax_anim.imshow(re_all[:, :, 0], aspect='auto', origin='lower',
                            extent=[xx[0], xx[-1], yy[0], yy[-1]])
    ax_anim.set_xlabel('x')
    ax_anim.set_ylabel('y')
    cbar = fig_anim.colorbar(im_anim, ax=ax_anim, fraction=0.046, pad=0.04)

    # Add time/iter label inside the image (upper left corner)
    iter_text = ax_anim.text(0.02, 0.95, '', color='white',
                            ha='left', va='top', transform=ax_anim.transAxes,
                            fontsize=10, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

    def update(frame):
        idx = frame % re_all.shape[2]
        im_anim.set_data(re_all[:, :, idx])
        iter_text.set_text(f't = {tt[idx]:.3f}\niter = {idx}')
        return [im_anim, iter_text]

    ani = animation.FuncAnimation(
        fig_anim, update, frames=range(re_all.shape[2]),
        interval=50, blit=True, repeat=True
    )

    plt.show()

    # --- Save animation using tif ---

    # Example: Create dummy 3D data
    if SAVE is True:
        data = re_all*1  # Use re_all as the data to visualize
        # data = I_xyt.cpu().numpy()  # Convert to numpy for visualization
        def create_frame(data, idx):
            fig, ax = plt.subplots()
            ax.imshow(data[:, :, idx], cmap='viridis')#, vmin=0, vmax=1)
            ax.axis('off')
            return fig

        # Build frames
        frames = []
        for idx in range(data.shape[2]):
            fig = create_frame(data, idx)
            fig.canvas.draw()
            # image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            ### Get RGBA buffer, convert to RGB
            rgba = np.asarray(fig.canvas.renderer.buffer_rgba())  # shape (H, W, 4)
            image = rgba[:, :, :3]  # drop alpha channel
            frames.append(Image.fromarray(image))
            plt.close(fig)

        # Save as GIF
        os.makedirs('video', exist_ok=True)
        filename = f"2Drelu_disorder_{datetime.now().strftime('%m-%d-%Y %H-%M-%S')}"
        output_path = os.path.join('video', filename + '.gif')
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        # Display in notebook (optional)
        from IPython.display import display, Image as IPImage
        display(IPImage(filename=output_path))
