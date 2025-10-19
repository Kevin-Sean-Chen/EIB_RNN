###################################################
# 2D EI network with disorder non-local connections
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

# relu2D_step function to perform one step of the ReLU 2D simulation
# This function uses PyTorch for efficient computation, especially on GPU.
# %% functional
def relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3, chi):
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
    for _ in range(npf):

        ### if there is local learnig rule (use Oja's rule here)
        # re_flat = re.squeeze().reshape(-1)  # flatten to 1D tensor
        # chi += dt/tau_J * (eta * (torch.outer(re_flat, re_flat)) - chi)# - torch.outer(re_flat, re_flat) @ chi

        # periodic boundary padding
        reP = F.pad(re, (pad[0], pad[0], pad[1], pad[1]), mode='circular')
        riP = F.pad(ri, (pad[0], pad[0], pad[1], pad[1]), mode='circular')

        conv_re = F.conv2d(reP, we_kernel) + (chi @ re.flatten()).reshape(N,N)
        conv_ri = F.conv2d(riP, wi_kernel)

        mue = K**0.5 * (u[0] + J0[0, 0] * conv_re + J0[0, 1] * conv_ri)
        mui = K**0.5 * (u[1] + J0[1, 0] * conv_re + J0[1, 1] * conv_ri)

        re = re + (dt / tau[0]) * (-re + torch.relu(mue))
        ri = ri + (dt / tau[1]) * (-ri + torch.relu(mui))

    re = re.squeeze().cpu().numpy()
    ri = ri.squeeze().cpu().numpy()
    return re, ri

def relu2D(N, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g):
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
        chi = (mv @ nv.T + mv2 @ nv2.T) / N  #### MN ###

    re = re0.copy()
    ri = ri0.copy()
    for n1 in range(1, int(np.floor(Nstep_init / npf)) + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 * npf / Nstep_init, 5))
        print(str_temp, end='', flush=True)

        re, ri = relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3, chi)

    print('\nrunning simulation... ', end='', flush=True)
    str_temp = ''

    n_record = int(np.floor(Nstep / npf))
    re_all = np.full((N, N, n_record), np.nan)
    ri_all = np.full((N, N, n_record), np.nan)

    for n1 in range(1, n_record + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 / n_record, 5))
        print(str_temp, end='', flush=True)

        re, ri = relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3, chi)

        re_all[:, :, n1 - 1] = re
        ri_all[:, :, n1 - 1] = ri

    print('\n')
    return re_all, ri_all

def coherence_metric(img, sigma=1.0):
    # gradients
    Ix = sobel(img, axis=1)
    Iy = sobel(img, axis=0)
    # structure tensor components (with Gaussian smoothing)
    Jxx = gaussian_filter(Ix*Ix, sigma)
    Jyy = gaussian_filter(Iy*Iy, sigma)
    Jxy = gaussian_filter(Ix*Iy, sigma)
    # eigenvalues of 2x2 tensor
    tmp = np.sqrt((Jxx - Jyy)**2 + 4*Jxy**2)
    l1 = 0.5*(Jxx + Jyy + tmp)
    l2 = 0.5*(Jxx + Jyy - tmp)
    # coherence index per pixel
    C = (l1 - l2) / (l1 + l2 + 1e-12)
    # global coherence
    return np.nanmean(C)

def coherence_chi(H, mv, eps=1e-12):
    H = np.asarray(H, dtype=float)
    m_t = mv.T @ H / mv.shape[0]  # project onto mv, shape (T,)
    # m_t = H.mean(axis=0) 
    num = np.mean(m_t**2)                 # <(mean_i h)^2>_t
    # per-channel power then average over channels
    den = np.mean(np.mean(H**2, axis=1))  # (1/N) * sum_i <h_i^2>_t
    return float(np.sqrt(num / (den + eps)))

def linear_dimention(img, variance_threshold=0.9):
    from sklearn.decomposition import PCA
    img_flat = img.reshape(img.shape[0]*img.shape[1], -1)  # flatten spatial dimensions
    pca = PCA()
    pca.fit(img_flat)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    dim = np.searchsorted(cumulative_variance, variance_threshold) + 1  # number of components to explain 90% variance
    return dim

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
    g = (mv, nv*5., mv2, nv2*5.)  ### pass in as a tuple

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
    re_all, ri_all = relu2D(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0, g)

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
    plt.plot(tt, (mv2.T @ hist_re / N).T, 'k-*', label='mv2@r/N', linewidth=2.5)
    plt.xlabel('Time')
    plt.ylabel('Firing rate of selected neurons')
    plt.title('Firing Rates of Selected Neurons')
    plt.legend()
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


    ### metic for now...
    sc = np.array([0.5, 0.36, 0.41, 0.51])
    st = np.array([0.05, 0.62, 0.67, 0.77])
    xa = np.array([0, 2, 5, 10])
    ### plot these in double y axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(xa, sc, 'g-o', label='space coherence')
    ax2.plot(xa, st, 'b-s', label='time coherence')
    ax1.set_xlabel('J0 (disorder strength)')
    ax1.set_ylabel('space coherence', color='g')
    ax2.set_ylabel('time coherence', color='b')
    plt.title('Coherence vs Disorder Strength')
    plt.show()