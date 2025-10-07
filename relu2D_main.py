import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# relu2D_step function to perform one step of the ReLU 2D simulation
# This function uses PyTorch for efficient computation, especially on GPU.
# %% functional
def relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3):
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

    for _ in range(npf):
        # periodic boundary padding
        reP = F.pad(re, (pad[0], pad[0], pad[1], pad[1]), mode='circular')
        riP = F.pad(ri, (pad[0], pad[0], pad[1], pad[1]), mode='circular')

        conv_re = F.conv2d(reP, we_kernel)
        conv_ri = F.conv2d(riP, wi_kernel)

        mue = K**0.5 * (u[0] + J0[0, 0] * conv_re + J0[0, 1] * conv_ri)
        mui = K**0.5 * (u[1] + J0[1, 0] * conv_re + J0[1, 1] * conv_ri)

        re = re + (dt / tau[0]) * (-re + torch.relu(mue))
        ri = ri + (dt / tau[1]) * (-ri + torch.relu(mui))

    re = re.squeeze().cpu().numpy()
    ri = ri.squeeze().cpu().numpy()
    return re, ri

def relu2D(N, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0):
    if N % 2 != 1:
        raise ValueError('N must be an odd integer')

    print('initializing simulation... ', end='', flush=True)
    str_temp = ''

    # Initialization steps
    re = re0.copy()
    ri = ri0.copy()
    for n1 in range(1, int(np.floor(Nstep_init / npf)) + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 * npf / Nstep_init, 5))
        print(str_temp, end='', flush=True)

        re, ri = relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3)

    print('\nrunning simulation... ', end='', flush=True)
    str_temp = ''

    n_record = int(np.floor(Nstep / npf))
    re_all = np.full((N, N, n_record), np.nan)
    ri_all = np.full((N, N, n_record), np.nan)

    for n1 in range(1, n_record + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 / n_record, 5))
        print(str_temp, end='', flush=True)

        re, ri = relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, J2, J3)

        re_all[:, :, n1 - 1] = re
        ri_all[:, :, n1 - 1] = ri

    print('\n')
    return re_all, ri_all

# %% Main function to run the simulation and visualize results
if __name__ == "__main__":
    # Simulation parameters
    L = 31
    N = L
    dt = 0.0001
    Nstep_init = 1 * 10 ** 3
    Nstep = 2 * 10 ** 3
    npf = 5

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
    re_all, ri_all = relu2D(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, J2, J3, re0, ri0)

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

    # Plot simulation
    plt.figure(figsize=(8, 4))
    plt.plot(tt, re_all.mean(axis=(0, 1)), label='re_all mean')
    plt.plot(tt, ri_all.mean(axis=(0, 1)), label='ri_all mean')
    plt.xlabel('Time')
    plt.ylabel('Mean firing rate')
    plt.legend()
    plt.title('Simulation Results')
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
