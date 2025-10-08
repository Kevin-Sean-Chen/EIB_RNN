import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %% functional
### general function for relu2D network simulations
def relu2D(N, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, re0, ri0, forward_step):
    """
    takes size, length, visualization, parameters, and network functions
    returns re, ri activities
    """
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

        re, ri = forward_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma)

    print('\nrunning simulation... ', end='', flush=True)
    str_temp = ''

    n_record = int(np.floor(Nstep / npf))
    re_all = np.full((N, N, n_record), np.nan)
    ri_all = np.full((N, N, n_record), np.nan)

    for n1 in range(1, n_record + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 / n_record, 5))
        print(str_temp, end='', flush=True)

        re, ri = forward_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma)

        re_all[:, :, n1 - 1] = re
        ri_all[:, :, n1 - 1] = ri

    print('\n')
    return re_all, ri_all

def visualize_2D(re_all):
    L = re_all.shape[0]
    xx = np.arange(1, L + 1) / L
    yy = np.arange(1, L + 1) / L
    tt = np.arange(re_all.shape[2])
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
    return  