from ctypes.wintypes import RGB
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from datetime import datetime
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def make_2D_stim_with_drift(N, lt, time_f, space_f, drift_rate, device='cpu'):
    """
    Generate a 2D spatiotemporal sine wave stimulus with changing drift direction over time
    and output the time series of drift directions.
    
    Args:
        N (int): Size of the 2D grid (NxN).
        lt (int): Number of time steps.
        time_f (float): Temporal frequency factor.
        space_f (float): Spatial frequency factor.
        drift_rate (float): Rate of change in drift direction (radians per time step).
        device (str): Device to place tensors on ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: Stimulus tensor of shape (N, N, lt).
        torch.Tensor: Drift direction angles (in radians) of shape (lt,).
    """
    x = torch.linspace(0, space_f * 2 * np.pi, N, device=device)  # Spatial frequency
    y = torch.linspace(0, space_f * 2 * np.pi, N, device=device)  # Spatial frequency
    X, Y = torch.meshgrid(x, y, indexing='ij')  # Create 2D grid
    I_xyt = torch.zeros((N, N, lt), device=device)  # Initialize the 3D tensor
    drift_series = torch.zeros(lt, device=device)  # Initialize the drift angle time series

    # Populate the tensor
    for t in range(lt):
        # Update drift direction based on drift_rate
        drift_angle = torch.cos(torch.tensor(drift_rate * t, device=device))  # Convert to tensor
        drift_series[t] = drift_angle  # Store the drift angle at time t

        # Compute directional shift
        direction_x = torch.cos(drift_angle)
        direction_y = torch.sin(drift_angle)

        # Create a 2D sine wave with time-varying drift direction
        temp = torch.sin(direction_x * X + direction_y * Y + t * time_f * (2 * np.pi / N))

        # Normalize and assign
        # I_xyt[:, :, t] = temp / torch.linalg.norm(temp)
        # Normalize temp to be between -1 and 1
        temp_norm = (temp - temp.min()) / (temp.max() - temp.min()) * 2 - 1
        I_xyt[:, :, t] = temp_norm

    return I_xyt, drift_series

def relu2D_driven_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, input_pattern):
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

        mue = K**0.5 * (u[0] + J0[0, 0] * conv_re + J0[0, 1] * conv_ri + input_pattern)
        mui = K**0.5 * (u[1] + J0[1, 0] * conv_re + J0[1, 1] * conv_ri)

        re = re + (dt / tau[0]) * (-re + torch.relu(mue))
        ri = ri + (dt / tau[1]) * (-ri + torch.relu(mui))

    re = re.squeeze().cpu().numpy()
    ri = ri.squeeze().cpu().numpy()
    return re, ri

def relu2D_driven(N, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, input_pattern, re0, ri0):
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

        input_pattern_t = input_pattern[:, :, n1 - 1]*0 ### no need to use input pattern in initialization
        re, ri = relu2D_driven_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, input_pattern_t)

    print('\nrunning simulation... ', end='', flush=True)
    str_temp = ''

    n_record = int(np.floor(Nstep / npf))
    re_all = np.full((N, N, n_record), np.nan)
    ri_all = np.full((N, N, n_record), np.nan)

    for n1 in range(1, n_record + 1):
        print('\b' * len(str_temp), end='', flush=True)
        str_temp = str(round(n1 / n_record, 5))
        print(str_temp, end='', flush=True)

        input_pattern_t = input_pattern[:, :, n1 - 1]
        re, ri = relu2D_driven_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, input_pattern_t)

        re_all[:, :, n1 - 1] = re
        ri_all[:, :, n1 - 1] = ri

    print('\n')
    return re_all, ri_all


# %% Main function to run the simulation and visualize results
if __name__ == "__main__":
    # stimulation parameters
    SAVE = False
    L = 31
    time_f, space_f, drift_rate, device = 1.0, 2.5*3, 0.0, 'cpu'
    N = L
    dt = 0.0001
    Nstep_init = 1 * 10 ** 3
    Nstep = 1 * 10 ** 3
    npf = 1

    I_xyt, drift_series = make_2D_stim_with_drift(N, Nstep, time_f, space_f, drift_rate, device=device)

    I_xyt = I_xyt*1*0

    ### network parameters
    ntype = 'relu_gaussian'
    J0 = np.array([[1, -4], [2, -2]])
    K = 10 ** 2
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

    # Run simulation
    re_all, ri_all = relu2D_driven(L, dt, Nstep_init, Nstep, npf, ntype, K, tau, u, J0, sigma, I_xyt, re0, ri0)

    ### visualizations
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(I_xyt[:, :, 0].cpu().numpy(), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.plot(I_xyt[0, 0, :].cpu().numpy())
    plt.subplot(1, 3, 3)
    plt.plot(drift_series.cpu().numpy())
    plt.show()

    # Create animation for I_xyt in time
    fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
    im_anim = ax_anim.imshow(I_xyt[:, :, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='gray')
    ax_anim.set_xlabel('x')
    ax_anim.set_ylabel('y')
    cbar = fig_anim.colorbar(im_anim, ax=ax_anim, fraction=0.046, pad=0.04)

    # Add time label inside the image (upper left corner)
    ax_anim.set_title('Input Stimulus (I_xyt)')
    time_text = ax_anim.text(0.02, 0.95, '', color='white',
                             ha='left', va='top', transform=ax_anim.transAxes,
                             fontsize=10, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

    def update(frame):
        im_anim.set_data(I_xyt[:, :, frame].cpu().numpy())
        time_text.set_text(f't = {frame}')
        return [im_anim, time_text]

    ani = animation.FuncAnimation(
        fig_anim, update, frames=range(I_xyt.shape[2]),
        interval=50, blit=True, repeat=True
    )

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
    # --- Visualization using matplotlib animation ---
    fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
    im_anim = ax_anim.imshow(re_all[:, :, 0], aspect='auto', origin='lower',
                            extent=[xx[0], xx[-1], yy[0], yy[-1]])
    ax_anim.set_xlabel('x')
    ax_anim.set_ylabel('y')
    cbar = fig_anim.colorbar(im_anim, ax=ax_anim, fraction=0.046, pad=0.04)

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
        filename = f"2Drelu_driven_{datetime.now().strftime('%m-%d-%Y %H-%M-%S')}"
        output_path = os.path.join('video', filename + '.gif')
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        # Display in notebook (optional)
        from IPython.display import display, Image as IPImage
        display(IPImage(filename=output_path))