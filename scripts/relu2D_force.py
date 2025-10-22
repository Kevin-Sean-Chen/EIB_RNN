import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import scipy as sp
from scipy.ndimage import shift
import math

# from relu2D_main import relu2D_step
import matplotlib.pyplot as plt

def relu2D_step(re, ri, N, dt, npf, ntype, K, tau, u, J0, sigma, m_vec, out_t):
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

        u0_expanded = torch.tensor(u[0], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape (1,1,N,N)
        # Ensure m_vec and out_t are tensors on the model device.
        # m_vec expected shape: (N, N, output_dim), out_t expected shape: (output_dim,)
        m_vec_t = m_vec.to(device).clone().detach()
        out_t_tensor = out_t.to(device).clone().detach()
        # Multiply along output_dim and sum to produce a spatial feedback map (N,N),
        # then reshape to (1,1,N,N) to match u0_expanded for the dynamics.
        feedback_map = (m_vec_t * out_t_tensor).sum(dim=-1)  # shape (N,N)
        feedback = feedback_map.unsqueeze(0).unsqueeze(0)  # shape (1,1,N,N)

        mue = K**0.5 * (u0_expanded + J0[0, 0] * conv_re + J0[0, 1] * conv_ri + feedback)
        mui = K**0.5 * (u[1] + J0[1, 0] * conv_re + J0[1, 1] * conv_ri)
        # mue = K**0.5 * (u[0] + J0[0, 0] * conv_re + J0[0, 1] * conv_ri)
        # mui = K**0.5 * (u[1] + J0[1, 0] * conv_re + J0[1, 1] * conv_ri)

        re = re + (dt / tau[0]) * (-re + torch.relu(mue))
        ri = ri + (dt / tau[1]) * (-ri + torch.relu(mui))

    re = re.squeeze().cpu().numpy()
    ri = ri.squeeze().cpu().numpy()
    return re, ri

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
        I_xyt[:, :, t] = temp / torch.linalg.norm(temp)

    return I_xyt, drift_series

def spatial_convolution(r,k):
    """
    2D spatial convolution given kernel k and neural field r
    """
    gr = sp.signal.convolve2d(r.squeeze(), k, mode='same',  boundary='wrap') #, fillvalue=0,
    return gr

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def g_kernel(sigma, size):
    """
    Generates a 2D Gaussian kernel in PyTorch.

    :param sigma: Standard deviation of the Gaussian.
    :param size: Size of the kernel (size x size).
    :return: 2D Gaussian kernel as a PyTorch tensor.
    """
    sigma = sigma * size
    center = (size - 1) / 2  # Center of the kernel

    # Create a 2D grid of (x, y) coordinates
    x_coords = torch.arange(size)
    y_coords = torch.arange(size)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")

    # Compute the Gaussian function
    kernel = (1 / (2 * math.pi * sigma**2)) * torch.exp(
        -((x_grid - center)**2 + (y_grid - center)**2) / (2 * sigma**2)
    )

    # Normalize the kernel so it sums to 1
    kernel /= kernel.sum()
    return kernel

def make_2D_stim_with_rigid_shift(N, lt, dt, sigma_xy, device='cpu'):
    """
    Generate a 2D spatiotemporal stimulus with rigid shifts and output PyTorch tensors.

    Args:
        N (int): Size of the 2D grid (NxN).
        lt (int): Number of time steps.
        dt (float): Time step size.
        tau (float): Time constant for angular dynamics.
        sigma_xy (float): Standard deviation for the spatial kernel.
        device (str): Device to place tensors on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: 3D tensor of shifted images with shape (N, N, lt).
        torch.Tensor: Drift angles (radians) of shape (lt,).
    """
    # Generate initial random pattern
    temp_space = np.random.randn(N, N)
    temp_k = g_kernel(sigma_xy, N)
    pattern = spatial_convolution(temp_space, temp_k)
    pattern = pattern - np.mean(pattern)  # Remove baseline
    pattern = pattern / np.max(np.abs(pattern))  # Normalize to unit strength

    # Initialize angular time series
    angt = np.zeros(lt)

    # Generate drift angles over time
    # for tt in range(lt - 1):
    #     # ang = angt[tt] + dt / tau * (mu - angt[tt]) + sig_noise * np.sqrt(dt) * np.random.randn()
    #     ang = torch.cos(torch.tensor(tau * tt, device=device))    ### test ########################
    #     angt[tt + 1] = wrap_to_pi(ang)

    # Add additional sine-based modulations
    time = np.arange(lt) * dt
    angt += np.sin(time / dt / np.pi / 2)  # Higher frequency (was 5)
    angt += np.sin(time / dt / np.pi / 4)  # Higher frequency (was 10)
    angt += np.sin(time / dt / np.pi / 7)  # Higher frequency (was 10)
    angt = angt / np.max(np.abs(angt)) * np.pi

    # Apply abrupt changes
    # angt[lt // 2 : lt // 2 + 20] = 2
    # angt[lt // 2 + 20 : lt // 2 + 80] = -2

    # Set fixed distance for shifts
    distance = 10

    # Initialize the 3D matrix to store the shifted images
    shifted_images = np.zeros((N, N, lt))

    # Loop over time to shift the image
    for i, angle in enumerate(angt):
        # Decompose the shift into x and y components
        shift_x = distance * np.cos(angle)
        shift_y = distance * np.sin(angle)

        # Use scipy.ndimage.shift to apply the shift with wrap mode for periodic boundaries
        shifted_image = shift(pattern, shift=[shift_y, shift_x], mode='wrap')

        # Store the shifted image in the 3D matrix
        shifted_images[:, :, i] = shifted_image

    # Convert numpy arrays to PyTorch tensors
    shifted_images = torch.tensor(shifted_images, device=device, dtype=torch.float32)
    angt = torch.tensor(angt, device=device, dtype=torch.float32)

    return shifted_images, angt

class Relu2DReservoirRNN(nn.Module):
    def __init__(self, N, T, output_dim, device, relu2D_params):
        super().__init__()
        self.N = N
        self.T = T
        self.output_dim = output_dim
        self.device = device
        self.relu2D_params = relu2D_params
        # Readout weights
        self.W_out = nn.Parameter(torch.randn(N*N, output_dim, device=device) * 0.1)

    def forward(self, input_pattern):
        # input_pattern: [N, N, T]
        N = self.N
        T = self.T
        re = torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
        ri = torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
        re_all = []
        out = []
        for t in range(T):
            ### time-dependent linear readout
            out_t = torch.relu(re).reshape(N*N) @ self.W_out  # [output_dim]
            out.append(out_t)
            # Add input as external drive to excitatory population
            u = self.relu2D_params['u']  # Read baseline u
            u[0] = u[0]*0 + input_pattern[:, :, t].cpu().numpy()*10  # Add input 2D array at time t
            re, ri = relu2D_step(
                re.cpu().numpy(), ri.cpu().numpy(), N,
                self.relu2D_params['dt'], 1, self.relu2D_params['ntype'],
                self.relu2D_params['K'], self.relu2D_params['tau'],
                u, self.relu2D_params['J0'], self.relu2D_params['sigma'],
                self.relu2D_params['m'], out_t
            )
            re = torch.tensor(re, device=self.device, dtype=torch.float32)
            ri = torch.tensor(ri, device=self.device, dtype=torch.float32)
            re_all.append(torch.relu(re)) ### testing with input nonlinearity

        re_all = torch.stack(re_all, dim=-1)  # [N, N, T]
        out = torch.stack(out, dim=-1)  # [output_dim, T]
        return out, re_all  # [output_dim, T], [N, N, T]

# --- Setup for training ---
N = 31
T = 500
output_dim = 1
device = 'cpu'

### make feedback weights ###
m_feedback = torch.randn(N, N, output_dim, device=device) * 0.5  # feedback weights
def smooth_random_matrix(N, sigma, device='cpu', scale=0.5):
    # sigma: if <1 interpreted as fraction of N, otherwise in pixels
    if sigma <= 0:
        return torch.randn(N, N, device=device) * scale

    sigma_pix = sigma * N if sigma < 1 else sigma
    ksize = max(3, int(2 * math.ceil(3 * sigma_pix) + 1))  # cover ±3 sigma, make odd

    coords = torch.arange(ksize, device=device) - (ksize - 1) / 2.0
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(xg**2 + yg**2) / (2.0 * (sigma_pix**2)))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,ksize,ksize)

    noise = torch.randn(1, 1, N, N, device=device)
    pad = ksize // 2
    noise_p = F.pad(noise, (pad, pad, pad, pad), mode='circular')
    smooth = F.conv2d(noise_p, kernel).squeeze(0).squeeze(0)  # shape (N,N)
    
    # Z-score the output (mean=0, std=1) then apply scale
    smooth_zscore = (smooth - smooth.mean()) / smooth.std()
    return smooth_zscore * scale

# Example: create an NxN smooth random matrix with smoothness controlled by sigma
sigma_smooth = 0.05  # try values like 0.01 (very rough) up to ~0.2 (very smooth)
m_feedback_smooth = smooth_random_matrix(N, sigma_smooth, device=device, scale=0.1) * 0.5
m_feedback_noise = torch.randn(N, N, output_dim, device=device) * 0.5
m_half = m_feedback_noise*1  # make sure it's (N,N,output_dim)
m_half[:,int(N//2):,:] = 0  # zero out half the matrix for testing

# relu2D params
relu2D_params = {
    'dt': 0.001,
    'ntype': 'relu_gaussian',
    'K': 10**2,
    'tau': np.array([.01, .01]),
    'u': [10, 0],  # Changed to a list
    # 'J0': np.array([[1, -1], [1, -1]]), #
    'J0': np.array([[1, -4], [2, -2]]),
    'sigma': 0.05 * np.array([1, np.sqrt(2)]),
    'm': m_half  # feedback weights
}

# Make drifting pattern as input and target
# ipt_img, drift = make_2D_stim_with_drift(N, T, 0.1, 2, .05*2)
ipt_img, drift = make_2D_stim_with_rigid_shift(N, T, 0.1, 2/N*.5)
ipt_img = ipt_img.to(device)
target = drift.unsqueeze(0)  # [1, T]

# print(np.max(ipt_img[:, :, 0].cpu().numpy()))
# print(np.min(ipt_img[:, :, 0].cpu().numpy()))

# visualize the input
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ipt_img[:, :, 0].cpu().numpy(), cmap='gray')
plt.title('Input Pattern at t=0')
plt.subplot(1, 2, 2)
plt.plot(drift.cpu().numpy(), label='Drift Direction')
plt.title('Drift Direction Over Time')
plt.xlabel('Time Step')
plt.ylabel('Drift Angle (radians)')
plt.legend()
plt.show()

# Model, optimizer, loss
model = Relu2DReservoirRNN(N, T, output_dim, device, relu2D_params)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
epochs = 60
for epoch in range(epochs):
    optimizer.zero_grad()
    output, _ = model(ipt_img)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Evaluation
# ipt_img, drift = make_2D_stim_with_rigid_shift(N, T, 0.1, 2/N*.5) #### new test for generalization!
output, re_all = model(ipt_img)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(output.detach().cpu().numpy().squeeze(), label='readout')
plt.plot(target.cpu().numpy().squeeze(), label='target')
plt.legend()
plt.show()

### visualize three frames of the reservoir state, from initial, middle, and end
re_all = re_all.detach()  # [N, N, T]
plt.figure(figsize=(15, 5))
for idx, i in enumerate([0, T//2, T-1]):
    plt.subplot(1, 3, idx + 1)  # Corrected indexing for subplot
    plt.imshow(re_all[:, :, i].detach().cpu().numpy(), cmap='gray')
    plt.title(f'Reservoir State at t={i}')
plt.show()
