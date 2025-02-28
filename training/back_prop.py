# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 04:36:21 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

import scipy as sp
from scipy.ndimage import shift

# %% convolution functions
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

# %% 2D RNN
class twoD_RNN(nn.Module):
    def __init__(self, N, T, output_dim, device='cpu'):
        super(twoD_RNN, self).__init__()
        self.N = N
        self.N_total = N**2
        self.device = device
        self.T = T
        self.output_dim = output_dim
        
        ### 2D EI parameters
        self.k_size = N-1 #23
        self.dt = 0.001
        self.sig_e = 0.1
        self.tau_e = 0.005
        self.sig_i = 0.2 ### balance is needed!?!?!? #############
        self.tau_i = 0.015
        rescale = 1.
        self.Wee = 1* 1.*(N**2*self.sig_e**2*torch.pi*1)**0.5 *rescale  # recurrent weights
        self.Wei = -2.*(N**2*self.sig_i**2*torch.pi*1)**0.5 *rescale
        self.Wie = 1*  .99*(N**2*self.sig_e**2*torch.pi*1)**0.5 *rescale
        self.Wii = -1.8*(N**2*self.sig_i**2*torch.pi*1)**0.5 *rescale
        self.mu_e = 1.*rescale*1
        self.mu_i = .8*rescale*1

        # self.log_ee = nn.Parameter(torch.randn(1, device=device)*0.)#0
        # self.log_ei = nn.Parameter(torch.randn(1, device=device)*0.+0.7) #0.7
        # Recurrent weights (J_ij)
        # self.Ji = nn.Parameter(torch.randn(N-1, N-1, device=device) * 0.1)
        self.log_sig_e = nn.Parameter(torch.randn(1, device=device)*0. + torch.log(torch.zeros(1)+0.1)) #torch.log(torch.zeros(1)+0.1) #
        self.log_sig_i = nn.Parameter(torch.randn(1, device=device)*0. + torch.log(torch.zeros(1)+0.2)) #torch.log(torch.zeros(1)+0.2) #

        # Input weights
        self.W_in = torch.ones(N, N)
        #nn.Parameter(torch.randn(Ns, Ns, device=device) * 0.1)

        # Readout weights (W_i)
        self.W_out = nn.Parameter(torch.randn(N**2, output_dim, device=device) * 0.1)
        
    def phi(self, x):
        nl = torch.where(x > 0, torch.tanh(x), torch.zeros_like(x))
        return nl
    
    def spatial_convolution_with_wrap(self, r, k):
        """
        2D spatial convolution with circular boundary conditions using PyTorch.
        The output size should remain N x N.
        """
        r = r.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        k = k.unsqueeze(0).unsqueeze(0)  # (kh, kw) -> (1, 1, kh, kw)
        
        # Compute the amount of padding needed for circular boundary conditions
        pad_h = k.shape[-2] // 2
        pad_w = k.shape[-1] // 2
    
        # Apply circular padding to the input tensor
        r_padded = F.pad(r, (pad_w, pad_w, pad_h, pad_h), mode="circular")
    
        # Perform the convolution and ensure the output size matches the input size
        gr = F.conv2d(r_padded, k, padding=0)  # Padding is 0 to ensure the output is N x N
    
        return gr.squeeze()


    # def forward(self, ipt_img):
        
    #     ### initialize tensors
    #     re_xy = torch.FloatTensor(self.N, self.N, self.T)
    #     ri_xy = torch.FloatTensor(self.N, self.N, self.T) 
    #     he_xy = torch.FloatTensor(self.N, self.N, self.T)
    #     hi_xy = torch.FloatTensor(self.N, self.N, self.T)
    #     readout = torch.FloatTensor(self.output_dim, self.T)
        
    #     ### loop
    #     for tt in range(self.T-1):
    #         ### convolve
    #         ge_conv_re = self.spatial_convolution_with_wrap(re_xy[:,:,tt], g_kernel(self.sig_e, self.N-1))
    #         gi_conv_ri = self.spatial_convolution_with_wrap(ri_xy[:,:,tt], self.Ji)
    #         # print(gi_conv_ri.shape)
    #         ### iterate time
    #         he_xy[:,:,tt+1] = he_xy[:,:,tt] + self.dt/self.tau_e*( -he_xy[:,:,tt] + (self.Wee*(ge_conv_re) + self.Wei*(gi_conv_ri) + self.mu_e)  + self.W_in*ipt_img[:,:,tt] )
    #         hi_xy[:,:,tt+1] = hi_xy[:,:,tt] + self.dt/self.tau_i*( -hi_xy[:,:,tt] + (self.Wie*(ge_conv_re) + self.Wii*(gi_conv_ri) + self.mu_i) )
    #         re_xy[:,:,tt+1] = self.phi(he_xy[:,:,tt+1])
    #         ri_xy[:,:,tt+1] = self.phi(hi_xy[:,:,tt+1])
            
    #         readout[:,tt+1] = re_xy[:,:,tt+1].reshape(-1) @ self.W_out
    #     return re_xy, ri_xy, readout
    def forward(self, ipt_img, return_current=None):
        """
        Forward pass through the 2D RNN model.
        """
        # Initialize tensors
        re_xyi = torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
        ri_xyi = torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
        he_xyi = torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
        hi_xyi = torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
        readout = torch.randn(self.output_dim, self.T, device=self.device, dtype=torch.float32)
        re_xy, ri_xy, he_xy, hi_xy = [],[],[],[]
        measure_mu, measure_mu_ex = [],[]
        
        # Pre-compute the kernel for efficiency
        # g_kernel_e = g_kernel(self.sig_e, self.N - 1).to(self.device)
        self.sig_e = torch.exp(self.log_sig_e)
        g_kernel_e = g_kernel(self.sig_e, self.k_size).to(self.device)
        self.sig_i = torch.exp(self.log_sig_i)
        g_kernel_i = g_kernel(self.sig_i, self.k_size).to(self.device)
        
        rescale = 1.
        # self.ee = torch.exp(self.log_ee)
        # self.ei = torch.exp(self.log_ei) ### test with learning mean weight!
        # self.Wee = self.ee* 1.*(N**2*self.sig_e**2*torch.pi*1)**0.5 *rescale  # recurrent weights
        # self.Wei = -self.ei*(N**2*self.sig_i**2*torch.pi*1)**0.5 *rescale
        
        # self.Wie = 1*  .99*(N**2*self.sig_e**2*torch.pi*1)**0.5 *rescale
        # self.Wii = -1.8*(N**2*self.sig_i**2*torch.pi*1)**0.5 *rescale
        
        # Loop through time steps
        for tt in range(self.T):
            # Convolve
            ge_conv_re = self.spatial_convolution_with_wrap(re_xyi, g_kernel_e)
            # gi_conv_ri = self.spatial_convolution_with_wrap(ri_xyi, self.Ji)
            gi_conv_ri = self.spatial_convolution_with_wrap(ri_xyi, g_kernel_i)
    
            # Compute next states
            he_xyi = (
                he_xyi + 
                self.dt / self.tau_e * (
                    -he_xyi +
                    (self.Wee * ge_conv_re + self.Wei * gi_conv_ri + self.mu_e) +
                    self.W_in * ipt_img[:, :, tt]
                )
            ).clone()  # Clone to avoid in-place issues
            
            hi_xyi = (
                hi_xyi +
                self.dt / self.tau_i * (
                    -hi_xyi +
                    (self.Wie * ge_conv_re + self.Wii * gi_conv_ri + self.mu_i)
                )
            ).clone()  # Clone to avoid in-place issues
            
            re_xyi = self.phi(he_xyi)
            ri_xyi = self.phi(hi_xyi)
    
            # Update states
            re_xy.append(re_xyi.clone())
            ri_xy.append(ri_xyi.clone())
            he_xy.append(he_xyi.clone())
            hi_xy.append(hi_xyi.clone())
    
            # Compute readout
            readout[:, tt] = re_xyi.reshape(-1) @ self.W_out
            
            # measure balanced current
            measure_mu.append( torch.abs(  self.Wee*(ge_conv_re) + self.Wei*(gi_conv_ri) + self.mu_e ).clone() )
            measure_mu_ex.append( (self.Wee*ge_conv_re + self.mu_e ).clone() )
        
        re_xy = torch.stack(re_xy, dim=-1)  # Shape: [N, N, T]
        ri_xy = torch.stack(ri_xy, dim=-1) 
        measure_mu = torch.stack(measure_mu, dim=-1)
        measure_mu_ex = torch.stack(measure_mu_ex, dim=-1)
        
        if return_current is None:
            return re_xy, ri_xy, readout
        else:
            return re_xy, ri_xy, readout, measure_mu, measure_mu_ex



# %% make stim
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

def make_2D_stim_with_rigid_shift(N, lt, dt, tau, sigma_xy, device='cpu'):
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
    for tt in range(lt - 1):
        ang = angt[tt] + dt / tau * (mu - angt[tt]) + sig_noise * np.sqrt(dt) * np.random.randn()
        angt[tt + 1] = wrap_to_pi(ang)

    # Add additional sine-based modulations
    time = np.arange(lt) * dt
    angt += np.sin(time / dt / np.pi / 10)
    angt += np.sin(time / dt / np.pi / 30)
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



# %% spontaneous
# Initialize the network
N = 30  # Size of the 2D grid
T = 700  # Number of time steps
output_dim = 1
device = "cpu"

rnn = twoD_RNN(N, T, output_dim, device=device)

# Input image (randomly generated)
# ipt_img = torch.randn(N, N,T, device=device)
ipt_img, drift = make_2D_stim_with_drift(N, T, 0.1, 2, .05*2)
#### for stim that shifts #####
sigma_xy = 2/N #0.1 #2/N
tau = .5
mu = 0
sig_noise = 2
dt = 0.001
ipt_img, drift = make_2D_stim_with_rigid_shift(N, T, dt, tau, sigma_xy)

# Run the forward pass
re_xy, ri_xy, readout, meas_mu, meas_mu_ex = rnn(ipt_img*1, return_current=True)

# Plot the dynamics
time = torch.arange(T)

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(time.numpy(), readout.detach().numpy().squeeze(), label="Readout")
plt.xlabel("Time")
plt.ylabel("Readout")
plt.title("Readout Dynamics")
plt.legend()

plt.subplot(3, 1, 2)
plt.imshow(re_xy[:, :, -1].detach().numpy(), cmap="viridis", aspect="auto")
plt.colorbar(label="Activity")
plt.title("Recurrent Excitatory Activity at Final Time Step")

plt.subplot(3, 1, 3)
plt.plot(time.numpy(), re_xy[1, 1, :].detach().numpy())
plt.title("Recurrent Excitatory Activity at Final Time Step")

plt.tight_layout()
plt.show()

# %% back-prop test!!
###############################################################################
# %% train setting
learning_rate = 0.01
epochs = 60

model = twoD_RNN(N, T, output_dim, device=device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# %% make target
stim, target = make_2D_stim_with_drift(N, T, 0.1, 2, .05)
stim, target = make_2D_stim_with_rigid_shift(N, T, dt, tau, sigma_xy)

# %% training loop
# Track loss over epochs
loss_history = []

for epoch in range(epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    _, _, output = model(stim)
    
    # Compute the loss
    loss = criterion(output.squeeze(), target.squeeze())
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Store and print loss
    loss_history.append(loss.item())
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# %% evaluation
# re_xy, ri_xy, readout = rnn(ipt_img)
re_xy, ri_xy, readout, meas_mu, meas_mu_ex = model(stim, return_current=True)
plt.figure()
plt.plot(readout.detach().numpy().T)
plt.plot(target)

beta_t = (meas_mu[:,:,:] / meas_mu_ex[:,:,:]).detach().numpy()
plt.figure()
plt.hist(beta_t.reshape(-1), 1000)
plt.xlabel(r'$\beta_t$', fontsize=20);plt.ylabel('counts', fontsize=20); plt.title(f'N= {N}', fontsize=20)
plt.xlim([0,6])
print('median of beta: ', np.nanmedian(beta_t))
### notes ###
# show that EI-balance is required for good training
# test with learning balance parameters
# test with distance-contrained code... like seRNN! -> adpating to input? --->>> connect back to tuning to freq/space
#### -> H0: EI is good for trainig, learned params are balanced, and there can be spatial matching...!?!?!?

# %% worsen if placed back!
# model.log_ee = nn.Parameter(torch.randn(1,)*0)
# model.log_ei = nn.Parameter(torch.randn(1,)*0+0.7)
# model.log_sig_e = nn.Parameter(torch.randn(1, device=device)*0. + torch.log(torch.zeros(1)+0.1))
# model.log_sig_i = nn.Parameter(torch.randn(1, device=device)*0. + torch.log(torch.zeros(1)+0.2))
