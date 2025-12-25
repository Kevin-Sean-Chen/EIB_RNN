import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import scipy as sp
from scipy.ndimage import shift
import math
from matplotlib import pyplot as plt

import sys
from pathlib import Path
# add repo root to sys.path so `scripts` files can import each other
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from scripts.relu2D_reservoir import Relu2DReservoirRNN, make_2D_stim_with_rigid_shift

###
# attempt to use reservoir computing to probe memory of the 2D network
###
# %% setup stimuli, target, and the network
N = 31
T = 500
output_dim = 1
device = 'cpu'

# relu2D params
relu2D_params = {
    'dt': 0.001,
    'ntype': 'relu_gaussian',
    'K': 10**1,
    'tau': np.array([.01, .01]),
    'u': [10, 0],  # Changed to a list
    # 'J0': np.array([[1, -1], [1, -1]]), #
    'J0': np.array([[1, -4], [2, -2]]),
    'sigma': 0.05 * np.array([1, np.sqrt(2)]),
    'J2': np.zeros((2,2)),
    'J3': np.zeros((2,2)),
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

# %% scan through delay tau
taus = np.array([0, 10, 20, 40, 80, 160])  # in steps
mses = np.zeros(len(taus))
epochs = 50
z_hats = []

### loop delays
for i, tau in enumerate(taus):
    ### make delayed target
    print(f"Training with delay tau={tau} steps")
    # Prepare target with delay
    if tau == 0:
        target_delayed = target
    else:
        target_delayed = torch.zeros_like(target)
        target_delayed[:, tau:] = target[:, :-tau]

    # Model, optimizer, loss
    model = Relu2DReservoirRNN(N, T, output_dim, device, relu2D_params)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _ = model(ipt_img)  # expected shape [batch=1, T] (or similar)
        # create a delayed version of the model output that lines up with target_delayed
        # if tau == 0:
        #     output_delayed = output
        # else:
        #     output_delayed = torch.zeros_like(output)
        #     output_delayed[:, tau:] = output[:, :-tau]

        loss = criterion(output, target_delayed)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # record mse and reconstruction (use delayed output to compare to target_delayed)
    with torch.no_grad():
        output, _ = model(ipt_img)
        # if tau == 0:
        #     output_delayed = output
        # else:
        #     output_delayed = torch.zeros_like(output)
        #     output_delayed[:, tau:] = output[:, :-tau]

        mse = criterion(output, target_delayed).item()
        mses[i] = mse
        z_hats.append(output.detach().cpu().numpy())

# %% plot mse vs tau
plt.figure()
plt.plot(taus, mses, '-o')
plt.xlabel('Delay Tau (steps)')
plt.ylabel('Mean Squared Error')
plt.title('Memory Performance vs Delay Tau')
plt.show()

### show z_hat for different taus in the same plot with graded color code
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))
for i, tau in enumerate(taus):
    plt.plot(z_hats[i].flatten(), color=colors[i], label=f'Tau={tau}')
plt.xlabel('Time Step')
plt.ylabel('Drift Angle (radians)')
plt.title('Reconstructed Drift for Different Delay Taus')
plt.legend()
plt.show()