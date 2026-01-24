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

import sys
from pathlib import Path
# add repo root to sys.path so `scripts` files can import each other
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from scripts.relu2D_reservoir import Relu2DReservoirRNN, make_2D_stim_with_rigid_shift, Vanilla_ReservoirRNN
from scripts.relu2D_disorder import gabor2d

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

# %% make WM trask
def make_wm_trial(N, T, delay_period=200, cue_interval=50, lr_trial=None):
    """
    Create a working-memory trial.
    Returns (cue_pattern, target_out, space_stim, input_traj).

    lr_trial: if None, randomly choose 0 or 1. If provided, must be 0 or 1.
             lr_trial==0 -> out_sign = +1, use trigger_pattern1
             lr_trial==1 -> out_sign = -1, use trigger_pattern2
    """
    if lr_trial is None:
        lr = np.random.randint(0, 2)
    else:
        lr = int(lr_trial)

    trigger_pattern1 = np.random.randn(N, N)
    trigger_pattern2 = np.random.randn(N, N)
    G1 = gabor2d(N, f=5*.5, theta=np.deg2rad(30), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5
    G2 = gabor2d(N, f=1*.5, theta=np.deg2rad(60), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5
    G3 = gabor2d(N, f=3*.5, theta=np.deg2rad(90), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5
    trigger_pattern1 = G1.reshape(N, N)*0.1
    trigger_pattern2 = G2.reshape(N, N)*0.1
    cue_pattern = np.random.randn(N, N)*0.1  ### amplitude matters #G3*0.1 #

    space_stim = np.zeros((N, N, T))
    target_out = np.zeros(T)
    input_traj = np.zeros(T)

    out_sign = +1 if lr == 0 else -1

    for tt in range(T):
        if tt < cue_interval:
            space_stim[:, :, tt] = trigger_pattern1 if lr == 0 else trigger_pattern2
            target_out[tt] = 0
            input_traj[tt] = out_sign
        elif tt < cue_interval + delay_period:
            # stay zero during delay
            pass
        elif tt < cue_interval + delay_period + cue_interval:
            space_stim[:, :, tt] = cue_pattern
            start = cue_interval + delay_period
            progress = (tt - start + 1) / float(cue_interval)
            progress = np.clip(progress, 0.0, 1.0)
            target_out[tt] = progress * out_sign
            input_traj[tt] = 1
        else:
            target_out[tt] = out_sign
            space_stim[:, :, tt] = cue_pattern #*np.random.randn()*0.01  ### holds on

    ### tensorize
    # Convert numpy arrays to PyTorch tensors
    target_out = torch.tensor(target_out, device=device, dtype=torch.float32)
    space_stim = torch.tensor(space_stim, device=device, dtype=torch.float32)
    input_traj = torch.tensor(input_traj, device=device, dtype=torch.float32)

    return target_out, space_stim, input_traj


# parameters used previously
delay_period = 250
cue_interval = 30

# create one trial (lr_trial left unspecified -> random)
target_out, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=None)

plt.figure()
plt.subplot(2,2,1)
plt.plot(input_traj.cpu().numpy())
plt.subplot(2,2,2)
plt.plot(target_out.cpu().numpy())
plt.show()
# plt.subplot(2,2,3)
# plt.imshow()

# %% training
# Model, optimizer, loss
model = Relu2DReservoirRNN(N, T, output_dim, device, relu2D_params)
#############
# compare to Vanilla_ReservoirRNN
# also check with input methods (u[0])
#############
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    ### randomized trial
    target_out, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=None)
    output, _ = model(space_stim)
    # loss = criterion(output[-1], target_out[-1]) ### just the output
    loss = criterion(output, target_out)  ### full trace
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# %% evaluation
# Evaluation
output, re_all = model(space_stim)
plt.figure()
plt.plot(output.detach().cpu().numpy().squeeze(), label='readout')
plt.plot(target_out.cpu().numpy().squeeze(), label='target')
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

# %% statistics of decision
n_trials = 10
plt.figure()
for ii in range(n_trials):
    target_out, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=0)
    output, _ = model(space_stim)
    plt.plot(output.detach().cpu().numpy().squeeze(), 'r--')
    plt.plot(target_out.cpu().numpy().squeeze(), 'r')

    target_out, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=1)
    output, _ = model(space_stim)
    plt.plot(output.detach().cpu().numpy().squeeze(), 'b--')
    plt.plot(target_out.cpu().numpy().squeeze(), 'b')

plt.plot(input_traj.cpu().numpy(),'k')
plt.show()