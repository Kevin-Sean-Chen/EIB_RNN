import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

### import WM task for space
import sys
from pathlib import Path
# add repo root to sys.path so `scripts` files can import each other
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

# from scripts.WM_task import make_wm_trial
from scripts.relu2D_disorder import gabor2d

# %% functional
def _make_1d_periodic_gaussian_weights(N: int, sigma: float, device, dtype):
    """
    Discrete periodic Gaussian weights on a 1D ring of length N.
    Matches the reference implementation from relu2D_step().

    Returns w: (N,) such that w[i] depends on periodic distance from center.
    Uses summation over integer wraps to approximate periodic Gaussian.
    """
    dx = 1.0 / N
    x = np.arange(-(N-1)//2, (N-1)//2 + 1)
    k = np.arange(-int(np.ceil(10 * sigma)), int(np.ceil(10 * sigma)) + 1)
    
    # Compute periodic Gaussian (no normalization - matches reference)
    w = np.sum(
        dx * (2 * np.pi * sigma**2)**-0.5 *
        np.exp(-0.5 * (dx * (x[:, None] + k))**2 / sigma**2),
        axis=1
    )
    
    w = torch.tensor(w, dtype=dtype, device=device)
    return w


def _make_2d_separable_kernel(N: int, sigma: float, device, dtype):
    """
    Build a separable 2D kernel K(x,y) = w(x)*w(y), shape (1,1,N,N).
    """
    w = _make_1d_periodic_gaussian_weights(N, sigma, device=device, dtype=dtype)  # (N,)
    K2 = torch.outer(w, w)  # (N,N)
    return K2[None, None, :, :]  # (1,1,N,N)


class Relu2DSpatialReservoir(nn.Module):
    """
    2D E/I spatial reservoir with circular boundary conditions (periodic),
    leaky integrator dynamics, and optional scalar feedback.

    State:
      re, ri: (1,1,N,N)

    Input:
      stim: (N, N, T) or (B, N, N, T) -> will be handled (we keep B optional)
      The stimulus is added to the excitatory drive.

    Readouts:
      out(t) = <phi(re_t), W_out>   (flattened)
      mem(t) = <phi(re_t), W_mem>   (flattened)

    Feedback (optional):
      mu_e += fb_gain * (W_fb_o * out_scalar + W_fb_m * mem_scalar) broadcast over space
      (W_fb_* are fixed random spatial patterns, like "feedback projection")
    """

    def __init__(self, N, T, output_dim, device, params):
        super().__init__()
        self.N = int(N)
        self.T = int(T)
        self.output_dim = int(output_dim)
        self.device = torch.device(device)

        # loading parameters from params dict
        # --- Dynamics params (load directly from params) ---
        self.dt = float(params['dt'])
        tau = params['tau']
        self.tau_e = float(tau[0])
        self.tau_i = float(tau[1])

        self.K = float(params['K'])
        self.sqrtK = self.K ** 0.5

        J0 = torch.tensor(params['J0'], device=self.device, dtype=torch.float32)
        self.register_buffer("J0", J0)

        u = params['u']
        self.u0_e = float(u[0])
        self.u0_i = float(u[1])

        sigma = params['sigma']
        self.sigma_e = float(sigma[0])
        self.sigma_i = float(sigma[1])

        # Precompute kernels once (buffers)
        Ke = _make_2d_separable_kernel(self.N, self.sigma_e, self.device, torch.float32)
        Ki = _make_2d_separable_kernel(self.N, self.sigma_i, self.device, torch.float32)
        self.register_buffer("Ke", Ke)
        self.register_buffer("Ki", Ki)

        # Padding size for conv2d with circular boundary
        self.pad = int(self.Ke.shape[-1] // 2)

        # --- Nonlinearity ---
        self.nl = params.get("nl", "relu")
        if self.nl not in ("relu", "tanh"):
            raise ValueError("params['nl'] must be 'relu' or 'tanh'")
        self.nl = params['nl']

        # --- Trainable readouts on excitatory activity phi(re) ---
        # Flattened dimension: N*N
        self.D = self.N * self.N
        self.W_out = nn.Parameter(torch.randn(self.D, self.output_dim, device=self.device) * 1/self.D**0.5)
        self.W_mem = nn.Parameter(torch.randn(self.D, 1, device=self.device) * 1/self.D**0.5)

        # --- Optional feedback (fixed spatial patterns) ---
        self.fb_gain = float(params['fb_gain'])#float(params.get("fb_gain", 0.0))  # set >0 to enable
        if self.fb_gain != 0.0:
            self.register_buffer("W_fb_o", torch.randn(1, 1, self.N, self.N, device=self.device) * .01)
            self.register_buffer("W_fb_m", torch.randn(1, 1, self.N, self.N, device=self.device) * .01)
        else:
            self.W_fb_o = None
            self.W_fb_m = None

        # self.W_fb_m = nn.Parameter(torch.randn(1, 1, self.N, self.N, device=self.device) * 0.1)

        # ### testing locality
        # self.W_fb_m = self.W_fb_m*0
        # self.W_fb_m[:,:, :5, :5] = torch.randn(1,1,5,5, device=self.device)*0.1

        # --- Optional init scale ---
        self.init_scale = float(params.get("init_scale", 0.1))

        # --- Input gain ---
        self.stim_gain = float(params.get("stim_gain", 1.0))

        # --- Optional: multiple internal steps per frame (like your npf) ---
        self.npf = int(params.get("npf", 1))

    def NL(self, x):
        return F.relu(x) if self.nl == "relu" else torch.tanh(x)

    def _conv_circular(self, x, kernel):
        # x: (B,1,N,N), kernel: (1,1,N,N) typically
        xP = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return F.conv2d(xP, kernel)

    def rnn_step(self, re, ri, stim_t):
        """
        One time-step update.

        re, ri: (B,1,N,N)
        stim_t: (B,1,N,N) external drive added to excitatory channel
        """
        # Possibly iterate multiple micro-steps per frame
        for _ in range(self.npf):
            conv_re_e = self._conv_circular(re, self.Ke)
            conv_ri_i = self._conv_circular(ri, self.Ki)

        # Pre-nonlinearity currents
            # mue = sqrtK * (u0_e + J_EE * conv(re) + J_EI * conv(ri) + stim)
            # mui = sqrtK * (u0_i + J_IE * conv(re) + J_II * conv(ri))
            mue = self.sqrtK * (
                self.u0_e
                + self.J0[0, 0] * conv_re_e
                + self.J0[0, 1] * conv_ri_i
                + self.stim_gain * stim_t
            )
            mui = self.sqrtK * (
                self.u0_i
                + self.J0[1, 0] * conv_re_e
                + self.J0[1, 1] * conv_ri_i
            )

            # Optional feedback (scalar -> spatial pattern)
            if self.fb_gain != 0.0:
                # Use current bounded excitatory activity for readout scalars
                phi_re = self.NL(re)
                rflat = phi_re.flatten(start_dim=1)  # (B, N*N)

                out_scalar = (rflat @ self.W_out).sum(dim=1, keepdim=True)  # (B,1) summed across output_dim
                mem_scalar = (rflat @ self.W_mem).view(-1, 1)               # (B,1)

                # broadcast to (B,1,N,N)
                feedback = (
                    self.W_fb_o * out_scalar[:, None, None]*1    ############## differential feedback test
                    + self.W_fb_m * mem_scalar[:, None, None]*1
                )
                mue = mue + self.fb_gain * feedback

            # Nonlinearity and leaky update
            re = re + (self.dt / self.tau_e) * (-re + self.NL(mue))
            ri = ri + (self.dt / self.tau_i) * (-ri + self.NL(mui))

        return re, ri

    def forward(self, stim):
        """
        stim: (N, N, T) or (B, N, N, T)
        Returns:
          out: (B, output_dim, T)
          mem: (B, 1, T)
          re_all: (B, N, N, T)  (bounded activity phi(re))
        """
        stim = stim.to(self.device).float()
        if stim.dim() == 3:
            stim = stim.unsqueeze(0)  # (1,N,N,T)
        B, N1, N2, T = stim.shape
        assert N1 == self.N and N2 == self.N, "stim spatial dims must match N"
        assert T == self.T, "stim time dim must match T"

        # init state
        re = torch.randn(B, 1, self.N, self.N, device=self.device) * self.init_scale
        ri = torch.randn(B, 1, self.N, self.N, device=self.device) * self.init_scale

        re_list = []
        out_list = []
        mem_list = []

        for t in range(self.T):
            stim_t = stim[:, :, :, t].unsqueeze(1)  # (B,1,N,N)
            re, ri = self.rnn_step(re, ri, stim_t)

            phi_re = self.NL(re)  # bounded / rectified activity used for readout
            re_list.append(phi_re.squeeze(1))  # (B,N,N)

            rflat = phi_re.flatten(start_dim=1)  # (B, N*N)
            out_t = (rflat @ self.W_out)         # (B, output_dim)
            mem_t = (rflat @ self.W_mem)         # (B, 1)

            out_list.append(out_t)
            mem_list.append(mem_t)

        re_all = torch.stack(re_list, dim=-1)                 # (B,N,N,T)
        out = torch.stack(out_list, dim=-1)                   # (B,output_dim,T)
        mem = torch.stack(mem_list, dim=-1)                   # (B,1,T)
        return out, mem, re_all

def make_wm_trial(N, T, delay_period=200, cue_interval=50, lr_trial=None, inpt_patterns=None):
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

    trigger_pattern1, trigger_pattern2, cue_pattern = inpt_patterns
    # trigger_pattern1 = np.random.randn(N)*.1
    # trigger_pattern2 = np.random.randn(N)*.1
    # cue_pattern = np.random.randn(N)*0.1  ### amplitude matters #G3*0.1 #

    space_stim = np.zeros((N, N, T))
    target_out = np.zeros(T)
    input_traj = np.zeros(T)
    target_mem = np.zeros(T)

    out_sign = +1 if lr == 0 else -1

    for tt in range(T):
        if tt < cue_interval:
            space_stim[:, :, tt] = trigger_pattern1 if lr == 0 else trigger_pattern2
            target_out[tt] = 0
            input_traj[tt] = out_sign
            target_mem[tt] = out_sign
        elif tt < cue_interval + delay_period:
            # stay zero during delay
            pass
            target_mem[tt] = out_sign
            progress = (tt - cue_interval + 1) / float(delay_period)
            progress = np.clip(progress, 0.0, 1.0)
            target_mem[tt] = progress * out_sign
            space_stim[:, :, tt] = np.random.randn(N, N)*0.0  #+ cue_pattern*0.0
        elif tt < cue_interval + delay_period + cue_interval:
            space_stim[:, :, tt] = cue_pattern + np.random.randn(N, N)*0.01
            start = cue_interval + delay_period
            progress = (tt - start + 1) / float(cue_interval)
            progress = np.clip(progress, 0.0, 1.0)
            target_out[tt] = progress * out_sign
            input_traj[tt] = 1
            target_mem[tt] = out_sign
        else:
            target_out[tt] = out_sign
            space_stim[:, :, tt] = np.random.randn(N, N)*0.0  #cue_pattern ### holds on
            target_mem[tt] = out_sign

    ### tensorize
    # Convert numpy arrays to PyTorch tensors
    target_out = torch.tensor(target_out, device=device, dtype=torch.float32)
    space_stim = torch.tensor(space_stim, device=device, dtype=torch.float32)
    input_traj = torch.tensor(input_traj, device=device, dtype=torch.float32)
    target_mem = torch.tensor(target_mem, device=device, dtype=torch.float32)

    return target_out, target_mem, space_stim, input_traj


if __name__ == "__main__":
    # --- Setup for training ---
    N = 19
    T = 500
    output_dim = 1
    device = 'cpu'

    # relu2D params
    relu2D_params = {
        'dt': 0.001,
        'ntype': 'relu_gaussian',
        'K': 10**1*3,
        'tau': np.array([.01, .01]),
        'u': [10, 0],  # Changed to a list
        # 'J0': np.array([[1, -1], [1, -1]]), #
        'J0': np.array([[1, -4], [2, -2]]),
        'sigma': 0.05 * np.array([1, np.sqrt(2)]),
        'fb_gain': .0,
        'nl': 'relu',
    }

    # %% simple check with spontaneous activity
    model = Relu2DSpatialReservoir(N, T, output_dim, device, relu2D_params)
    stim = torch.randn(N, N, T) * 0.0  # small random noise
    output, mem, re_all = model(stim)
    print("output shape:", output.shape)
    print("mem shape:", mem.shape)
    print("re_all shape:", re_all.shape)

    re_all = re_all.detach()  # [N, N, T]
    plt.figure(figsize=(15, 5))
    for idx, i in enumerate([0, T//2, T-1]):
        plt.subplot(1, 3, idx + 1)  # Corrected indexing for subplot
        plt.imshow(re_all.squeeze(0)[:, :, i].detach().cpu().numpy(), cmap='gray')
        plt.title(f'Reservoir State at t={i}')
    plt.show()

    plt.figure()
    plt.plot(output.detach().cpu().numpy().squeeze(), label='Output')
    plt.plot(mem.detach().cpu().numpy().squeeze(), label='Mem')
    plt.title("Output on random noise input")
    plt.legend()
    plt.show()

    # %% task trial
    delay_period = 250
    cue_interval = 20

    G1 = gabor2d(N, f=5*.5, theta=np.deg2rad(30), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5
    G2 = gabor2d(N, f=1*.5, theta=np.deg2rad(60), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5
    G3 = gabor2d(N, f=3*.5, theta=np.deg2rad(90), gamma=0.1, phi=.5, normalize=True) ### 0.5,1,1.5

    ipt_patterns = (G1*.01,
                    G2*.01,
                    np.random.randn(N, N)*.01)

    # create one trial (lr_trial left unspecified -> random)
    target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=None, inpt_patterns=ipt_patterns)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(input_traj.cpu().numpy())
    plt.subplot(2,1,2)
    plt.plot(target_out.cpu().numpy())
    plt.show()

    # %% training
    # Model, optimizer, loss
    model = Relu2DSpatialReservoir(N, T, output_dim, device, relu2D_params)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    # Training loop
    epochs = 60
    alpha = 1.0  # weight for memory loss
    for epoch in range(epochs):
        optimizer.zero_grad()
        ### randomized trial
        target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=None, inpt_patterns=ipt_patterns)
        output, mem, _ = model(space_stim)
        # loss = criterion(output[-1], target_out[-1]) ### just the output
        # loss = criterion(output, target_out)  ### full trace
        loss_out = criterion(output.view(-1)[cue_interval:], target_out.view(-1)[cue_interval:])
        loss_mem = criterion(mem.view(-1)[cue_interval:], target_mem.view(-1)[cue_interval:])

        loss = loss_out*1 + alpha * loss_mem
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # %% evaluation
    # Evaluation
    output, mem, re_all = model(space_stim)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(output.detach().cpu().numpy().squeeze(), label='readout')
    plt.plot(target_out.cpu().numpy().squeeze(), label='target')
    plt.title("Output Readout vs Target")
    plt.ylabel('Output')
    plt.xlabel('Time')
    plt.subplot(2,1,2)
    plt.plot(mem.detach().cpu().numpy().squeeze(), label='readout')
    plt.plot(target_mem.cpu().numpy().squeeze(), label='target')
    plt.title("Memory Readout vs Target")
    plt.ylabel('Output')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

    ### visualize three frames of the reservoir state, from initial, middle, and end
    re_all = re_all.detach()  # [N, N, T]

    ### randomly simple some neurons to check activity
    n_sample = 5
    sample_indices = np.random.choice(N*N, n_sample, replace=False)
    plt.figure()
    for idx, neuron_idx in enumerate(sample_indices):
        r_trace = re_all.view(N*N, T)[neuron_idx, :].detach().cpu().numpy()  #### for 2D
        # r_trace = re_all[neuron_idx, :].detach().cpu().numpy()   ### for random
        plt.subplot(n_sample, 1, idx + 1)
        plt.plot(r_trace)
        plt.title(f'Neuron {neuron_idx} Activity Trace')
    plt.tight_layout()
    plt.show()

    # %% statistics of decision
    n_trials = 10
    plt.figure()
    for ii in range(n_trials):
        target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=0, inpt_patterns=ipt_patterns)
        output, mem, _ = model(space_stim)
        plt.subplot(2,1,1)
        plt.plot(output.detach().cpu().numpy().squeeze(), 'r--')
        plt.plot(target_out.cpu().numpy().squeeze(), 'r')
        plt.subplot(2,1,2)
        plt.plot(mem.detach().cpu().numpy().squeeze(), 'r--')
        plt.plot(target_mem.cpu().numpy().squeeze(), 'r')

        target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=1, inpt_patterns=ipt_patterns)
        output, mem, _ = model(space_stim)
        plt.subplot(2,1,1)
        plt.plot(output.detach().cpu().numpy().squeeze(), 'b--')
        plt.plot(target_out.cpu().numpy().squeeze(), 'b')
        plt.subplot(2,1,2)
        plt.plot(mem.detach().cpu().numpy().squeeze(), 'b--')
        plt.plot(target_mem.cpu().numpy().squeeze(), 'b')

    plt.subplot(2,1,1)
    plt.plot(input_traj.cpu().numpy(),'k')

    ### add labels
    plt.subplot(2,1,1)
    plt.title('Output Readout vs Target')
    plt.ylabel('Output')
    plt.legend(['Output', 'Target'])

    plt.subplot(2,1,2)
    plt.title('Memory Readout vs Target')
    plt.ylabel('Output')
    plt.legend(['Output', 'Target'])
    plt.show()