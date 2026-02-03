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

# %% NOTE:
# random chaotic RNN trained as a reservoir, but with memory feedback readout to support working memory task
##########################################
# %% RNN step function

def RNN_step(rt, dt, tau, u, npf, Jij):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert rt to a 1D float tensor on the correct device
    if isinstance(rt, torch.Tensor):
        rt_tensor = rt.to(device).float().reshape(-1)
    else:
        rt_tensor = torch.tensor(rt, dtype=torch.float32, device=device).reshape(-1)
    n = rt_tensor.shape[0]

    # Convert Jij to a tensor and ensure it is an (n, n) matrix if possible
    if isinstance(Jij, torch.Tensor):
        Jij_tensor = Jij.to(device).float()
    else:
        Jij_tensor = torch.tensor(Jij, dtype=torch.float32, device=device)

    if Jij_tensor.dim() == 2 and Jij_tensor.shape == (n, n):
        pass  # already correct shape
    elif Jij_tensor.numel() == n * n:
        Jij_tensor = Jij_tensor.reshape(n, n)
    elif Jij_tensor.dim() == 1 and Jij_tensor.numel() == n:
        # Interpret a length-n vector as a diagonal matrix
        Jij_tensor = torch.diag(Jij_tensor)
    else:
        raise ValueError(f"Jij has incompatible shape {tuple(Jij_tensor.shape)} for rt length {n}")

    # Convert u to a 1D tensor of length n (broadcast/scalar allowed)
    if isinstance(u, torch.Tensor):
        u_tensor = u.to(device).float().reshape(-1)
    else:
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device).reshape(-1)

    if u_tensor.numel() == 1:
        u_tensor = u_tensor.repeat(n)
    elif u_tensor.numel() == n:
        u_tensor = u_tensor.reshape(n)
    else:
        try:
            u_tensor = u_tensor.reshape(n)
        except Exception:
            raise ValueError(f"u has incompatible shape {tuple(u_tensor.shape)} for rt length {n}")

    # Ensure tau is scalar for the update step
    tau_val = float(np.array(tau).reshape(-1)[0])

    for _ in range(npf):
        # compute input drive (matrix product plus external input)
        mu = torch.matmul(Jij_tensor, rt_tensor) + u_tensor
        rt_tensor = rt_tensor + (dt / tau_val) * (-rt_tensor + torch.relu(mu))  ### relu vs. tanh
        nl_input = torch.relu(mu)
        # rt_tensor = rt_tensor + (dt / tau_val) * (-rt_tensor) + nl_input

    return rt_tensor.cpu().numpy()
    # return nl_input.cpu().numpy()

# %% RNN model
# class WM_ReservoirRNN(nn.Module):
#     def __init__(self, N, T, output_dim, device, rnn_params):
#         super().__init__()
#         self.N = N
#         self.T = T
#         self.output_dim = output_dim
#         self.device = device
#         self.rnn_params = rnn_params
#         # Readout weights
#         self.W_out = nn.Parameter(torch.randn(N, output_dim, device=device) * .1)
#         self.W_mem = nn.Parameter(torch.randn(N, 1, device=device) * .1)
#         self.W_fb = torch.randn(1, N, device=device, dtype=torch.float32) * .1  # feedback weights, not trained

#     def forward(self, input_pattern):
#         # input_pattern: [N, N, T]
#         rt = torch.randn(self.N, device=self.device, dtype=torch.float32)
#         r_all = []
#         for t in range(self.T):
#             # Add input as external drive to excitatory population
#             u = self.rnn_params['u']  # Read baseline u
#             # Vectorize input_pattern at time t to an N^2 vector and add to baseline u
#             input_vec = input_pattern[:, t].reshape(-1).cpu().numpy()  # shape (N*N,)
#             u = np.asarray(u) * 1 + input_vec * 1
#             ### pass (rt, dt, tau, u, npf, Jij)
#             rt = RNN_step(
#                 rt.detach().numpy(), self.rnn_params['dt'], self.rnn_params['tau'], u, 1, self.rnn_params['J0']
#             )
#             rt = torch.tensor(rt, device=self.device, dtype=torch.float32)
#             # print(rt.shape)
#             # compute scalar feedback as dot product between activity and memory weights
#             feedback_scalar = torch.dot(torch.tanh(rt), self.W_mem.view(-1))
#             rt = rt + self.W_fb.view(-1) * feedback_scalar * (self.rnn_params['dt'] / self.rnn_params['tau'])  # add N-vector feedback
#             r_all.append(torch.tanh(rt)) ### testing with input nonlinearity

#         r_all = torch.stack(r_all, dim=-1)  # [N, T] for non-spatial network
#         # Readout: already shape [N, T]
#         readout = r_all
#         out = readout.T @ self.W_out  # [T, output_dim]
#         mem = readout.T @ self.W_mem  # [T, 1]
#         return out.T, mem.T, r_all  # [output_dim, T], [N, N, T]
    

class WM_ReservoirRNN(nn.Module):
    def __init__(self, N, T, output_dim, device, rnn_params):
        super().__init__()
        self.N = N
        self.T = T
        self.output_dim = output_dim
        self.device = torch.device(device)

        # --- Trainable readouts ---
        self.W_out = nn.Parameter(torch.randn(N, output_dim, device=self.device) * 0.1)
        self.W_mem = nn.Parameter(torch.randn(N, 1, device=self.device) * 0.1)

        # --- Fixed feedback projection (N,) ---
        self.register_buffer("W_fb_o", torch.randn(N, device=self.device) * 0.1)
        self.register_buffer("W_fb_m", torch.randn(N, device=self.device) * 0.1)

        # --- Fixed reservoir weights ---
        J0 = rnn_params["J0"]
        if not torch.is_tensor(J0):
            J0 = torch.tensor(J0, dtype=torch.float32)
        self.register_buffer("J0", J0.to(self.device).float())

        # Scalars
        self.dt = float(rnn_params["dt"])
        self.tau = float(rnn_params["tau"])

        # Baseline input u0: allow scalar or vector length N
        u0 = rnn_params.get("u", 0.0)
        if not torch.is_tensor(u0):
            u0 = torch.tensor(u0, dtype=torch.float32)
        u0 = u0.to(self.device).float()
        if u0.numel() == 1:
            u0 = u0.repeat(N)
        self.register_buffer("u0", u0.reshape(N))

        # Optional feedback gain (recommended to start small)
        self.fb_gain = float(rnn_params.get("fb_gain", 1.))

    def NL_function(self, x):
        # return torch.relu(x)
        return torch.tanh(x)
    
    def rnn_step(self, rt, u_t):
        """
        rt: (N,)
        u_t: (N,)
        Returns rt_next: (N,)
        Feedback enters BEFORE nonlinearity: mu = J0@rt + u_t + W_fb_o * feedback_scalar + W_fb_m * feedback_scalar_mem
        """
        # feedback scalar read out from current state (differentiable wrt W_mem)
        feedback_output = torch.dot(self.NL_function(rt), self.W_out.view(-1))  # scalar
        feedback_memory = torch.dot(self.NL_function(rt), self.W_mem.view(-1))  # scalar

        # total current BEFORE nonlinearity
        mu = self.J0 @ rt + u_t + self.fb_gain * (self.W_fb_o * feedback_output + self.W_fb_m * feedback_memory)  # (N,)

        # nonlinearity (tanh is usually more stable than ReLU for ESN-like setups)
        nl = self.NL_function(mu)

        # leaky integrator update
        rt_next = rt + (self.dt / self.tau) * (-rt + nl)
        return rt_next

    def forward(self, space_stim):
        """
        space_stim: (N, T)
        Returns:
          out: (output_dim, T)
          mem: (1, T)
          r_all: (N, T)
        """
        space_stim = space_stim.to(self.device).float()

        # deterministic init helps debugging (try random later if desired)
        # rt = torch.zeros(self.N, device=self.device)
        rt = torch.randn(self.N, device=self.device) * 0.1

        r_list = []
        for t in range(self.T):
            u_t = self.u0 + space_stim[:, t]
            rt = self.rnn_step(rt, u_t)
            r_list.append(self.NL_function(rt))  # store bounded state for readout

        r_all = torch.stack(r_list, dim=1)  # (N, T)

        out = (r_all.T @ self.W_out).T      # (output_dim, T)
        mem = (r_all.T @ self.W_mem).T      # (1, T)
        return out, mem, r_all


# %% make WM trask
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

    space_stim = np.zeros((N, T))
    target_out = np.zeros(T)
    input_traj = np.zeros(T)
    target_mem = np.zeros(T)

    out_sign = +1 if lr == 0 else -1

    for tt in range(T):
        if tt < cue_interval:
            space_stim[:, tt] = trigger_pattern1 if lr == 0 else trigger_pattern2
            target_out[tt] = 0
            input_traj[tt] = out_sign
            target_mem[tt] = out_sign
        elif tt < cue_interval + delay_period:
            # stay zero during delay
            pass
            target_mem[tt] = out_sign
            # progress = (tt - cue_interval + 1) / float(delay_period)
            # progress = np.clip(progress, 0.0, 1.0)
            # target_mem[tt] = progress * out_sign
        elif tt < cue_interval + delay_period + cue_interval:
            space_stim[:, tt] = cue_pattern
            start = cue_interval + delay_period
            progress = (tt - start + 1) / float(cue_interval)
            progress = np.clip(progress, 0.0, 1.0)
            target_out[tt] = progress * out_sign
            input_traj[tt] = 1
            target_mem[tt] = out_sign
        else:
            target_out[tt] = out_sign
            space_stim[:, tt] = np.random.randn(N)*0.01  #cue_pattern ### holds on
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
    N = 201
    T = 500
    output_dim = 1
    device = 'cpu'
    alpha = .1  ### weight for memory loss


    reluRNN_params = {
        'dt': 0.001,
        'tau': 0.005,
        'J0': torch.randn(N, N, device=device, dtype=torch.float32) * (15 / math.sqrt(N*N)), ### tune this!
        'u': 0,
    }
    model = WM_ReservoirRNN(N, T, output_dim, device, reluRNN_params)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    # task parameters
    # parameters used previously
    delay_period = 200
    cue_interval = 30

    # inspect one trial
    # create one trial (lr_trial left unspecified -> random)
    ipt_patterns = (np.random.randn(N)*.1,
                    np.random.randn(N)*.1,
                    np.random.randn(N)*.1)
    target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=None, inpt_patterns=ipt_patterns)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(input_traj.cpu().numpy()); plt.ylabel('input traj')
    plt.subplot(3,1,2)
    plt.plot(target_out.cpu().numpy()); plt.ylabel('target out')
    plt.subplot(3,1,3)
    plt.plot(target_mem.cpu().numpy()); plt.ylabel('target mem')
    plt.show()

    # Training loop
    epochs = 150
    for epoch in range(epochs):
        optimizer.zero_grad()
        ### randomized trial
        target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=None, inpt_patterns=ipt_patterns)
        output, mem, _ = model(space_stim)
        # loss_out = criterion(output[-1], target_out[-1]) ### just the output
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
    plt.plot(output.detach().cpu().numpy().squeeze(), label='readout')
    plt.plot(target_out.cpu().numpy().squeeze(), label='target')
    plt.legend()
    plt.show()

    ### visualize three frames of the reservoir state, from initial, middle, and end
    re_all = re_all.detach()  # [N, N, T]
    # plt.figure(figsize=(15, 5))
    # for idx, i in enumerate([0, T//2, T-1]):
    #     plt.subplot(1, 3, idx + 1)  # Corrected indexing for subplot
    #     plt.imshow(re_all[:, :, i].detach().cpu().numpy(), cmap='gray')
    #     plt.title(f'Reservoir State at t={i}')
    # plt.show()

    ### randomly simple some neurons to check activity
    n_sample = 5
    sample_indices = np.random.choice(N, n_sample, replace=False)
    plt.figure()
    for idx, neuron_idx in enumerate(sample_indices):
        # r_trace = re_all.view(N*N, T)[neuron_idx, :].detach().cpu().numpy()  #### for 2D
        r_trace = re_all[neuron_idx, :].detach().cpu().numpy()   ### for random
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