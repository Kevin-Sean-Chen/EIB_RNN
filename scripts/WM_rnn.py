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

### import WM task for space
import sys
from pathlib import Path
# add repo root to sys.path so `scripts` files can import each other
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

# from scripts.WM_task import make_wm_trial
from scripts.relu2D_disorder import gabor2d
from scripts.WM_res import eval_model


# %% NOTE:
# random chaotic RNN trained as a reservoir, for comparison to 2D spatial EI network!
##########################################
# %% RNN class for reservoir computing, on WM task

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
        return torch.relu(x)
        # return torch.tanh(x)
    
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
def make_wm_trial(N, T, delay_period=200, cue_interval=50, lr_trial=None, inpt_patterns=None, ramp_mem=False):
    """
    Create a working-memory trial for 1D RNN.
    Returns (target_out, target_mem, space_stim, lr).
    
    lr_trial: if None, randomly choose 0 or 1. If provided, must be 0 or 1.
             lr_trial==0 -> out_sign = +1, use trigger_pattern1
             lr_trial==1 -> out_sign = -1, use trigger_pattern2
    """
    if lr_trial is None:
        lr = np.random.randint(0, 2)
    else:
        lr = int(lr_trial)

    trigger_pattern1, trigger_pattern2, cue_pattern = inpt_patterns

    space_stim = np.zeros((N, T), dtype=np.float32)
    target_out = np.zeros((T,), dtype=np.float32)
    target_mem = np.zeros((T,), dtype=np.float32)

    out_sign = +1.0 if lr == 0 else -1.0

    for tt in range(T):
        if tt < cue_interval:
            # triggered pattern
            space_stim[:, tt] = trigger_pattern1 if lr == 0 else trigger_pattern2
            target_out[tt] = 0.0
            target_mem[tt] = out_sign

        elif tt < cue_interval + delay_period:
            # no input during delay
            space_stim[:, tt] = 0.0
            if ramp_mem:  ### ramping memory target
                progress = (tt - cue_interval + 1) / float(delay_period)
                target_mem[tt] = np.clip(progress, 0.0, 1.0) * out_sign
            else:  ### fixed memory target
                target_mem[tt] = out_sign
            target_out[tt] = 0.0

        elif tt < cue_interval + delay_period + cue_interval:
            # go cue (same for both)
            space_stim[:, tt] = cue_pattern
            start = cue_interval + delay_period
            progress = (tt - start + 1) / float(cue_interval)
            target_out[tt] = np.clip(progress, 0.0, 1.0) * out_sign
            target_mem[tt] = out_sign

        else:
            space_stim[:, tt] = 0.0
            target_out[tt] = out_sign
            target_mem[tt] = out_sign

    return (
        torch.tensor(target_out, dtype=torch.float32),
        torch.tensor(target_mem, dtype=torch.float32),
        torch.tensor(space_stim, dtype=torch.float32),
        lr
    )


# -------------------------
# Ridge regression utilities
# -------------------------
def ridge_solve(X, Y, lam):
    """
    X: (M, P), Y: (M, Q)
    W = (X^T X + lam I)^-1 X^T Y
    """
    P = X.shape[1]
    A = X.T @ X + lam * np.eye(P, dtype=np.float64)
    B = X.T @ Y
    return np.linalg.solve(A, B)

@torch.no_grad()
def collect_xy(model, trial_fn, n_trials, time_selector):
    """
    Collect design matrix X and targets Y_out, Y_mem.
    Adds bias column.

    time_selector: function(T)->1D numpy/int array of indices
    """
    model.eval()
    X_list, Yout_list, Ymem_list = [], [], []

    for _ in range(n_trials):
        target_out, target_mem, stim, _ = trial_fn()
        _, _, r_all = model(stim)          # r_all: (N, T)
        N, T = r_all.shape

        R = r_all.T                         # (T, N)
        idx = time_selector(T)              # indices we train on
        R = R[idx, :]                       # (Tsel, N)

        yout = target_out.view(-1, 1)[idx]
        ymem = target_mem.view(-1, 1)[idx]

        X_list.append(R.cpu())
        Yout_list.append(yout.cpu())
        Ymem_list.append(ymem.cpu())

    X = torch.cat(X_list, dim=0).numpy().astype(np.float64)      # (M, N)
    Yout = torch.cat(Yout_list, dim=0).numpy().astype(np.float64) # (M,1)
    Ymem = torch.cat(Ymem_list, dim=0).numpy().astype(np.float64) # (M,1)

    # add bias column for regression
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)  # (M, N+1)
    return Xb, Yout, Ymem

def set_readouts_from_ridge(model, Wout, Wmem):
    """
    Wout: (N+1, output_dim)
    Wmem: (N+1, 1)
    """
    N = model.N
    with torch.no_grad():
        model.W_out.copy_(torch.tensor(Wout[:N, :], dtype=torch.float32, device=model.device))
        model.W_mem.copy_(torch.tensor(Wmem[:N, :], dtype=torch.float32, device=model.device))
        # Note: biases stored in last row of Wout/Wmem; you could store separately if needed


def make_strict_dale_exact_row_balance(
    N: int,
    g: float = 1.7,
    fE: float = 0.8,
    p: float = 1.0,
    w_scale: float = 1.0,          # base magnitude scale before g/sqrt(N)
    eps: float = 1e-8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Strict Dale: E columns >=0, I columns <=0.
    Exact row balance: for each postsynaptic neuron i,
        sum_j J[i,j] = 0   (up to eps)
    achieved by row-wise scaling of inhibitory inputs.

    Caveat: row-wise scaling makes inhibitory strengths depend on postsynaptic neuron,
    which is biologically plausible (different I synapses onto different targets),
    but it's not a single global inhibitory gain anymore.
    """

    assert 0.0 < fE < 1.0
    NE = int(round(fE * N))
    NI = N - NE

    # assign types (columns)
    is_E = torch.zeros(N, device=device, dtype=torch.bool)
    is_E[:NE] = True
    perm = torch.randperm(N, device=device)
    is_E = is_E[perm]
    is_I = ~is_E

    E_cols = torch.nonzero(is_E, as_tuple=False).squeeze(1)
    I_cols = torch.nonzero(is_I, as_tuple=False).squeeze(1)

    # sparsity
    if p < 1.0:
        mask = (torch.rand(N, N, device=device) < p).to(dtype)
    else:
        mask = torch.ones(N, N, device=device, dtype=dtype)

    # sample positive magnitudes
    J = torch.zeros(N, N, device=device, dtype=dtype)
    J[:, E_cols] = torch.relu(torch.randn(N, NE, device=device, dtype=dtype)) * w_scale
    J[:, I_cols] = -torch.relu(torch.randn(N, NI, device=device, dtype=dtype)) * w_scale
    J = J * mask

    # exact row balance by scaling inhibitory part per row
    JE = J[:, E_cols]                    # >= 0
    JI = J[:, I_cols]                    # <= 0
    sumE = JE.sum(dim=1, keepdim=True)   # (N,1) >= 0
    sumI = (-JI).sum(dim=1, keepdim=True)  # magnitude of inhibition, (N,1) >= 0

    # scale inhibitory magnitudes so sumE == sumI per row
    alpha = sumE / (sumI + eps)          # (N,1)
    J[:, I_cols] = -alpha * (-J[:, I_cols])

    # final scale
    J = (g / math.sqrt(N)) * J

    return J ##, is_E

def make_dense_dale_exact_row_balance_J(
    N: int,
    K: float,
    g: float = 1.0,
    fE: float = 0.8,
    sigma_E: float = 1.0,
    sigma_I: float = 1.0,
    w_scale: float = 1.0,
    eps: float = 1e-8,
    device="cpu",
    dtype=torch.float32,
):
    """
    Dense Dale EI matrix with exact row-balance: sum_j J[i,j] = 0 for every i.
    K is ONLY used in 1/sqrt(K) scaling (strong coupling).
    """
    assert 0 < fE < 1
    NE = int(round(fE * N))
    NI = N - NE

    is_E = torch.zeros(N, device=device, dtype=torch.bool)
    is_E[:NE] = True
    perm = torch.randperm(N, device=device)
    is_E = is_E[perm]
    is_I = ~is_E

    E_cols = torch.nonzero(is_E).squeeze(1)
    I_cols = torch.nonzero(is_I).squeeze(1)

    J = torch.zeros(N, N, device=device, dtype=dtype)

    # positive magnitudes
    JE = torch.relu(sigma_E * torch.randn(N, NE, device=device, dtype=dtype)) * w_scale
    JI = torch.relu(sigma_I * torch.randn(N, NI, device=device, dtype=dtype)) * w_scale

    J[:, E_cols] = +JE
    J[:, I_cols] = -JI

    # exact row balance by row-wise scaling of inhibitory magnitudes
    sumE = J[:, E_cols].sum(dim=1, keepdim=True)          # (N,1) >=0
    sumI = (-J[:, I_cols]).sum(dim=1, keepdim=True)       # (N,1) >=0
    alpha = sumE / (sumI + eps)
    J[:, I_cols] = -alpha * (-J[:, I_cols])

    # strong coupling scaling
    J = (g / math.sqrt(K)) * J

    return J, is_E



if __name__ == "__main__":
    # --- Setup for training ---
    ll = 23
    N = ll**2
    T = 500
    output_dim = 1
    device = 'cpu'

    delay_period = 250
    cue_interval = 20

    # Input patterns
    # --- Fixed input patterns (critical for generalization) ---
    G1 = gabor2d(ll, f=5 * 0.5, theta=np.deg2rad(30), gamma=0.1, phi=0.5, normalize=True).astype(np.float32)
    G2 = gabor2d(ll, f=1 * 0.5, theta=np.deg2rad(60), gamma=0.1, phi=0.5, normalize=True).astype(np.float32)
    G3 = gabor2d(ll, f=3 * 0.5, theta=np.deg2rad(90), gamma=0.1, phi=0.5, normalize=True).astype(np.float32)

    inpt_patterns = (
        G1.reshape(-1) * 0.1,  # trigger class 0
        G2.reshape(-1) * 0.1,  # trigger class 1
        (np.random.randn(ll, ll).astype(np.float32).reshape(-1) * 0.1),  # go cue (shared)
    )

    ### make random RNN, with row balance
    Jij = torch.randn(N, N, device=device, dtype=torch.float32) * (1.5 / math.sqrt(N)) #1.7
    ### row balance, (testing)
    # Jij = Jij - Jij.mean(dim=1, keepdim=True)
    ### may still explod ###

    # will need to implement the true EI network
    ########################
    # Jij = make_strict_dale_exact_row_balance(N=N, g=1., fE=0.8, device=device)
    # K = 4
    # Jij = Jij*np.sqrt(K)  # strong coupling scaling
    # inpt_patterns = tuple(pat * np.sqrt(K) for pat in inpt_patterns)
    # Jij = make_dense_dale_exact_row_balance_J(N=N, K=N//1, g=1., fE=0.8, sigma_E=1.0, sigma_I=4.0, device=device)[0]
    ########################

    # Jij = Jij.T
    # print(Jij.sum(dim=0))  # should be close to zero for all rows
    plt.figure()
    plt.imshow(Jij.cpu().numpy())
    plt.show()

    reluRNN_params = {
        'dt': 0.001,
        'tau': 0.005,
        'J0': Jij, ### tune this! #35 for Relu
        'u': 0,
        'fb_gain': 0.0,  # OFF for ridge training
    }
    model = WM_ReservoirRNN(N, T, output_dim, device, reluRNN_params)

    # Trial function
    trial_fn = lambda lr=None: make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, 
                                             lr_trial=lr, inpt_patterns=inpt_patterns)

    # Time selectors for ridge training
    delay_start = cue_interval
    delay_end = cue_interval + delay_period
    go_start = delay_end

    def time_selector_mem(T_):
        # train mem on delay + post-go
        return np.arange(delay_start, T_, dtype=np.int64)

    def time_selector_out(T_):
        # train out on post-go only (where target_out is informative)
        return np.arange(go_start, T_, dtype=np.int64)

    # Collect training data
    n_train = 50
    lam = 1e-2

    print("Collecting training data...")
    X_mem, _, Y_mem = collect_xy(model, trial_fn, n_train, time_selector_mem)
    X_out, Y_out, _ = collect_xy(model, trial_fn, n_train, time_selector_out)

    # Solve ridge regression
    print("Solving ridge regression...")
    Wmem = ridge_solve(X_mem, Y_mem, lam)          # (N+1, 1)
    Wout = ridge_solve(X_out, Y_out, lam)          # (N+1, 1)

    # Assign into model
    set_readouts_from_ridge(model, Wout, Wmem)

    # Evaluate
    mask = np.zeros((T,), dtype=bool)
    mask[go_start:] = True
    lo, lm = eval_model(model, trial_fn, n_trials=10, mask=mask)
    print(f"Ridge eval MSE: out={lo:.4g}, mem={lm:.4g}")

    # %% Visualize a couple of trials
    n_trials = 10
    plt.figure()
    for ii in range(n_trials):
        # lr=0 trials (red)
        target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=0, inpt_patterns=inpt_patterns)
        output, mem, _ = model(space_stim)
        plt.subplot(2,1,1)
        plt.plot(output.detach().cpu().numpy().squeeze(), 'r--')
        plt.plot(target_out.cpu().numpy().squeeze(), 'r')
        plt.subplot(2,1,2)
        plt.plot(mem.detach().cpu().numpy().squeeze(), 'r--')
        plt.plot(target_mem.cpu().numpy().squeeze(), 'r')

        # lr=1 trials (blue)
        target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=1, inpt_patterns=inpt_patterns)
        output, mem, rt = model(space_stim)
        plt.subplot(2,1,1)
        plt.plot(output.detach().cpu().numpy().squeeze(), 'b--')
        plt.plot(target_out.cpu().numpy().squeeze(), 'b')
        plt.subplot(2,1,2)
        plt.plot(mem.detach().cpu().numpy().squeeze(), 'b--')
        plt.plot(target_mem.cpu().numpy().squeeze(), 'b')

    ### add labels
    plt.subplot(2,1,1)
    plt.title('Output Readout vs Target')
    plt.ylabel('Output')
    plt.legend(['Output', 'Target'])

    plt.subplot(2,1,2)
    plt.title('Memory Readout vs Target')
    plt.ylabel('Memory')
    plt.xlabel('Time')
    plt.legend(['Memory', 'Target'])
    plt.show()


    ### PCA analysis of re_all
    X = rt.detach().cpu().numpy().reshape(N, T).T  # (T, N)
    Xc = X - X.mean(axis=0, keepdims=True)  # center per neuron
    C = np.cov(Xc, rowvar=False, bias=False)  # (N, N)
    U, s, Vt = np.linalg.svd(C, full_matrices=False)  # s are eigenvalues (variances)
    var_ratio = s / s.sum()
    plt.figure()
    plt.plot(np.cumsum(var_ratio))
    plt.xlabel("Number of PCs")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("PCA of RNN States")
    plt.grid(True)
    plt.show()
