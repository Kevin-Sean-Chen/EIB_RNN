"""
Reservoir computing (ESN-style) training for your 2D spatial reservoir:
- Reservoir dynamics are fixed without back-propagation through time.
- Train readouts W_out, W_mem by ridge regression; this is closed-form, without iterative update yet.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

### import WM task for space
import sys
from pathlib import Path
# add repo root to sys.path so `scripts` files can import each other
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

# from scripts.WM_task import make_wm_trial
from scripts.relu2D_disorder import gabor2d

# -----------------------------
# 2D Gaussian kernels
# -----------------------------
### old, with explicit normalization ###
def _make_1d_periodic_gaussian_weights(N: int, sigma: float, device, dtype):
    x = torch.arange(-(N - 1) // 2, (N - 1) // 2 + 1, device=device, dtype=dtype)
    dx = 1.0 / N
    wrap = int(math.ceil(10.0 * float(sigma))) if sigma > 0 else 0
    k = torch.arange(-wrap, wrap + 1, device=device, dtype=dtype)
    dist = dx * (x[:, None] + k[None, :])
    gauss = torch.exp(-0.5 * (dist / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)
    w = (dx * gauss).sum(dim=1)
    w = w / (w.sum() + 1e-12)
    return w

### New, without explicit normalization (but more like old relu code...) ###
# def _make_1d_periodic_gaussian_weights(N: int, sigma: float, device, dtype):
#     """
#     Discrete periodic Gaussian weights on a 1D ring of length N.
#     Matches the reference implementation from relu2D_step().

#     Returns w: (N,) such that w[i] depends on periodic distance from center.
#     Uses summation over integer wraps to approximate periodic Gaussian.
#     """
#     dx = 1.0 / N
#     x = np.arange(-(N-1)//2, (N-1)//2 + 1)
#     k = np.arange(-int(np.ceil(10 * sigma)), int(np.ceil(10 * sigma)) + 1)
    
#     # Compute periodic Gaussian (no normalization - matches reference)
#     w = np.sum(
#         dx * (2 * np.pi * sigma**2)**-0.5 *
#         np.exp(-0.5 * (dx * (x[:, None] + k))**2 / sigma**2),
#         axis=1
#     )
    
#     w = torch.tensor(w, dtype=dtype, device=device)
#     return w

def _make_2d_separable_kernel(N: int, sigma: float, device, dtype):
    w = _make_1d_periodic_gaussian_weights(N, sigma, device=device, dtype=dtype)
    K2 = torch.outer(w, w)
    return K2[None, None, :, :]  # (1,1,N,N)

# -------------------------------------
# Reservoir model (OG one that only has readout trained offline)
# -------------------------------------
class Relu2DSpatialReservoir(nn.Module):
    def __init__(self, N, T, output_dim, device, params, readout_masked=False, local_dim=10):
        super().__init__()
        self.N = int(N)
        self.T = int(T)
        self.output_dim = int(output_dim)
        self.device = torch.device(device)

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

        Ke = _make_2d_separable_kernel(self.N, self.sigma_e, self.device, torch.float32)
        Ki = _make_2d_separable_kernel(self.N, self.sigma_i, self.device, torch.float32)
        self.register_buffer("Ke", Ke)
        self.register_buffer("Ki", Ki)

        self.pad = int(self.Ke.shape[-1] // 2)

        self.nl = params.get("nl", "relu")
        if self.nl not in ("relu", "tanh"):
            raise ValueError("params['nl'] must be 'relu' or 'tanh'")

        # Full-neuron dimension
        self.D = self.N * self.N

        # Readout weights along with biases(will be set by ridge, not backprop; keep as Parameters for convenience)
        if readout_masked is False:
            twoD_mask = torch.ones(self.N, self.N, device=self.device)
        else:
            twoD_mask = torch.zeros(self.N, self.N, device=self.device)
            center = self.N // 2
            half_local = local_dim // 2
            twoD_mask[center - half_local:center + half_local + 1, center - half_local:center + half_local + 1] = 1.0
        self.twoD_mask = twoD_mask
        self.W_out = nn.Parameter(torch.zeros(self.D, self.output_dim, device=self.device))
        self.W_mem = nn.Parameter(torch.zeros(self.D, 1, device=self.device))
        self.register_buffer("b_out", torch.zeros(1, self.output_dim, device=self.device))
        self.register_buffer("b_mem", torch.zeros(1, 1, device=self.device))

        # Feedback optional (keep off for classic reservoir training; can be used later with iterative update methods)
        self.fb_gain = float(params.get('fb_gain', 0.0))
        if self.fb_gain != 0.0:
            self.register_buffer("W_fb_o", torch.randn(1, 1, self.N, self.N, device=self.device) * .01)
            self.register_buffer("W_fb_m", torch.randn(1, 1, self.N, self.N, device=self.device) * .01)
        else:
            self.W_fb_o = None
            self.W_fb_m = None

        # Initial states and iterations
        self.init_scale = float(params.get("init_scale", 0.1))
        self.npf = int(params.get("npf", 1))

        # Optional stim gain (so stimulus can compete with baseline)
        self.stim_gain = float(params.get("stim_gain", 1.0))

    def NL(self, x):
        return F.relu(x) if self.nl == "relu" else torch.tanh(x)

    def _conv_circular(self, x, kernel):
        xP = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return F.conv2d(xP, kernel)

    def rnn_step(self, re, ri, stim_t):
        for _ in range(self.npf):
            conv_re_e = self._conv_circular(re, self.Ke)
            conv_ri_i = self._conv_circular(ri, self.Ki)

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

            # Optional feedback (typically OFF for ridge training)
            if self.fb_gain != 0.0:
                phi_re = self.NL(re*self.twoD_mask)
                rflat = phi_re.flatten(start_dim=1)  # (B, D)
                out_scalar = (rflat @ self.W_out).sum(dim=1, keepdim=True)  # (B,1)
                mem_scalar = (rflat @ self.W_mem).view(-1, 1)              # (B,1)
                feedback = (
                    self.W_fb_o * out_scalar[:, None, None]*1   ############### testing for now
                    + self.W_fb_m * mem_scalar[:, None, None]*1  ############### testing for now
                )
                mue = mue + self.fb_gain * feedback

            re = re + (self.dt / self.tau_e) * (-re + self.NL(mue))
            ri = ri + (self.dt / self.tau_i) * (-ri + self.NL(mui))

        return re, ri

    @torch.no_grad()
    def forward(self, stim):
        stim = stim.to(self.device).float()
        if stim.dim() == 3:
            stim = stim.unsqueeze(0)  # (1,N,N,T)
        B, N1, N2, T = stim.shape
        # Provide a clearer error message than a bare assert so callers can debug shape/order issues.
        if not (N1 == self.N and N2 == self.N and T == self.T):
            raise ValueError(
                f"Stim shape mismatch: received (B,N1,N2,T)=({B},{N1},{N2},{T}), "
                f"expected (B,N,N,T)=({B},{self.N},{self.N},{self.T}).\n"
                "Common causes: you passed stim with axes (T,N,N) or "
                "(N*N, T) flattened; or the model was constructed with the wrong N/T.\n"
                "Check your trial generator and the model initialization."
            )

        re = torch.randn(B, 1, self.N, self.N, device=self.device) * self.init_scale
        ri = torch.randn(B, 1, self.N, self.N, device=self.device) * self.init_scale

        re_list = []
        out_list = []
        mem_list = []

        for t in range(self.T):
            stim_t = stim[:, :, :, t].unsqueeze(1)  # (B,1,N,N)  # B for batch training in the future
            re, ri = self.rnn_step(re, ri, stim_t)

            phi_re = self.NL(re)  # (B,1,N,N)
            re_list.append(phi_re.squeeze(1))  # (B,N,N)

            rflat = phi_re.flatten(start_dim=1)  # (B, D)
            out_t = (rflat @ self.W_out) + self.b_out  # (B, output_dim)
            mem_t = (rflat @ self.W_mem) + self.b_mem  # (B, 1)

            out_list.append(out_t)
            mem_list.append(mem_t)

        re_all = torch.stack(re_list, dim=-1)  # (B,N,N,T)
        out = torch.stack(out_list, dim=-1)    # (B,output_dim,T)
        mem = torch.stack(mem_list, dim=-1)    # (B,1,T)
        return out, mem, re_all


# -------------------
# Task for WM
# -------------------
def make_wm_trial(N, T, delay_period=200, cue_interval=50, lr_trial=None, inpt_patterns=None, ramp_mem=False):
    if lr_trial is None:
        lr = np.random.randint(0, 2)
    else:
        lr = int(lr_trial)

    trigger_pattern1, trigger_pattern2, cue_pattern = inpt_patterns

    space_stim = np.zeros((N, N, T), dtype=np.float32)
    target_out = np.zeros((T,), dtype=np.float32)
    target_mem = np.zeros((T,), dtype=np.float32)

    out_sign = +1.0 if lr == 0 else -1.0

    for tt in range(T):
        if tt < cue_interval:
            # triggered pattern
            space_stim[:, :, tt] = trigger_pattern1 if lr == 0 else trigger_pattern2
            target_out[tt] = 0.0
            target_mem[tt] = out_sign

        elif tt < cue_interval + delay_period:
            # no input during delay
            space_stim[:, :, tt] = 0.0
            if ramp_mem: ### ramping memory target
                progress = (tt - cue_interval + 1) / float(delay_period)
                target_mem[tt] = np.clip(progress, 0.0, 1.0) * out_sign
            else: ### fixed memory target
                target_mem[tt] = out_sign
            target_out[tt] = 0.0

        elif tt < cue_interval + delay_period + cue_interval:
            # go cue (same for both)
            space_stim[:, :, tt] = cue_pattern
            start = cue_interval + delay_period
            progress = (tt - start + 1) / float(cue_interval)
            target_out[tt] = np.clip(progress, 0.0, 1.0) * out_sign
            target_mem[tt] = out_sign

        else:
            space_stim[:, :, tt] = 0.0
            target_out[tt] = out_sign
            target_mem[tt] = out_sign

    return (
        torch.tensor(target_out, dtype=torch.float32),
        torch.tensor(target_mem, dtype=torch.float32),
        torch.tensor(space_stim, dtype=torch.float32),
        lr
    )

# -------------------------
# Ridge regression utilities (GPT helping...)
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
        _, _, re_all = model(stim)          # re_all: (1,N,N,T)
        phi = re_all.squeeze(0)                 # (N,N,T)
        N, _, T = phi.shape
        D = N * N

        R = phi.reshape(D, T).T                 # (T, D)
        idx = time_selector(T)                  # indices we train on
        R = R[idx, :]                           # (Tsel, D)

        yout = target_out.view(-1, 1)[idx]
        ymem = target_mem.view(-1, 1)[idx]

        X_list.append(R.cpu())
        Yout_list.append(yout.cpu())
        Ymem_list.append(ymem.cpu())

    X = torch.cat(X_list, dim=0).numpy().astype(np.float64)      # (M, D)
    Yout = torch.cat(Yout_list, dim=0).numpy().astype(np.float64) # (M,1)
    Ymem = torch.cat(Ymem_list, dim=0).numpy().astype(np.float64) # (M,1)

    # add bias column for regression
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)  # (M, D+1)
    return Xb, Yout, Ymem

def set_readouts_from_ridge(model, Wout, Wmem):
    """
    Wout: (D+1, output_dim)
    Wmem: (D+1, 1)
    """
    D = model.D
    with torch.no_grad():
        model.W_out.copy_(torch.tensor(Wout[:D, :], dtype=torch.float32, device=model.device))
        model.b_out.copy_(torch.tensor(Wout[D:D+1, :], dtype=torch.float32, device=model.device))
        model.W_mem.copy_(torch.tensor(Wmem[:D, :], dtype=torch.float32, device=model.device))
        model.b_mem.copy_(torch.tensor(Wmem[D:D+1, :], dtype=torch.float32, device=model.device))

@torch.no_grad()
def eval_model(model, trial_fn, n_trials, mask=None):
    """
    Evaluate model MSE on output and memory readouts over n_trials.
    mask: optional 1D array/tensor of length T with non-negative weights.
          If None, compute plain mean squared error over all timepoints.
    """
    model.eval()
    # prepare mask if given
    mask_t = None
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_t = torch.tensor(mask, dtype=torch.float32)
        elif isinstance(mask, torch.Tensor):
            mask_t = mask.to(dtype=torch.float32)
        else:
            # allow lists
            mask_t = torch.tensor(np.asarray(mask), dtype=torch.float32)
        # keep on CPU for now; will move to output device per-trial
        if mask_t.ndim != 1:
            raise ValueError("mask must be a 1D array/tensor")

    lo_list, lm_list = [], []
    for _ in range(n_trials):
        target_out, target_mem, stim, lr = trial_fn()
        out, mem, _ = model(stim)
        out = out.squeeze(0).squeeze(0)  # (T,)
        mem = mem.squeeze(0).squeeze(0)  # (T,)

        T = out.shape[0]

        if mask_t is None:
            lo = ((out - target_out.to(out.device)) ** 2).mean().item()
            lm = ((mem - target_mem.to(mem.device)) ** 2).mean().item()
        else:
            if mask_t.shape[0] != T:
                raise ValueError(f"mask length ({mask_t.shape[0]}) does not match trial length ({T})")
            m = mask_t.to(out.device)
            m_sum = m.sum().item()
            if m_sum <= 0:
                raise ValueError("mask must have positive sum")
            se_out = (out - target_out.to(out.device)) ** 2
            se_mem = (mem - target_mem.to(mem.device)) ** 2
            lo = (se_out * m).sum().item() / m_sum
            lm = (se_mem * m).sum().item() / m_sum

        lo_list.append(lo)
        lm_list.append(lm)

    return float(np.mean(lo_list)), float(np.mean(lm_list))


# -------------------------
# Main: ridge training script
# -------------------------
if __name__ == "__main__":

    # --- Setup ---
    N = 23
    T = 500
    output_dim = 1
    device = "cpu"

    delay_period = 250
    cue_interval = 20

    # Patterns for trigger and cues
    G1 = gabor2d(N, f=5*.5, theta=np.deg2rad(30), gamma=0.1, phi=.5, normalize=True).astype(np.float32)
    G2 = gabor2d(N, f=1*.5, theta=np.deg2rad(60), gamma=0.1, phi=.5, normalize=True).astype(np.float32)
    G3 = gabor2d(N, f=3*.5, theta=np.deg2rad(90), gamma=0.1, phi=.5, normalize=True).astype(np.float32)

    mask_input = np.ones((N, N), dtype=np.float32)
    # mask_input[1:10, 1:10] = 1.0  # top-left corner, away from center readout mask
    ipt_patterns = (
        mask_input*G1 * 0.1,           # make cues not tiny vs baseline
        mask_input*G2 * 0.1,
        (np.random.randn(N, N).astype(np.float32) * 0.1)          # go cue
    )

    # Reservoir params
    params = {
        "dt": 0.001,
        "K": 20.0, ### 20 seems great for small network; 30 for larger, but more chaotic
        "tau": np.array([0.01, 0.01]),
        "u": [10.0, 0.0],
        "J0": np.array([[1, -4], [2, -2]]),
        "sigma": 0.05 * np.array([1, np.sqrt(2)]),
        "fb_gain": 0.0,                   # OFF for ridge ESN training
        "nl": "relu",                     # relu as usual
        "stim_gain": 10.0,                # the input strength matters
        "init_scale": 0.1,
        "npf": 1,
    }

    ### full readout
    model = Relu2DSpatialReservoir(N, T, output_dim, device, params)
    ### testing local mask
    # model = Relu2DSpatialReservoir(N, T, output_dim, device, params, readout_masked=True, local_dim=10)


    ### test to observe spontaneous activity ###
    target_out, target_mem, space_stim, input_traj = make_wm_trial(N, T, delay_period=delay_period, cue_interval=cue_interval, lr_trial=0, inpt_patterns=ipt_patterns)
    output, mem, re_all = model(space_stim*0)
    re_all = re_all.detach().squeeze()  # [N, N, T]
    plt.figure(figsize=(15, 5))
    for idx, i in enumerate([0, T//2, T-1]):
        plt.subplot(1, 3, idx + 1)  # Corrected indexing for subplot
        plt.imshow(re_all[:, :, i].detach().cpu().numpy(), cmap='gray')
        plt.title(f'Reservoir State at t={i}')
    plt.show()
    ############################################

    # Trial factory
    def trial_fn(lr=None):
        return make_wm_trial(
            N, T,
            delay_period=delay_period,
            cue_interval=cue_interval,
            lr_trial=lr,
            inpt_patterns=ipt_patterns,
            ramp_mem=False  # start with constant WM target
        )

    # Choose which timepoints to train on:
    # - Train memory readout mainly on delay (and after), and output mainly after go cue.
    delay_start = cue_interval
    delay_end = cue_interval + delay_period
    go_start = delay_end

    def time_selector_mem(T_):
        # train mem on delay + post-go
        return np.arange(delay_start, T_, dtype=np.int64)

    def time_selector_out(T_):
        # train out on post-go only (where target_out is informative)
        return np.arange(go_start, T_, dtype=np.int64)

    # Collect (separately) for out and mem (best practice)
    n_train = 30  #50
    lam = 1e-2*1  ### this matters

    # Collect for mem
    X_mem, _, Y_mem = collect_xy(model, lambda: trial_fn(lr=None), n_train, time_selector_mem)
    # Collect for out
    X_out, Y_out, _ = collect_xy(model, lambda: trial_fn(lr=None), n_train, time_selector_out)

    # Solve ridge
    Wmem = ridge_solve(X_mem, Y_mem, lam)          # (D+1, 1)
    Wout = ridge_solve(X_out, Y_out, lam)          # (D+1, 1)

    # Assign into model
    set_readouts_from_ridge(model, Wout, Wmem)

    # Evaluate
    lo, lm = eval_model(model, lambda: trial_fn(lr=None), n_trials=10)
    print(f"Ridge eval MSE: out={lo:.4g}, mem={lm:.4g}")

    # %% Visualize a couple of trials
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

    # plt.subplot(2,1,1)
    # plt.plot(input_traj.cpu().numpy(),'k')

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
