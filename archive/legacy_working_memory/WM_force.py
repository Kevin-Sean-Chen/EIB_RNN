"""
2D Spatial Reservoir + Working Memory Task
==========================================

This script implements a *two-stage* training pipeline:

Stage A (open-loop / no feedback):
  - Reservoir dynamics are fixed.
  - Train readouts W_out, W_mem by offline ridge regression.

Stage B (closed-loop / feedback ON):
  - Enable feedback into excitatory current.
  - Update readouts online using FORCE / RLS (recursive least squares),
    which is stable for feedback learning (no BPTT).

Key design choices:
- Feedback projection maps W_fb_o, W_fb_m are FIXED random spatial patterns.
- Only readouts (W_out, W_mem) and biases are trained (offline ridge + online RLS).
- Uses a bias-augmented feature vector [r; 1].

We can later extend this to learn feedback maps by low-rank parameterizations
(e.g., learn coefficients over a basis of spatial patterns)

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

### image and functional IO
import sys
from pathlib import Path
from PIL import Image
import os
import datetime

# --- repo import (as in your file) ---
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
from scripts.relu2D_disorder import gabor2d
from scripts.WM_res import make_wm_trial, ridge_solve, collect_xy, eval_model

# -----------------------------
# 2D Gaussian kernels (periodic)
# -----------------------------
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


# ----------------------------
# 2D Spatial Reservoir (E/I)
# ----------------------------
class Relu2DSpatialReservoir(nn.Module):
    """
    State:
      re, ri: (B,1,N,N)

    Input:
      stim: (N,N,T) or (B,N,N,T)  (added to excitatory current)

    Readouts (trained):
      out(t) = r(t)^T W_out + b_out
      mem(t) = r(t)^T W_mem + b_mem
      where r(t) = flatten(phi(re_t)) in R^D, D=N*N

    Feedback (optional):
      mue += fb_gain * ( W_fb_o * out_scalar + W_fb_m * mem_scalar )
      where out_scalar and mem_scalar are scalars (batch-wise) derived from readouts.
    """

    def __init__(self, N, T, output_dim, device, params):
        super().__init__()
        self.N = int(N)
        self.T = int(T)
        self.output_dim = int(output_dim)
        self.device = torch.device(device)

        self.dt = float(params["dt"])
        tau = params["tau"]
        self.tau_e = float(tau[0])
        self.tau_i = float(tau[1])

        self.K = float(params["K"])
        self.sqrtK = self.K ** 0.5

        J0 = torch.tensor(params["J0"], device=self.device, dtype=torch.float32)
        self.register_buffer("J0", J0)

        u = params["u"]
        self.u0_e = float(u[0])
        self.u0_i = float(u[1])

        sigma = params["sigma"]
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

        # Feature dimension
        self.D = self.N * self.N

        # Trainable readouts (we will set via ridge + RLS; no BPTT)
        self.W_out = nn.Parameter(torch.zeros(self.D, self.output_dim, device=self.device))
        self.W_mem = nn.Parameter(torch.zeros(self.D, 1, device=self.device))
        self.register_buffer("b_out", torch.zeros(1, self.output_dim, device=self.device))
        self.register_buffer("b_mem", torch.zeros(1, 1, device=self.device))

        # Feedback projection maps (fixed)
        self.fb_gain = float(params.get("fb_gain", 0.0))
        fb_scale = float(params.get("fb_scale", 0.01))
        if self.fb_gain != 0.0:
            self.register_buffer("W_fb_o", torch.randn(1, 1, self.N, self.N, device=self.device) * fb_scale)
            self.register_buffer("W_fb_m", torch.randn(1, 1, self.N, self.N, device=self.device) * fb_scale)
        else:
            self.W_fb_o = None
            self.W_fb_m = None

        # Init + microsteps
        self.init_scale = float(params.get("init_scale", 0.1))
        self.npf = int(params.get("npf", 1))

        # Input gain
        self.stim_gain = float(params.get("stim_gain", 1.0))

        # Whether to include OUT and/or MEM in feedback
        self.use_fb_out = bool(params.get("use_fb_out", True))
        self.use_fb_mem = bool(params.get("use_fb_mem", True))

    def NL(self, x):
        return F.relu(x) if self.nl == "relu" else torch.tanh(x)

    def _conv_circular(self, x, kernel):
        xP = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return F.conv2d(xP, kernel)

    @torch.no_grad()
    def rnn_step(self, re, ri, stim_t):
        """
        One time step (npf microsteps) of E/I spatial dynamics.
        re, ri, stim_t: (B,1,N,N)
        """
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

            # ---- CLAMP CURRENTS (most important) ----
            mue = torch.clamp(mue, -100, 100)
            mui = torch.clamp(mui, -100, 100)

            # Feedback (scalar -> spatial pattern), if enabled
            if self.fb_gain != 0.0:
                phi_re = self.NL(re)                       # (B,1,N,N)
                rflat = phi_re.flatten(start_dim=1)        # (B,D)

                out_vec = (rflat @ self.W_out) + self.b_out  # (B, output_dim)
                mem_vec = (rflat @ self.W_mem) + self.b_mem  # (B, 1)

                # turn output_dim into a scalar per batch element (sum; you can change to select component)
                out_scalar = out_vec.sum(dim=1, keepdim=True)  # (B,1)
                mem_scalar = mem_vec                            # (B,1)

                feedback = 0.0
                if self.use_fb_out:
                    feedback = feedback + self.W_fb_o * out_scalar[:, None, None]
                if self.use_fb_mem:
                    feedback = feedback + self.W_fb_m * mem_scalar[:, None, None]

                mue = mue + self.fb_gain * feedback

            re = re + (self.dt / self.tau_e) * (-re + self.NL(mue))
            ri = ri + (self.dt / self.tau_i) * (-ri + self.NL(mui))

        return re, ri

    @torch.no_grad()
    def forward(self, stim):
        """
        stim: (N,N,T) or (B,N,N,T)
        Returns:
          out: (B,output_dim,T)
          mem: (B,1,T)
          re_all: (B,N,N,T) storing phi(re)
        """
        stim = stim.to(self.device).float()
        if stim.dim() == 3:
            stim = stim.unsqueeze(0)  # (1,N,N,T)
        B, N1, N2, T = stim.shape
        assert (N1, N2, T) == (self.N, self.N, self.T)

        re = torch.randn(B, 1, self.N, self.N, device=self.device) * self.init_scale
        ri = torch.randn(B, 1, self.N, self.N, device=self.device) * self.init_scale

        re_list, out_list, mem_list = [], [], []

        for t in range(self.T):
            stim_t = stim[:, :, :, t].unsqueeze(1)  # (B,1,N,N)
            re, ri = self.rnn_step(re, ri, stim_t)

            phi_re = self.NL(re)                     # (B,1,N,N)
            re_list.append(phi_re.squeeze(1))        # (B,N,N)

            rflat = phi_re.flatten(start_dim=1)      # (B,D)
            out_t = (rflat @ self.W_out) + self.b_out
            mem_t = (rflat @ self.W_mem) + self.b_mem

            out_list.append(out_t)
            mem_list.append(mem_t)

        re_all = torch.stack(re_list, dim=-1)        # (B,N,N,T)
        out = torch.stack(out_list, dim=-1)          # (B,output_dim,T)
        mem = torch.stack(mem_list, dim=-1)          # (B,1,T)
        return out, mem, re_all


def set_readouts_from_aug(model, Wout_aug, Wmem_aug):
    """
    Wout_aug: (D+1, output_dim)
    Wmem_aug: (D+1, 1)
    """
    D = model.D
    with torch.no_grad():
        model.W_out.copy_(torch.tensor(Wout_aug[:D, :], dtype=torch.float32, device=model.device))
        model.b_out.copy_(torch.tensor(Wout_aug[D:D+1, :], dtype=torch.float32, device=model.device))
        model.W_mem.copy_(torch.tensor(Wmem_aug[:D, :], dtype=torch.float32, device=model.device))
        model.b_mem.copy_(torch.tensor(Wmem_aug[D:D+1, :], dtype=torch.float32, device=model.device))


# -------------------------
# FORCE / RLS (online)
# -------------------------
def rls_init(P_dim, delta=1e-2, device="cpu"):
    """
    Initialize inverse correlation estimate:
      P0 = I / delta
    delta small -> large initial uncertainty -> larger early updates.
    """
    return (1.0 / delta) * torch.eye(P_dim, device=device)


@torch.no_grad()
def rls_update(W, P, r, e, lam=1.0):
    """
    RLS update for multi-output.
    W: (P_dim, Q)
    P: (P_dim, P_dim)
    r: (P_dim,)
    e: (Q,)  where e = y_hat - y_target
    lam: forgetting factor (1.0 = no forgetting)

    k = P r / (lam + r^T P r)
    W <- W - k e^T
    P <- (P - k (r^T P)) / lam
    """
    Pr = P @ r
    denom = (lam + (r @ Pr)).clamp_min(1e-12)
    k = Pr / denom  # (P_dim,)

    W = W - torch.outer(k, e)                 # (P_dim,Q)
    P = (P - torch.outer(k, (r @ P))) / lam   # (P_dim,P_dim)
    return W, P


def force_train_closed_loop(
    model,
    trial_fn,
    n_trials=200,
    lam=1.0,
    delta=1e-2,
    fb_gain=0.1,
    train_out=True,
    train_mem=True,
    device="cpu",
    report_every=10,
):
    """
    Online FORCE/RLS training with feedback ON.
    Updates augmented readout weights [W; b] so feedback uses updated predictions.

    Returns:
      mse_out_hist, mse_mem_hist
    """
    model.to(device)
    model.train(False)

    # Enable feedback
    model.fb_gain = float(fb_gain)
    if model.fb_gain != 0.0 and (model.W_fb_o is None or model.W_fb_m is None):
        raise RuntimeError("fb_gain>0 but feedback maps not initialized. Set fb_gain>0 in params at init.")

    Dp1 = model.D + 1
    P_out = rls_init(Dp1, delta=delta, device=device)
    P_mem = rls_init(Dp1, delta=delta, device=device)

    # Initialize augmented weights from current model values
    with torch.no_grad():
        Wout_aug = torch.cat([model.W_out.detach(), model.b_out.detach()], dim=0).clone()  # (D+1,output_dim)
        Wmem_aug = torch.cat([model.W_mem.detach(), model.b_mem.detach()], dim=0).clone()  # (D+1,1)

    mse_out_hist, mse_mem_hist = [], []

    for it in range(n_trials):
        target_out, target_mem, stim, lr = trial_fn()
        stim = stim.to(device)
        target_out = target_out.to(device)
        target_mem = target_mem.to(device)

        # init state for each trial
        re = torch.randn(1, 1, model.N, model.N, device=device) * model.init_scale
        ri = torch.randn(1, 1, model.N, model.N, device=device) * model.init_scale

        se_out = 0.0
        se_mem = 0.0

        for t in range(model.T):
            stim_t = stim[:, :, t].unsqueeze(0).unsqueeze(0)  # (1,1,N,N)

            # --- one reservoir update (this will use model.W_* and feedback) ---
            re, ri = model.rnn_step(re, ri, stim_t)

            # --- features ---
            phi_re = model.NL(re)                           # (1,1,N,N)
            rflat = phi_re.flatten(start_dim=1).squeeze(0)  # (D,)
            r = torch.cat([rflat, torch.ones(1, device=device)], dim=0)  # (D+1,)

            # --- predictions from augmented weights (not necessarily identical to model.* if you don't sync) ---
            y_hat = (r @ Wout_aug).view(-1)  # (output_dim,)
            m_hat = (r @ Wmem_aug).view(-1)  # (1,)

            y_t = target_out[t].view(-1)     # (1,)
            m_t = target_mem[t].view(-1)     # (1,)

            e_out = (y_hat - y_t)            # (output_dim,)
            e_mem = (m_hat - m_t)            # (1,)

            se_out += float((e_out ** 2).mean().item())
            se_mem += float((e_mem ** 2).mean().item())

            # --- RLS updates ---
            if train_out:
                Wout_aug, P_out = rls_update(Wout_aug, P_out, r, e_out, lam=lam)
            if train_mem:
                Wmem_aug, P_mem = rls_update(Wmem_aug, P_mem, r, e_mem, lam=lam)

            # --- sync updated weights back into model so feedback uses them next step ---
            set_readouts_from_aug(model, Wout_aug, Wmem_aug)

        mse_out_hist.append(se_out / model.T)
        mse_mem_hist.append(se_mem / model.T)

        if report_every and ((it + 1) % report_every == 0):
            print(
                f"[FORCE] trial {it+1}/{n_trials} | "
                f"MSE_out={mse_out_hist[-1]:.4g} | MSE_mem={mse_mem_hist[-1]:.4g} | "
                f"fb_gain={model.fb_gain}"
            )

    return mse_out_hist, mse_mem_hist


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # --- Setup ---
    device = "cpu"
    torch.set_default_dtype(torch.float32)
    # np.random.seed(0)
    # torch.manual_seed(0)

    N = 23
    T = 500
    output_dim = 1

    delay_period = 250
    cue_interval = 20

    # --- Fixed input patterns (critical for generalization) ---
    G1 = gabor2d(N, f=5 * 0.5, theta=np.deg2rad(30), gamma=0.1, phi=0.5, normalize=True).astype(np.float32)
    G2 = gabor2d(N, f=1 * 0.5, theta=np.deg2rad(60), gamma=0.1, phi=0.5, normalize=True).astype(np.float32)
    G3 = gabor2d(N, f=3 * 0.5, theta=np.deg2rad(90), gamma=0.1, phi=0.5, normalize=True).astype(np.float32)

    ipt_patterns = (
        G1 * 0.1,  # trigger class 0
        G2 * 0.1,  # trigger class 1
        (np.random.randn(N, N).astype(np.float32) * 0.1),  # go cue (shared)
    )

    # --- Reservoir params ---
    params = {
        "dt": 0.001,
        "K": 20.0,  # gain / spatial scaling
        "tau": np.array([0.01, 0.01]),
        "u": [10.0, 0.0],  # baseline drives; if learning is weak, try reducing u0_e
        "J0": np.array([[1, -4], [2, -2]]),
        "sigma": 0.05 * np.array([1, np.sqrt(2)]),
        "nl": "relu",
        "stim_gain": 10.0,
        "init_scale": 0.1,
        "npf": 1,

        # Feedback configuration (we will toggle fb_gain later)
        "fb_gain": 0.0,       # Stage A uses 0; Stage B sets >0
        "fb_scale": 0.01,     # spatial feedback map scale
        "use_fb_out": True,
        "use_fb_mem": True,
    }

    model = Relu2DSpatialReservoir(N, T, output_dim, device, params)

    # Trial factory
    def trial_fn(lr=None):
        return make_wm_trial(
            N, T,
            delay_period=delay_period,
            cue_interval=cue_interval,
            lr_trial=lr,
            inpt_patterns=ipt_patterns,
            ramp_mem=False,
        )

    # -------------------------
    # Stage A: Offline ridge (fb_gain=0)
    # -------------------------
    delay_start = cue_interval
    delay_end = cue_interval + delay_period
    go_start = delay_end

    def time_selector_mem(T_):
        # train mem during delay + after
        return np.arange(delay_start, T_, dtype=np.int64)

    def time_selector_out(T_):
        # train out only after go cue
        return np.arange(go_start, T_, dtype=np.int64)

    n_train = 30          # you can increase (e.g., 200-1000)
    lam_ridge = 1e-2      # sweep 1e-6 ... 1e0 if needed

    # collect and solve
    X_mem, _, Y_mem = collect_xy(model, lambda: trial_fn(lr=None), n_train, time_selector_mem)
    X_out, Y_out, _ = collect_xy(model, lambda: trial_fn(lr=None), n_train, time_selector_out)

    Wmem_aug = ridge_solve(X_mem, Y_mem, lam_ridge)  # (D+1,1)
    Wout_aug = ridge_solve(X_out, Y_out, lam_ridge)  # (D+1,1)

    set_readouts_from_aug(model, torch.tensor(Wout_aug).float(), torch.tensor(Wmem_aug).float())

    lo, lm = eval_model(model, lambda: trial_fn(lr=None), n_trials=10)
    print(f"[Ridge] eval MSE: out={lo:.4g}, mem={lm:.4g}")

    # -------------------------
    # Stage B: Online FORCE / RLS with feedback ON
    # -------------------------
    # NOTE: feedback maps W_fb_* were initialized at model init only if fb_gain!=0.
    # We initialized with fb_gain=0.0 above, so we need to create maps now if we want feedback.
    # Easiest: set fb_gain in params nonzero at init, but keep it off during Stage A by setting model.fb_gain=0.
    # We'll do that here by re-initializing feedback maps if absent.

    if model.W_fb_o is None or model.W_fb_m is None:
        # Create fixed maps now (same distribution as init)
        with torch.no_grad():
            fb_scale = float(params.get("fb_scale", 0.01))
            delattr(model, "W_fb_o")
            delattr(model, "W_fb_m")
            model.register_buffer("W_fb_o", torch.randn(1, 1, N, N, device=model.device) * fb_scale)
            model.register_buffer("W_fb_m", torch.randn(1, 1, N, N, device=model.device) * fb_scale)

    # FORCE hyperparams
    force_trials = 10
    lam_forget = 1.0       # 1.0 = standard (no forgetting)
    delta = 1e-1           # smaller -> larger early updates
    fb_gain = 0.001          # start small; increase gradually if stable (0.05 -> 0.5)

    mse_out_hist, mse_mem_hist = force_train_closed_loop(
        model,
        lambda: trial_fn(lr=None),
        n_trials=force_trials,
        lam=lam_forget,
        delta=delta,
        fb_gain=fb_gain,
        train_out=True,
        train_mem=True,
        device=device,
        report_every=10,
    )

    lo2, lm2 = eval_model(model, lambda: trial_fn(lr=None), n_trials=20)
    print(f"[FORCE] eval MSE: out={lo2:.4g}, mem={lm2:.4g}")

    # -------------------------
    # Plot training curves
    # -------------------------
    plt.figure()
    plt.plot(mse_out_hist, label="FORCE MSE out")
    plt.plot(mse_mem_hist, label="FORCE MSE mem")
    plt.yscale("log")
    plt.xlabel("trial")
    plt.ylabel("MSE (log)")
    plt.legend()
    plt.title("FORCE learning curves")
    plt.show()

    # -------------------------
    # Visualize a few trials (lr=0 and lr=1)
    # -------------------------
    plt.figure(figsize=(10, 6))
    for i in range(5):
        target_out, target_mem, stim, lr = trial_fn(lr=0)
        out, mem, re_0 = model(stim)
        plt.subplot(2, 1, 1)
        plt.plot(out.squeeze().cpu().numpy(), "r--", alpha=0.6)
        plt.plot(target_out.cpu().numpy(), "r", alpha=0.6)
        plt.subplot(2, 1, 2)
        plt.plot(mem.squeeze().cpu().numpy(), "r--", alpha=0.6)
        plt.plot(target_mem.cpu().numpy(), "r", alpha=0.6)

        target_out, target_mem, stim, lr = trial_fn(lr=1)
        out, mem, re_1 = model(stim)
        plt.subplot(2, 1, 1)
        plt.plot(out.squeeze().cpu().numpy(), "b--", alpha=0.6)
        plt.plot(target_out.cpu().numpy(), "b", alpha=0.6)
        plt.subplot(2, 1, 2)
        plt.plot(mem.squeeze().cpu().numpy(), "b--", alpha=0.6)
        plt.plot(target_mem.cpu().numpy(), "b", alpha=0.6)

    plt.subplot(2, 1, 1)
    plt.title("Output: dashed=model, solid=target (red lr=0, blue lr=1)")
    plt.ylabel("out")

    plt.subplot(2, 1, 2)
    plt.title("Memory: dashed=model, solid=target")
    plt.ylabel("mem")
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()


    # %% inspect trained activity re_0 and re_1 during the task
    # --- Save animation using tif ---
    # Example: Create dummy 3D data
    SAVE = False  # set True to save; False to skip (and avoid dependency on PIL)
    if SAVE is True:
        re_0, re_1 = re_0.cpu().numpy().squeeze(0), re_1.cpu().numpy().squeeze(0)  # (N,N,T)
        data = re_0*1  # Use re_all as the data to visualize
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
            # Create figure with two subplots (re_0 and re_1)
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
            ax0.imshow(re_0[:, :, idx], cmap='viridis')
            ax0.set_title(f'lr=0 (t={idx})')
            ax0.axis('off')
            ax1.imshow(re_1[:, :, idx], cmap='viridis')
            ax1.set_title(f'lr=1 (t={idx})')
            ax1.axis('off')
            fig.canvas.draw()
            
            rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
            image = rgba[:, :, :3]
            frames.append(Image.fromarray(image))
            plt.close(fig)
            # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            ### Get RGBA buffer, convert to RGB
            rgba = np.asarray(fig.canvas.renderer.buffer_rgba())  # shape (H, W, 4)
            image = rgba[:, :, :3]  # drop alpha channel
            frames.append(Image.fromarray(image))
            plt.close(fig)

        # Save as GIF
        os.makedirs('video', exist_ok=True)
        filename = f"2D_forced_WM_{datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')}"
        output_path = os.path.join('video', filename + '.gif')
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        # Display in notebook (optional)
        from IPython.display import display, Image as IPImage
        display(IPImage(filename=output_path))
