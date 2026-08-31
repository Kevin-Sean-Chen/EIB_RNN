"""
Reservoir computing (ESN-style) training for your 2D spatial reservoir:
- scanning effects of K strength
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

### import functions
from scripts.relu2D_disorder import gabor2d
from archive.legacy_working_memory.WM_res import Relu2DSpatialReservoir, make_wm_trial, collect_xy, ridge_solve, set_readouts_from_ridge, eval_model


# --- Setup ---
N = 31
T = 500
output_dim = 1
device = "cpu"

delay_period = 250
cue_interval = 20

# Collect (separately) for out and mem (best practice)
n_train = 20#30
lam = 1e-2*1  ### this matters

# Patterns for trigger and cues
G1 = gabor2d(N, f=5*.5, theta=np.deg2rad(30), gamma=0.1, phi=.5, normalize=True).astype(np.float32)
G2 = gabor2d(N, f=1*.5, theta=np.deg2rad(60), gamma=0.1, phi=.5, normalize=True).astype(np.float32)
G3 = gabor2d(N, f=3*.5, theta=np.deg2rad(90), gamma=0.1, phi=.5, normalize=True).astype(np.float32)

ipt_patterns = (
    G1 * 0.1,           # make cues not tiny vs baseline
    G2 * 0.1,
    (np.random.randn(N, N).astype(np.float32) * 0.1)          # go cue
)

def make_stim_withN(N):
    G1 = gabor2d(N, f=5*.5, theta=np.deg2rad(30), gamma=0.1, phi=.5, normalize=True).astype(np.float32)
    G2 = gabor2d(N, f=1*.5, theta=np.deg2rad(60), gamma=0.1, phi=.5, normalize=True).astype(np.float32)
    G3 = gabor2d(N, f=3*.5, theta=np.deg2rad(90), gamma=0.1, phi=.5, normalize=True).astype(np.float32)

    ipt_patterns = (
        G1 * 0.1,           # make cues not tiny vs baseline
        G2 * 0.1,
        (np.random.randn(N, N).astype(np.float32) * 0.1)          # go cue
    )
    return ipt_patterns

# Reservoir params
params = {
    "dt": 0.001,
    "K": 20.0, ### 20 seems great!!
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

# Trial factory
def trial_fn(lr=None, ipt_patterns=ipt_patterns, N=N):
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


@torch.no_grad()
def eval_r2(model, trial_fn, n_trials, mask=None):
    """
    Compute pooled R^2 scores for output and memory readouts.
    If mask is provided, it is applied before pooling.
    """
    model.eval()
    if mask is not None:
        mask = np.asarray(mask).astype(bool)

    out_true_all, out_pred_all = [], []
    mem_true_all, mem_pred_all = [], []

    for _ in range(n_trials):
        target_out, target_mem, stim, _ = trial_fn()
        out, mem, _ = model(stim)

        out = out.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float64)
        mem = mem.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float64)
        target_out = target_out.detach().cpu().numpy().astype(np.float64)
        target_mem = target_mem.detach().cpu().numpy().astype(np.float64)

        if mask is not None:
            if mask.shape[0] != out.shape[0]:
                raise ValueError(f"mask length ({mask.shape[0]}) does not match trial length ({out.shape[0]})")
            out = out[mask]
            target_out = target_out[mask]
            mem = mem[mask]
            target_mem = target_mem[mask]

        out_true_all.append(target_out)
        out_pred_all.append(out)
        mem_true_all.append(target_mem)
        mem_pred_all.append(mem)

    def _r2(y_true_list, y_pred_list):
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        sse = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - y_true.mean()) ** 2)
        return float(1.0 - sse / (sst + 1e-12))

    return _r2(out_true_all, out_pred_all), _r2(mem_true_all, mem_pred_all)

# %% looping K for performance
Ks = np.array([1, 10, 20, 40, 80, 160])
Ks = np.array([1, 10, 100, 1000])
Ns = np.array([15, 21, 27, 33, 37, 41])
errs_K = np.zeros((len(Ks), 2))  # out, mem
r2s_K = np.zeros((len(Ks), 2))    # out, mem

mask = np.zeros((T,), dtype=bool)
mask[go_start:] = True
for kk in range(len(Ks)):

    ### for changing  K
    K = Ks[kk]
    print(f"Training with K={K}...")
    params['K'] = K
    
    ### for changing N
    # N = Ns[kk]
    # print(f"Training with N={N}...")

    # Make model
    model = Relu2DSpatialReservoir(N, T, output_dim, device, params)

    ### if chaing N size
    ipt_patterns = make_stim_withN(N)
    # print(ipt_patterns[0].shape)

    # Collect for mem
    X_mem, _, Y_mem = collect_xy(model, lambda: trial_fn(lr=None, ipt_patterns=ipt_patterns, N=N), n_train, time_selector_mem)
    # Collect for out
    X_out, Y_out, _ = collect_xy(model, lambda: trial_fn(lr=None, ipt_patterns=ipt_patterns, N=N), n_train, time_selector_out)
    # Solve ridge
    Wmem = ridge_solve(X_mem, Y_mem, lam)          # (D+1, 1)
    Wout = ridge_solve(X_out, Y_out, lam)          # (D+1, 1)

    # Assign into model
    set_readouts_from_ridge(model, Wout, Wmem)

    # Evaluate (pass the same ipt_patterns and N used for training to avoid default-arg mismatch)
    lo, lm = eval_model(model, lambda: trial_fn(lr=None, ipt_patterns=ipt_patterns, N=N), n_trials=10, mask=mask)
    r2o, r2m = eval_r2(model, lambda: trial_fn(lr=None, ipt_patterns=ipt_patterns, N=N), n_trials=10, mask=mask)
    print(f"Ridge eval MSE: out={lo:.4g}, mem={lm:.4g}")
    print(f"Ridge eval R2:  out={r2o:.4g}, mem={r2m:.4g}")
    errs_K[kk, 0] = lo
    errs_K[kk, 1] = lm
    r2s_K[kk, 0] = r2o
    r2s_K[kk, 1] = r2m

# %% Plotting
plt.figure(figsize=(6, 4))
plt.plot(Ks, errs_K[:, 0], marker='o', label='Output MSE')
plt.plot(Ks, errs_K[:, 1], marker='o', label='Memory MSE')
plt.xscale('log')
# plt.xlabel('Strength K')
plt.xlabel('Reservoir Strength K')
plt.ylabel('MSE')
plt.title('Ridge Regression Performance vs K')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(Ks, r2s_K[:, 0], marker='o', label='Output R2')
plt.plot(Ks, r2s_K[:, 1], marker='o', label='Memory R2')
plt.xscale('log')
plt.xlabel('Reservoir Strength K')
plt.ylabel('R2')
plt.title('Ridge Regression R2 vs K')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# plt.figure(figsize=(6, 4))
# plt.plot(Ns, errs_K[:, 0], marker='o', label='Output MSE')
# plt.plot(Ns, errs_K[:, 1], marker='o', label='Memory MSE')
# plt.xscale('log')
# # plt.xlabel('Strength K')
# plt.xlabel('Reservoir Size N')
# plt.ylabel('MSE')
# plt.title('Ridge Regression Performance vs N')
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()

# %% just show readout
plt.figure(figsize=(12, 6))
plt.plot(Ks, r2s_K[:, 1], marker='o', label='Memory R2')
plt.xlabel('Reservoir Strength K')
plt.ylabel('R2')
plt.xscale('log')
plt.title('Memory R2 vs K')
plt.legend()
plt.show()
