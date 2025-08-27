# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 01:43:24 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% degree vs. dynamics
###
# a simple numerical demonstration about fluctuation in RNN
# the in degrees will correlated with flucation and time scales
# and this should depdent on the regime it is in
# if distinction is large, makes strong prediction about connectome!
###

# %%
# --- Parameters ---
N = 1000               # number of neurons
p = 0.15               # connection probability
mean_in_deg = int(p * N)
T = 300                # total simulation time
dt = 0.1               # time step
steps = int(T / dt)    # number of steps
g =  0.1               # connection strength scaling
noise = 10              # noise strength

# --- Activation function ---
def phi(x):
    return np.tanh(x)

# --- Generate in-degree and matching out-degree sequences ---
in_degrees = np.random.poisson(mean_in_deg, N)
total_in = in_degrees.sum()
out_degrees = np.random.multinomial(total_in, np.ones(N)/N)

# --- Create random directed graph with desired degree distributions ---
G = nx.directed_configuration_model(out_degrees, in_degrees, create_using=nx.DiGraph())
G = nx.DiGraph(G)  # convert to simple directed graph
G.remove_edges_from(nx.selfloop_edges(G))  # remove self-loops

# --- Create connectivity matrix and assign weights ---
A = nx.to_numpy_array(G)
J = np.random.normal(0, g / np.sqrt(mean_in_deg*p), size=(N, N)) * A
deg = A.sum(1)
# J = (J.T / np.sqrt(deg + 1e-8)).T  # row-wise scaling

# --- Simulate RNN dynamics: dx/dt = -x + J*phi(x) ---
x = np.random.randn(N)
X = []

for t in range(steps):
    # if 100t==0:
        # print(t)
    ### w/o noise
    # x = x + dt * (-x + J @ phi(x))
    ### current noise
    # x = x + dt * (-x + J @ phi(x)) + np.sqrt(dt)*noise*np.random.randn(N)*0
    # x = x + dt * (-x + J @ phi(x)) + np.sqrt(dt) * (A @ (noise * np.random.randn(N)))
    ### synaptic noise
    # Jn = J + A*(noise * np.random.randn(N,N))
    # x = x + dt * (-x + Jn @ phi(x)) 
    sn = (A / deg[:, None]**1) @ (noise * np.random.randn(N))  #### HERE: scale with 1, 1/K or 1/sqrt(K), makes scaling different
    x = x + dt * (-x + J @ phi(x)) + np.sqrt(dt)*sn
    
    X.append(x.copy())

X = np.array(X)

# --- Analyze fluctuations ---
mean_activity = X.mean(axis=1)
std_activity = X.std(axis=1)

# --- Plot results ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mean_activity)
plt.title("Mean activity over time")
plt.xlabel("Time step")
plt.ylabel("Mean activity")

plt.subplot(1, 2, 2)
plt.plot(std_activity)
plt.title("Std of activity over time")
plt.xlabel("Time step")
plt.ylabel("Std deviation")

plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(X.T, aspect='auto')

# %%
plt.figure()
degree = A.sum(1)
fluc = np.std(X,0)**2
plt.plot(degree, fluc, '.', alpha=.7)
plt.xlabel('in degrees K'); plt.ylabel('fluctuation')

# %% scaling
# from scipy.optimize import curve_fit
# from scipy.stats import linregress

# # Define inverse model: a / k + b
# def inverse_model(k, a, b):
#     return a / k + b

# # Fit the inverse model
# params, _ = curve_fit(inverse_model, degree, fluc)
# a_fit, b_fit = params

# # Generate fit line
# k_fit = np.linspace(min(degree), max(degree), 100)
# var_fit = inverse_model(k_fit, a_fit, b_fit)

# # Plot original data and fit
# plt.figure(figsize=(6, 5))
# plt.scatter(degree, fluc, alpha=0.4, label="data")
# plt.plot(k_fit, var_fit, 'r-', label=f'fit: a/k + b\n(a={a_fit:.2f}, b={b_fit:.2f})')
# plt.xlabel("In-degree (k)")
# plt.ylabel("Variance of activity")
# plt.title("Variance vs In-degree (CLT test)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # --- Log-log plot (optional) ---
# log_k = np.log(degree)
# log_var = np.log(fluc)
# slope, intercept, r_value, _, _ = linregress(log_k, log_var)

# plt.figure()
# plt.scatter(log_k, log_var, alpha=0.4)
# plt.plot(log_k, intercept + slope * log_k, 'r-', label=f'log-log slope = {slope:.2f}')
# plt.xlabel("log(k)")
# plt.ylabel("log(Var)")
# plt.title("Log-log plot: Var ~ k^(-α)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print(f"Linear log-log slope: {slope:.2f}, R² = {r_value**2:.2f}")
