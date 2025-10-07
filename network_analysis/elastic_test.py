# -*- coding: utf-8 -*-
"""
Created on Fri May  2 09:49:06 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Define grid
L = 50
T = 100
dx = 1.0
dt = 0.1
x = np.linspace(0, L-1, L)
y = np.linspace(0, L-1, L)
X, Y = np.meshgrid(x, y)

# Simulate simple displacement field over time (2D wave-like behavior)
u = np.zeros((T, L, L))  # displacement field u(x, y, t)

# Initialize a wave packet in the center
for t in range(T):
    radius = 5 + 0.1 * t
    u[t] = np.exp(-((X - L//2)**2 + (Y - L//2)**2) / (2 * radius**2)) * np.sin(0.2 * t)

# Compute strain tensor at each point (using central difference approximation)
def compute_strain(u_t):
    du_dx = (np.roll(u_t, -1, axis=1) - np.roll(u_t, 1, axis=1)) / (2 * dx)
    du_dy = (np.roll(u_t, -1, axis=0) - np.roll(u_t, 1, axis=0)) / (2 * dx)
    eps_xx = du_dx
    eps_yy = du_dy
    eps_xy = 0.5 * ((np.roll(u_t, -1, axis=0) - np.roll(u_t, 1, axis=0)) / (2 * dx) +
                    (np.roll(u_t, -1, axis=1) - np.roll(u_t, 1, axis=1)) / (2 * dx))
    return eps_xx, eps_yy, eps_xy

# Assume simple Hookean response (stress = E * strain), with E = 1
def compute_stress(eps_xx, eps_yy, eps_xy, E=1.0, nu=0.3):
    # Plane stress assumption
    lam = E * nu / (1 - nu**2)
    mu = E / (2 * (1 + nu))
    sigma_xx = lam * (eps_xx + eps_yy) + 2 * mu * eps_xx
    sigma_yy = lam * (eps_xx + eps_yy) + 2 * mu * eps_yy
    sigma_xy = 2 * mu * eps_xy
    return sigma_xx, sigma_yy, sigma_xy

# Choose a frame to visualize
frame = 50
eps_xx, eps_yy, eps_xy = compute_strain(u[frame])
sigma_xx, sigma_yy, sigma_xy = compute_stress(eps_xx, eps_yy, eps_xy)

# Plot stress fields
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
im0 = axs[0].imshow(sigma_xx, cmap='coolwarm', origin='lower')
axs[0].set_title(r'$\sigma_{xx}$')
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(sigma_yy, cmap='coolwarm', origin='lower')
axs[1].set_title(r'$\sigma_{yy}$')
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(sigma_xy, cmap='coolwarm', origin='lower')
axs[2].set_title(r'$\sigma_{xy}$')
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 50
T = 100
dx = 1.0
dt = 0.1
x = np.linspace(0, L - 1, L)
y = np.linspace(0, L - 1, L)
X, Y = np.meshgrid(x, y)

# Simulate wave-like displacement field u(x, y, t)
u = np.zeros((T, L, L))
for t in range(T):
    radius = 5 + 0.1 * t
    u[t] = np.exp(-((X - L // 2)**2 + (Y - L // 2)**2) / (2 * radius**2)) * np.sin(0.2 * t)

# Compute acceleration: second derivative over time
def compute_acceleration(u, dt):
    return (u[2:] - 2 * u[1:-1] + u[:-2]) / (dt ** 2)

acc = compute_acceleration(u, dt)

# Estimate stress from acceleration via inverse divergence
def inverse_divergence(acc_field, dx):
    stress_x = np.zeros_like(acc_field)
    stress_y = np.zeros_like(acc_field)
    for t in range(acc_field.shape[0]):
        ax = acc_field[t]
        stress_x[t] = np.cumsum(ax, axis=1) * dx
        stress_y[t] = np.cumsum(ax, axis=0) * dx
    return stress_x, stress_y

stress_x, stress_y = inverse_divergence(acc, dx)

# Compute strain tensor from spatial gradients
def compute_strain_tensor(u_t):
    du_dx = (np.roll(u_t, -1, axis=1) - np.roll(u_t, 1, axis=1)) / (2 * dx)
    du_dy = (np.roll(u_t, -1, axis=0) - np.roll(u_t, 1, axis=0)) / (2 * dx)
    eps_xx = du_dx
    eps_yy = du_dy
    eps_xy = 0.5 * ((np.roll(u_t, -1, axis=0) - np.roll(u_t, 1, axis=0)) / (2 * dx) +
                    (np.roll(u_t, -1, axis=1) - np.roll(u_t, 1, axis=1)) / (2 * dx))
    return eps_xx, eps_yy, eps_xy

# Prepare dataset of (strain, stress) pairs
num_samples = (T - 2) * L * L
strain_data = np.zeros((num_samples, 3))
stress_data = np.zeros((num_samples, 3))
index = 0

for t in range(T - 2):
    u_t = u[t + 1]
    eps_xx, eps_yy, eps_xy = compute_strain_tensor(u_t)
    sig_x = stress_x[t]
    sig_y = stress_y[t]
    sig_xy = 0.5 * (sig_x + sig_y)  # approximate symmetric shear

    for i in range(L):
        for j in range(L):
            strain_data[index] = [eps_xx[i, j], eps_yy[i, j], eps_xy[i, j]]  ### change of basis here to test odd elasticity
            stress_data[index] = [sig_x[i, j], sig_y[i, j], sig_xy[i, j]]
            index += 1

# Fit effective elasticity tensor K (sigma = K * epsilon)
E = strain_data
S = stress_data
K_fit = np.linalg.lstsq(E, S, rcond=None)[0]

# Display result
print("Fitted elasticity tensor K:")
print(K_fit)

# Optional: visualize one frame of stress_x and stress_y
frame = 40
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im0 = axs[0].imshow(stress_x[frame], cmap='coolwarm', origin='lower')
axs[0].set_title(r'$\sigma_x$ (frame {})'.format(frame))
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(stress_y[frame], cmap='coolwarm', origin='lower')
axs[1].set_title(r'$\sigma_y$ (frame {})'.format(frame))
plt.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()
