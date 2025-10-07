# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:49:58 2024

@author: kevin
"""

import numpy as np
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% test low-rank random
N = 10
R = np.random.randn(N,N)
# Define individual blocks
block1 = np.outer(R[:,0], R[:,1])
block2 = np.outer(R[:,2], R[:,3])
block3 = np.outer(R[:,0], R[:,3])
block4 = np.outer(R[:,0], R[:,3])

# Create block diagonal matrix
block_matrix = block_diag(block1, block2, block3, block4)

uu,ss,vv = np.linalg.svd(block_matrix)
plt.figure()
plt.plot(ss,'-o')

# %% test Gaussian local connectivity
# Define matrix size
n = 100
# Define Gaussian parameters
sigma = 5  # Standard deviation of the Gaussian

# Create the matrix
gaussian_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        # Gaussian decay based on distance from the diagonal
        gaussian_matrix[i, j] = np.exp(-((i - j) ** 2) / (2 * sigma ** 2))

# Display the matrix
print("Matrix with Gaussian bump around the diagonal:")
print(np.round(gaussian_matrix, 2))

# Visualize the matrix
plt.figure()
plt.imshow(gaussian_matrix, cmap='viridis')
plt.colorbar(label='Value')
plt.title('Matrix with Gaussian Bump Around Diagonal')
plt.show()

uu,ss,vv = np.linalg.svd(gaussian_matrix)
plt.figure()
plt.plot(ss,'-o')

# %% local connection matrix
def construct_periodic_connectivity(N):
    # Total number of nodes
    total_nodes = N * N
    
    # Connectivity matrix
    connectivity = np.zeros((total_nodes, total_nodes), dtype=int)
    
    # Helper function to map 2D coordinates to a 1D index
    def idx(x, y):
        return (x % N) * N + (y % N)  # Modulo ensures periodicity
    
    # Iterate through all nodes
    for x in range(N):
        for y in range(N):
            current = idx(x, y)
            # Add connections to neighbors with periodic boundary
            neighbors = [
                idx(x - 1, y),  # Up
                idx(x + 1, y),  # Down
                idx(x, y - 1),  # Left
                idx(x, y + 1)   # Right
            ]
            for neighbor in neighbors:
                connectivity[current, neighbor] = 1

    return connectivity

# Example for a 3x3 grid
N = 3
connectivity_matrix = construct_periodic_connectivity(N)

# Display the connectivity matrix
print("Connectivity Matrix (N=3):")
print(connectivity_matrix)

# %% larger periodic network
def construct_weighted_periodic_connectivity(N, sigma):
    # Total number of nodes
    total_nodes = N * N
    sigma = sigma*N
    # Connectivity matrix
    connectivity = np.zeros((total_nodes, total_nodes))
    
    # Helper function to map 2D coordinates to a 1D index
    def idx(x, y):
        return (x % N) * N + (y % N)  # Modulo ensures periodicity
    
    # Iterate through all nodes
    for x1 in range(N):
        for y1 in range(N):
            # 1D index of the current node
            current = idx(x1, y1)
            for x2 in range(N):
                for y2 in range(N):
                    # 1D index of the neighboring node
                    neighbor = idx(x2, y2)
                    
                    # Compute the distance considering periodic boundaries
                    dx = min(abs(x1 - x2), N - abs(x1 - x2))
                    dy = min(abs(y1 - y2), N - abs(y1 - y2))
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Compute the weight as an exponential decay
                    connectivity[current, neighbor] = np.exp(-distance**2 / (2 * sigma**2))
    
    return connectivity

# Example for a 3x3 grid
N = 50
sigma = N*0.1  # Standard deviation for the Gaussian decay
weighted_connectivity_matrix = construct_weighted_periodic_connectivity(N, sigma)

plt.figure()
plt.imshow(weighted_connectivity_matrix)

### visualize projection
location = (4, 4)  # Node to visualize
    
# Map the 2D location to 1D index
i, j = location
node_index = (i % N) * N + (j % N)

# Extract the row corresponding to this location
projections = weighted_connectivity_matrix[node_index, :]

# Reshape projections back to 2D grid
projection_grid = projections.reshape(N, N)

# Plot the projections
plt.figure(figsize=(8, 6))
plt.imshow(projection_grid, cmap='viridis', extent=[0, N, 0, N])
plt.colorbar(label='Projection Weight')
plt.title(f'Projections from Node ({i}, {j})')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().invert_yaxis()  # To match typical matrix orientation
plt.show()

uu,ss,vv = np.linalg.svd(weighted_connectivity_matrix)
plt.figure()
plt.plot(ss, '-o')

# %% test with 2D EI connnectivity
###############################################################################
###############################################################################
# %% parameters
N = 30
rescale = 1
sig_e, sig_i = 0.1/2, 0.2/2
Wee = 1* 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = 1*  .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale

# %% combine sub-matices
def make_2dM(N, sig_e, sig_i):
    Mee = Wee*construct_weighted_periodic_connectivity(N, sig_e)
    Mei = Wei*construct_weighted_periodic_connectivity(N, sig_i)
    Mie = Wie*construct_weighted_periodic_connectivity(N, sig_e)
    Mii = Wii*construct_weighted_periodic_connectivity(N, sig_i)
    M_2d = np.block([[Mee, Mei],[Mie, Mii]])
    return M_2d
M_2d = make_2dM(N, sig_e, sig_i)

# %%
uu,ss,vv = np.linalg.svd(M_2d)
plt.figure()
plt.plot(ss,'-o')

# %% local motifs
def motif_rec(Jij):
    tau_rec = np.mean( (Jij - np.mean(Jij)) * (Jij.T - np.mean(Jij)) ) / \
                (np.mean((Jij-np.mean(Jij))**2)*np.mean((Jij.T-np.mean(Jij.T))**2))**0.5
    return tau_rec

def motif_chain(Jij):
    ### slow loopy method
    # zij_zjk = 0
    # zjk = 0
    # count = 0
    # n = Jij.shape[0]
    # for i in range(n):
    #     for k in range(n):
    #         if i != k:  # Exclude pairs where i == k
    #             for j in range(n):
    #                 zij_zjk += Jij[i, j] * Jij[k, j]
    #                 zjk += Jij[k,j]**2
    #                 count += 1
    # mean_zz = zij_zjk/count
    # meanz2 = zjk/count
    
    ### fast einsum method
    n = Jij.shape[0]
    pairwise_products = Jij @ Jij.T  # Efficiently compute the sum over j: (n x n matrix)
    # Mask out diagonal elements where i == k
    off_diagonal_sum = pairwise_products[~np.eye(n, dtype=bool)].sum()
    # Number of valid off-diagonal elements
    num_off_diagonal = n * (n - 1)
    # Compute mean
    zij_zjk = off_diagonal_sum / num_off_diagonal
    zij2 = np.mean(Jij**2)
    self_product = Jij * Jij
    off_diagonal_sum2 = self_product[~np.eye(n, dtype=bool)].sum()
    zjk2 = off_diagonal_sum2 / num_off_diagonal
    out = zij_zjk / (zij2*zjk2)**0.5
    
    #### normalize corretly... ################################################
    # n = Jij.shape[0]
    # num_off_diagonal = n*(n - 1)
    # JijJjk = Jij @ Jij.T
    # JijJjk = JijJjk[~np.eye(n, dtype=bool)].sum() / num_off_diagonal  ### <Jij Jjk>
    # Jij_ = np.mean(Jij)   ### <Jij>
    # Jji = Jij.T
    # Jjk_ = Jji[~np.eye(n, dtype=bool)].sum() / num_off_diagonal  ### <Jjk>
    # Jij_std = np.mean((Jij - np.mean(Jij))**2)
    # Jjk_std = (Jji - Jjk_)**2
    # Jjk_std = Jjk_std[~np.eye(n, dtype=bool)].sum() / num_off_diagonal
    # out = (JijJjk - Jij_*Jjk_ ) / np.sqrt(Jij_std*Jjk_std)
    
    ###########################################################################
    # out = mean_zz / (np.mean(Jij**2)*meanz2)
    # out = np.mean(Jij*Jij)
    return out

# %%
space_ie = np.array([.5,1,1.5,2,2.5])
tau_recs = np.zeros(len(space_ie))
tau_chain = tau_recs*0
eigs_s = np.zeros((len(space_ie), N**2*2))
for ii in range(len(tau_recs)):
    M_2d = make_2dM(N, sig_e, sig_e*space_ie[ii])
    ### measurements
    tau_recs[ii] = motif_rec(M_2d)
    # tau_chain[ii] = motif_chain(M_2d)
    # uu,vv = np.linalg.eig(M_2d)
    # eigs_s[ii,:] = uu
    print(ii)
    
# %%
plt.figure()
# plt.plot(space_ie, n30_recs/1**2, '-o', label='N=30')
plt.plot(space_ie, tau_recs, '--.', label='N=40')
plt.xlabel(r'$\sigma_i/\sigma_e$')
plt.ylabel('Reciprocalness')
plt.legend()

# %%
plt.figure()
plt.plot(space_ie, tau_chain/N**2, '-o', label='N=30')
# plt.plot(space_ie, n40_chain/40**2, '--.', label='N=40')
plt.xlabel(r'$\sigma_i/\sigma_e$')
plt.ylabel('Chain-ness')
plt.legend()

# %%
plt.figure()
# plt.plot(space_ie, n30_trace, '-o', label='N=30')
plt.plot(space_ie, np.max(np.real(eigs_s.T),0),'--.', label='N=40')
plt.xlabel(r'$\sigma_i/\sigma_e$')
plt.ylabel(r'$\lambda_{max}$')
# plt.xlim([1700,1800]); plt.ylim([-1,2])
plt.legend()