# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:28:57 2025

@author: kevin
"""

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

import pandas as pd

from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

# %% load gz file
# Load a compressed CSV file
connectome_path = r'C:/Users/kevin/Downloads/connections.csv.gz'
df = pd.read_csv(connectome_path)

# Show the first few rows
print(df.head())

# %% extract counts
pre_id = np.array(df['pre_root_id'])
post_id = np.array(df['post_root_id'])
nt_type = np.array(df['nt_type']).tolist()

print('number of neurons:', len(np.unique(pre_id)))
print('types of synapse:', set(nt_type))

# %% show degree distribution
pos = np.where(np.array(df['nt_type'])=='OCT')
_, counts = np.unique(post_id[pos], return_counts=True)
_, counts = np.unique(post_id, return_counts=True)

# Step 2: Sort frequencies in descending order
sorted_counts = np.sort(counts)[::-1]

# Step 3: Compute ranks
ranks = np.arange(1, len(sorted_counts) + 1)

# Step 4: Log-log plot
plt.figure()
plt.loglog(ranks, sorted_counts, marker='o', linestyle='none')

# %% build Aij matrix
import numpy as np
from scipy.sparse import coo_matrix

# Step 1: Map unique neuron IDs to 0-based indices
unique_ids = np.unique( np.concatenate((pre_id , post_id)) )
id_to_index = {id_: i for i, id_ in enumerate(unique_ids)}
index_to_id = {i: id_ for id_, i in id_to_index.items()}  # optional reverse map

# Step 2: Build row/col index lists for sparse matrix
rows = [id_to_index[pre] for pre in pre_id]
cols = [id_to_index[post] for post in post_id]
data = [1] * len(pre_id)  # or use weights if available

# Step 3: Create sparse adjacency matrix A[i, j] = 1 if edge i → j
n = len(unique_ids)
A = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=int)

# Optional: inspect
print("Adjacency matrix shape:", A.shape)
print("Non-zero entries (edges):", A.nnz)

# Optional: convert to dense (small graphs only)
# dense_A = A.toarray()

# %% plots
# Ensure A is in CSR format for fast row slicing
Asub = A.tocsr()

# Pick a small block: say, rows 0–99 and cols 0–99
sub_A = Asub[0:15000, 0:15000].toarray()

plt.figure(figsize=(6,6))
plt.imshow(np.log(sub_A+1e-5), cmap='Greys', interpolation='none')
plt.title("Submatrix of Adjacency (0–99)")
plt.xlabel("Post")
plt.ylabel("Pre")
plt.colorbar(label="Connection")
plt.show()

# %%
###############################################################################
# %% test out some coarse-graining ideas?
L = csgraph.laplacian(A, normed=True)

# Step 3: Safety check after Laplacian
print("Any NaNs in L:", np.isnan(L.data).any())
print("Any Infs in L:", np.isinf(L.data).any())

# Step 4: Randomized SVD for spectral embedding
n_components = 50  # how many eigenvectors to keep
U, Sigma, VT = randomized_svd(L, n_components=n_components, random_state=1)

# %%
# Step 5: Cluster nodes in spectral space
n_supernodes = 2000  # how many groups you want
kmeans = KMeans(n_clusters=n_supernodes, random_state=1, n_init="auto")
labels = kmeans.fit_predict(U)

# %%
# Step 6: Build coarse-grained connectivity matrix
def coarsegrain_sparse(A, labels, n_supernodes):
    """
    Coarse-grain a sparse matrix A given cluster labels.
    """
    A_coarse = np.zeros((n_supernodes, n_supernodes))

    rows, cols = A.nonzero()
    values = A.data  # <- Here's the key fix!

    for r, c, v in zip(rows, cols, values):
        group_r = labels[r]
        group_c = labels[c]
        A_coarse[group_r, group_c] += v

    return A_coarse

A_coarse = coarsegrain_sparse(A, labels, n_supernodes)

print(f"Coarse-grained matrix shape: {A_coarse.shape}")

# Step 7: (Optional) Visualize coarse-grained matrix
plt.figure(figsize=(8,6))
plt.imshow(np.log(A_coarse+1e-9), cmap='viridis')
plt.colorbar()
plt.title("Coarse-Grained Connectivity Matrix")
plt.xlabel("Supernode (target)")
plt.ylabel("Supernode (source)")
plt.show()

# %% now analyze CG network
temp = np.sum(A_coarse,1)
sorted_temp = np.sort(temp)[::-1]

# Step 3: Compute ranks
ranks = np.arange(1, len(sorted_temp) + 1)

# Step 4: Log-log plot
plt.figure()
plt.loglog(ranks, sorted_temp, marker='o', linestyle='none')