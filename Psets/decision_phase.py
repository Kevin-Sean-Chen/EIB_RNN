# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:18:36 2024

@author: kevin
"""

import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %%
###
# code for pset2 presentation
# from Srdjan and Maxime's modification
###
# %% define activation function
def F(x):
    # beta,gamma = 4.0, 0.7
    # nl = 1/(1+np.exp(-beta*(x-gamma)))  # sigmoid
    nl = np.where(x>0,x,0)  # ReLU
    return nl

# %% setup network
J = np.zeros((2,2))
J_inh = -1

J[0,1] = J_inh
J[1,0] = J_inh

J = np.array([[0.2, -0.7],
              [-0.7, 0.2]])  ### what we have in the pset

# %% defining neural dynamics
def IntegrateAuto(X, t):
    dXdT = -X + np.dot(J, F(X)) + I  ### RNN form
    return dXdT

def LookActAuto(_t,_x0):
    _x=scipy.integrate.odeint(IntegrateAuto,_x0,_t)   ### numerical ODE solve
    return _x

# %% setup nullclines
def nullclines_rhs(x):
    X = np.array([x,x])
    return np.dot(J, F(X)) + I  ### rhs of the RNN model

# %% plotting
I = 0
x = np.arange(-2, 5, 0.1)
nullclines = nullclines_rhs(x)  ### 2D nullclines

# %%
I = 3.0
J_inh = -3.0

J[0,1] = J_inh
J[1,0] = J_inh

x = np.arange(-1, 5, 0.1)
nullclines = nullclines_rhs(x)

t_vals=np.arange(0,5,.01)

fig1, ax1 = plt.subplots()
ax1.plot(x, nullclines[0], label="dx1/dt = 0")
ax1.plot(nullclines[1], x, label="dx2/dt = 0")
ax1.plot(x, x, color='k', lw=1)
ax1.set_xlabel('x1', fontsize=20)
ax1.set_ylabel('x2', fontsize=20)
ax1.legend(fontsize=20)

fig2, ax2 = plt.subplots()
for x0_1 in np.arange(0, 4, 0.5):
     for x0_2 in np.arange(0, 4, 0.5):
          x_0 = np.array([x0_1, x0_2])
          x_t = LookActAuto(t_vals,x_0)
          ax1.scatter(x_t[:,0], x_t[:,1],
                    marker='o', c=np.arange(len(x_t)), cmap='viridis',
                    lw=0.5, s=2, zorder=10)
          ax2.plot(t_vals, x_t[:,0], color='magenta', lw=0.5, alpha=0.5)
          ax2.plot(t_vals, x_t[:,1], color='cyan', lw=0.5, alpha=0.5)
ax2.set_xlabel('time', fontsize=20)
ax2.set_ylabel('activity', fontsize=20)