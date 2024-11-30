# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 01:07:05 2024

@author: kevin
"""

# Import libraries
import numpy as np
import autograd.numpy as anp
import scipy as sp
import matplotlib.pyplot as plt
from lyapynov import ContinuousDS, DiscreteDS
from lyapynov import mLCE, LCE, CLV, ADJ
import math

import jax
key = jax.random.PRNGKey(13)

# %% RNN example
### edit from https://github.com/RainerEngelken/RNN-LyapunovSpectra
### model
N = 100  # number of neurons
g = 4  # strength
J = g*np.random.randn(N, N)/np.sqrt(N)  # Initialize random network realization
J = J-np.diag(np.diag(J))  # remove autapse 
dt = 0.1  # time step
T = 1000  # full time in seconds
nLE = 100  # number of LEs
tau = 1  # characteristic time

### time steps
init_t = 100  # inital time to run
tONS = 1  # time between orthonormalizations
nStepTransient = math.ceil(init_t/dt)  # Steps during initial transient
nStep = math.ceil(T/dt)  # Total number of steps
nstepONS = math.ceil(tONS/dt)  # Steps between orthonormalizations
nStepTransientONS = math.ceil(nStep/(10))  # Steps during transient of ONS
nONS = math.ceil((nStepTransientONS+nStep)/nstepONS) - 1  # Number of ONS steps
stepDisplay = 100

### initialization
h = np.zeros((N, (nStep + nStepTransient + nStepTransientONS)))  # Preallocate local fields
h[:, 0] = (g-1)*np.random.randn(N)  # Initialize network state
t = 0
lsidx = -1
LSall = np.zeros((nONS, nLE))  # Initialize array to store convergence of Lyapunov spectrum
normdhAll = np.zeros(nONS)  # Initialize
pAll = np.zeros(nONS)  # participant ratio
LS = np.zeros(nLE)  # Lyapunov spectrum
clv = np.zeros((N, nONS)) # tracking full CLV

### for dynamical systems
q, r = np.linalg.qr(np.random.randn(N, nLE))  # Initialize orthonormal system
Ddiag = np.eye(N)*(1-dt)  # Diagonal elements of Jacobian

### dynamics
for n in range(nStep + nStepTransient + nStepTransientONS-1):
    h[:, n+1] = h[:, n]*(1-dt)+np.dot(J, np.tanh(h[:, n]))*dt  # network dynamics

    if (n+1 > nStepTransient):
        ######### Jacobian #########
        hsechdt = dt/np.cosh(h[:, n])**2  # derivative of tanh(h)*dt
        D = Ddiag + J*hsechdt  # Jacobian
        ############################
        q = np.dot(D, q)  # evolve orthonormal system using Jacobian
        if np.mod(n+1, nstepONS) == 0:
            q, r = np.linalg.qr(q)  # performe QR-decomposition
            if nLE == 1:
                q4 = q*q*q*q
            else:
                q1 = q[:, 0]
                q4 = q1*q1*q1*q1
            lsidx += 1
            clv[:, lsidx] = q1
            pAll[lsidx] = 1.0/np.sum(q4)
            LSall[lsidx, :] = np.log(np.abs(np.diag(r)))/nstepONS/dt  # store convergence of Lyapunov spectrum
            if n + 1 > nStepTransientONS + nStepTransient:
                LS += np.log(np.abs(np.diag(r)))  # collect log of diagonal elements or  R for Lyapunov spectrum
                t += nstepONS*dt  # increment time
        
        if np.mod(n + 1, stepDisplay) == 0:  # plot progress
            if n + 1 > nStepTransient + nStepTransientONS:
                PercentageFinished = (n + 1 - nStepTransient - nStepTransientONS)*100/nStep
                print(round(PercentageFinished), ' % done after ', round(3.3), 's SimTime: ', round(dt*(n+1)), ' tau, std(h) =', round(np.std(h[:, n+1]), 2))

Lspectrum = LS/t  # Normalize sum of log of diagonal elements of R by total simulation time

# %% metrics
def entropy_rate(Lspec):
    pos = np.where(Lspec>0)[0]
    return np.sum(Lspec[pos])

def attractor_dim(Lspec):
    """
    D = d + S_d / |lamb_{d+1}| (with d the largest integer such that S+d = Sim^d_i lamb_i >= 0)
    """
    lamb_sum = np.cumsum(Lspec)
    d = np.where(lamb_sum>=0)[0][-1]
    D = d + np.sum(Lspec[:d])/np.abs(Lspec[d+1])
    return D

# %% example plotting
# from pylab import *
# import random
# import math
# subplot(221)  # Plot example trajectories
# plot(dt*arange(h.shape[1]), transpose(h[0:3, :]))
# title('example activity')
# ylabel('$x_i$')
# xlabel(r'Time ($\tau$)')
# xlim(0, dt*h.shape[1])

# subplot(223)  # Plot Lyapunov spectrum
# plot(1.0*arange(nLE)/nLE, Lspectrum, '.k')
# plot(1.0*arange(nLE)/N, zeros(nLE), ':', color=[0.5, 0.5, 0.5])
# ylabel(r'$\lambda_i (1/\tau)$')
# xlabel(r'$i/N$')
# title("Lyapunov exponents")

# subplot(222)  # Plot time-resolved first local Lyapunov exponent
# # (Jacobian-based and direct simulations)
# plot(arange(nONS)*nstepONS*dt, LSall[:, 0], 'k')
# xlabel(r'Time ($\tau$)')
# ylabel('$\lambda_i^{local}$')
# title('first local Lyapunov exponent')
# xlim(0, nONS*nstepONS*dt)

# subplot(224)  # Plot participation ratio
# plot(arange(nONS)*nstepONS*dt, pAll[:], 'k')
# xlabel(r'Time ($\tau$)')
# ylabel('P')
# title('participation ratio')
# xlim(0, nONS*nstepONS*dt)

# %% TO-DO
###############################################################################
### try mLCE for 2D EI with Euler dynamics
### be able to compute state x and dx
### mumerical Jacobian
### later explore vectors...

### driven signal!!
### later scan through sigma/tau phase space

# %%
import jax.numpy as np

# %% setup function version of EI-2D network
### params
N = 30  # neurons
tau_e = 0.005  # time constant ( 5ms in seconds )
sig_e = 0.1  #.1 spatial kernel
tau_i, sig_i = 0.015, 0.2#0.015, 0.14 #15*0.001, 0.2 
kernel_size = 23

### EI-balanced
rescale = 3. ##(N*sig_e*np.pi*1)**0.5 #1
Wee = 1*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = 1* -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = 1*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = 1* -1.8 * (N**2*sig_i**2*np.pi*1)**0.5 *rescale  #1.8
mu_e = 1.*rescale
mu_i = .8*rescale

### try mean field
# rescale = 4.9 #N/2  #8 20 30... linear with N
# Wee = 1. *rescale  # recurrent weights
# Wei = -1. *rescale
# Wie = 1. *rescale
# Wii = -1. *rescale
# mu_e = .01 *1
# mu_i = .01 *1

# %% functions for 2D EI network
def phi(x):
    """
    rectified quardratic nonlinearity
    """
    nl = np.where(x > 0, np.tanh(x)*1, 0)
    return nl

def g_kernel(sigma, size=kernel_size):
    """
    Generates a 2D Gaussian kernel.
    """
    sigma = sigma*size
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2) + (y-x)*0.), 
        (size, size)
    )
    return kernel / np.sum(kernel)

def spatial_convolution_(r,k):
    """
    2D spatial convolution given kernel k and neural field r
    """
    gr = jax.scipy.signal.convolve2d(r.squeeze(), k, mode='same',  boundary='fill')
    # gr = sp.signal.convolve2d(r.squeeze(), k, mode='same',  boundary='wrap') #, fillvalue=0,
    return gr

# def convolve_wrap(a1,a2):

#     N = a1.shape[0]
#     a1 = jnp.pad(a1,N,mode='wrap')

#     return convolve_jax(a1, a2,mode='same')[N:2*N,N:2*N]

def spatial_convolution(r, k):
    # Roll the array to wrap around edges
    # r_padded = np.pad(r, ((k.shape[0]//2, k.shape[0]//2), (k.shape[1]//2, k.shape[1]//2)), mode='wrap')
    N = r.shape[0]
    r_padded = np.pad(r, N, mode='wrap')
    
    # Perform convolution using the padded array
    result = jax.scipy.signal.convolve2d(r_padded, k, mode='same')#, boundary='fill')
    
    result = result[N:2*N,N:2*N] #[:r.shape[0], :r.shape[1]] ################################### fix this!!!
    return result

def EI_network(vec_h_ei, dt=0.001, inpt=0):
    ### unpack
    midpoint = len(vec_h_ei) // 2
    he_xy = vec_h_ei[:midpoint].reshape(N,N)
    hi_xy = vec_h_ei[midpoint:].reshape(N,N)
    ### neural dynamics
    re_xy, ri_xy = phi(he_xy), phi(hi_xy)
    ge_conv_re = spatial_convolution(re_xy, g_kernel(sig_e))
    gi_conv_ri = spatial_convolution(ri_xy, g_kernel(sig_i))
    dhe_dt = he_xy + dt/tau_e*( -he_xy + (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) ) + inpt*dt
    dhi_dt = hi_xy + dt/tau_i*( -hi_xy + (Wie*(ge_conv_re) + Wii*(gi_conv_ri) + mu_i) )
    ### put back
    dhei_dt = np.hstack([dhe_dt.reshape(-1), dhi_dt.reshape(-1)])
    return dhei_dt

def compute_beta_t(vec_h_ei):
    midpoint = len(vec_h_ei) // 2
    he_xy = vec_h_ei[:midpoint].reshape(N,N)
    hi_xy = vec_h_ei[midpoint:].reshape(N,N)
    ### neural dynamics
    re_xy, ri_xy = phi(he_xy), phi(hi_xy)
    ge_conv_re = spatial_convolution(re_xy, g_kernel(sig_e))
    gi_conv_ri = spatial_convolution(ri_xy, g_kernel(sig_i))
    measure_mu = np.abs(  (Wee*(ge_conv_re) + Wei*(gi_conv_ri) + mu_e) )
    measure_mu_ex = (Wee*ge_conv_re + mu_e)
    beta_t = measure_mu / measure_mu_ex
    return beta_t

# %% test Jacobian with analytic
# N = 50
# J = jax.random.uniform(key,shape=(N,N))
# def RNN_test(v):
#     dhdt = v*(1-dt)+np.dot(J, np.tanh(v))*dt
#     return dhdt

# hv = jax.random.uniform(key,shape=(N,))
# hsechdt = dt/np.cosh(hv)**2  # derivative of tanh(h)*dt
# D = np.eye(N)*(1-dt) + J*hsechdt

# jacobian_rnn = jax.jacobian(RNN_test)
# D_jax = jacobian_rnn(hv)

# %% numerical tests
dt = 0.001  # 1ms time steps
T = 1.0  # a few seconds of simulation
time = np.arange(0, T, dt)
lt = len(time)
inpt = jax.random.normal(key, shape=(lt,))*1.5 #np.sin(time* 20)*1  ### test input drive
re_xy = np.zeros((N,N, lt))
ri_xy = re_xy*1
re_xy = jax.random.uniform(key,shape=(N,N, lt))*.1
ri_xy = jax.random.uniform(key,shape=(N,N, lt))*.1
he_xy = re_xy*1
hi_xy = ri_xy*1
### x = x.at[1].set(10)
he_xy_t, hi_xy_t = jax.random.uniform(key,shape=(N,N))*.1,jax.random.uniform(key,shape=(N,N))*.1#he_xy[:,:,0], hi_xy[:,:,0]
temp_hei = np.hstack([he_xy_t.reshape(-1), hi_xy_t.reshape(-1)])
for tt in range(lt):
    ### ODE form
    vec_h_ei = np.hstack([he_xy_t.reshape(-1), hi_xy_t.reshape(-1)])
    temp_hei = 0 + EI_network(temp_hei, inpt=inpt[tt])*1
    ### unpack
    midpoint = len(temp_hei) // 2
    he_xy_t = temp_hei[:midpoint].reshape(N,N) #+ inpt[tt]
    hi_xy_t = temp_hei[midpoint:].reshape(N,N)
    he_xy = he_xy.at[:,:,tt].set(he_xy_t)
    hi_xy = hi_xy.at[:,:,tt].set(hi_xy_t)
    # he_xy[:,:,tt] = jax.device_get(he_xy_t) #he_xy_t
    # hi_xy[:,:,tt] = jax.device_get(hi_xy_t) #hi_xy_t
    re_xy = re_xy.at[:,:,tt].set(phi(he_xy_t))
    ri_xy = ri_xy.at[:,:,tt].set(phi(hi_xy_t))

# %%
plt.figure()
plt.plot(re_xy[0,0,:])  # rate
plt.figure()
plt.imshow(re_xy[:,:,lt//2+1])  # spatial pattern

# %% testing jax Jacobian
import jax.numpy as np
jacobian_f = jax.jacobian(EI_network)
# Example state vector x
x = vec_h_ei*1
# Evaluate the Jacobian at x
jacobian_matrix = jacobian_f(x)
print(jacobian_matrix)

# %%
# import numpy as np
# def compute_jacobian(x, f=EI_network, epsilon=1e-6):
#     x = np.asarray(x, dtype=float)
#     n = x.size
#     m = f(x).size
#     jacobian = np.zeros((m, n))
    
#     # Perturb each variable in turn
#     for i in range(n):
#         x_perturbed = x*1 #np.copy(x)
#         x_perturbed[i] += epsilon
#         # test = x_perturbed[i]+epsilon
#         # x_perturbed = x_perturbed[i].set(test)
#         f_x_perturbed = f(x_perturbed)
#         f_x = f(x)
#         # Compute partial derivatives
#         # jacobian = jacobian[:,i].set((f_x_perturbed - f_x) / epsilon)
#         jacobian[:, i] = (f_x_perturbed - f_x) / epsilon

#     return jacobian

# print(compute_jacobian(x))

# %% do this for 2D-EI!
T = .3  # full time in seconds
nLE = N**2*2 #300  # number of LEs
tau = .02 #.2 # characteristic time

### time steps
init_t = .1  # inital time to run
tONS = .01 #0.01 #.005  # time between orthonormalizations ##################################################
nStepTransient = math.ceil(init_t/dt)  # Steps during initial transient
nStep = math.ceil(T/dt)  # Total number of steps
nstepONS = math.ceil(tONS/dt)  # Steps between orthonormalizations ####################################
nStepTransientONS = math.ceil(10) #(nStep/(dt/tONS))  # Steps during transient of ONS
nONS = math.ceil((nStepTransientONS+nStep)/nstepONS) - 1  # Number of ONS steps
stepDisplay = 100

### initialization
vec_h_ei = np.zeros((N**2*2, (nStep + nStepTransient + nStepTransientONS)))  # Preallocate local fields
# h[:, 0] = (g-1)*np.random.randn(N)  # Initialize network state
vec_h_ei = vec_h_ei.at[:,0].set(jax.random.normal(key,shape=(N**2*2,))*.1)
temp_hei = jax.random.normal(key,shape=(N**2*2,))*.1

t = 0
lsidx = -1
LSall = np.zeros((nONS, nLE))  # Initialize array to store convergence of Lyapunov spectrum
normdhAll = np.zeros(nONS)  # Initialize
pAll = np.zeros(nONS)  # participant ratio
LS = np.zeros(nLE)  # Lyapunov spectrum
clv = np.zeros((nONS, nLE)) # tracking full CLV
snap_h = np.zeros((nONS, nLE))  # track activity at the right timing
betas = np.zeros((nONS, nLE//2))  # store beta measurement whenever there is clv calculation (can also do per step...)

### for dynamical systems
q, r = np.linalg.qr(jax.random.normal(key, shape=(N**2*2, nLE)))  # Initialize orthonormal system
Ddiag = np.eye(N)*(1-dt)  # Diagonal elements of Jacobian

lt_jac = nStep + nStepTransient + nStepTransientONS-1
inpt = jax.random.normal(key, shape=(lt_jac,))*0 #1.5  # random input ################ play with correlation !!!!!!!!!!!!!!!!!!!

# %%
### dynamics
for n in range(nStep + nStepTransient + nStepTransientONS-1):
    ### network dynamics
    temp_hei = 0 + EI_network(temp_hei, inpt=inpt[n])*1
    vec_h_ei = vec_h_ei.at[:, n].set(temp_hei)
    # h[:, n+1] = h[:, n]*(1-dt)+np.dot(J, np.tanh(h[:, n]))*dt  # network dynamics
    print(n)
    if (n+1 > nStepTransient):
        ######### Jacobian #########
        # hsechdt = dt/np.cosh(h[:, n])**2  # derivative of tanh(h)*dt
        # D = Ddiag + J*hsechdt  # Jacobian
        D = jacobian_f(temp_hei)
        ############################
        q = np.dot(D, q)  # evolve orthonormal system using Jacobian
        if np.mod(n+1, nstepONS) == 0:
            q, r = np.linalg.qr(q)  # performe QR-decomposition
            if nLE == 1:
                q4 = q*q*q*q
            else:
                q1 = q[:, 0]
                q4 = q1*q1*q1*q1
            lsidx += 1
            clv = clv.at[lsidx,:].set(q1)
            pAll = pAll.at[lsidx].set(1.0/np.sum(q4))
            LSall = LSall.at[lsidx,:].set(np.log(np.abs(np.diag(r)))/nstepONS/dt) # store convergence of Lyapunov spectrum
            snap_h = snap_h.at[lsidx,:].set(temp_hei)
            beta_t = compute_beta_t(temp_hei).reshape(-1)
            betas = betas.at[lsidx,:].set(beta_t)
            if n + 1 > nStepTransientONS + nStepTransient:
                LS += np.log(np.abs(np.diag(r)))  # collect log of diagonal elements or  R for Lyapunov spectrum
                t += nstepONS*dt  # increment time
        
        if np.mod(n + 1, stepDisplay) == 0:  # plot progress
            if n + 1 > nStepTransient + nStepTransientONS:
                PercentageFinished = (n + 1 - nStepTransient - nStepTransientONS)*100/nStep
                print(round(PercentageFinished), ' % done after ', round(3.3), 's SimTime: ', round(dt*(n+1)), ' tau, std(h) =', round(np.std(temp_hei), 2))

Lspectrum = LS/t  # Normalize sum of log of diagonal elements of R by total simulation time

# %%
plt.figure()
plt.plot(1.0*np.arange(nLE)/(nLE)*(1), Lspectrum*tau_i, '.k', label='N=1800')
# plt.plot(1.0*np.arange(nLE)/(nLE)*(nLE/50**2*2), N50_LS, '.b', label='N=5000')
plt.ylabel(r'$\lambda (1/\tau)$', fontsize=20)
plt.xlabel('top sorted values (i/N)', fontsize=20)
plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.ylim([-12,1.9])
# N50_LS

print('entropy rate: ', entropy_rate(Lspectrum))
print('attractor dimension: ', attractor_dim(Lspectrum))

# %% show Layapunov vectors
tt = 30
plt.figure(figsize=(10, 6))
plt.subplot(131)
plt.imshow(clv[tt,:N**2].reshape(N,N)); plt.colorbar(); plt.title('Lyapunov vector')
plt.subplot(132)
plt.imshow(phi(snap_h[tt,N**2:].reshape(N,N))); plt.colorbar(); plt.title('activty')
plt.subplot(133)
plt.imshow(betas[tt,:].reshape(N,N)); plt.colorbar(); plt.title('beta_t')

# %% spatial spectrum
plt.figure()
data4fft = clv[tt,:N**2].reshape(N,N)*1
# data4fft = phi(snap_h[tt,N**2:].reshape(N,N)*1)
data_fft = np.fft.fft2(data4fft)
data_fft_shifted = np.fft.fftshift(data_fft)
magnitude_spectrum = np.abs(data_fft_shifted)
plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')

# %%
plt.figure()
plt.imshow(phi(vec_h_ei[:N**2, 100+(tt+1)*10+2].reshape(N,N))); plt.colorbar(); plt.title('activty')

# %%
### https://github.com/ThomasSavary08/Lyapynov/tree/main

# Definition of a continuous dynamical system, here Lorenz63.
# sigma = 10.
# rho = 28.
# beta = 8./3.
# x0 = np.array([1.5, -1.5, 20.])
# t0 = 0.
# dt = 1e-2

# def f(x,t):
#     res = np.zeros_like(x)
#     res[0] = sigma*(x[1] - x[0])
#     res[1] = x[0]*(rho - x[2]) - x[1]
#     res[2] = x[0]*x[1] - beta*x[2]
#     return res

# def jac(x,t):
#     res = np.zeros((x.shape[0], x.shape[0]))
#     res[0,0], res[0,1] = -sigma, sigma
#     res[1,0], res[1,1], res[1,2] = rho - x[2], -1., -x[0]
#     res[2,0], res[2,1], res[2,2] = x[1], x[0], -beta
#     return res

# Lorenz63 = ContinuousDS(x0, t0, f, jac, dt)
# lorenz_time = Lorenz63.forward(10**3, True)
# plt.figure()
# plt.plot(lorenz_time)

# # Compute mLCE
# mLCE_, history = mLCE(Lorenz63, 0, 10**4, True)
# # Print mLCE
# print("mLCE: {:.3f}".format(mLCE_))
# # Plot of mLCE evolution
# plt.figure(figsize = (10,6))
# plt.plot(history[:5000])
# plt.xlabel("Number of time steps")
# plt.ylabel("mLCE")
# plt.title("Evolution of the mLCE for the first 5000 time steps")
# plt.show()

# %%
# def CLV(system: EI_network, p : int, n_forward : int, n_A : int, n_B : int, n_C : int, traj : bool, check = False):
#     '''
#     Compute CLV.
#         Parameters:
#             system (DynamicalSystem): Dynamical system for which we want to compute the mLCE.
#             p (int): Number of CLV to compute.
#             n_forward (int): Number of steps before starting the CLV computation. 
#             n_A (int): Number of steps for the orthogonal matrice Q to converge to BLV.
#             n_B (int): Number of time steps for which Phi and R matrices are stored and for which CLV are computed.
#             n_C (int): Number of steps for which R matrices are stored in order to converge A to A-. 
#             traj (bool): If True return a numpy array of dimension (n_B,system.dim) containing system's trajectory at the times CLV are computed.
#         Returns:
#             CLV (List): List of numpy.array containing CLV computed during n_B time steps.
#             history (numpy.ndarray): Trajectory of the system during the computation of CLV.
#     '''
#     # Forward the system before the computation of CLV
#     system.forward(n_forward, False)

#     # Make W converge to Phi
#     W = np.eye(system.dim)[:,:p]
#     for _ in range(n_A):
#         W = system.next_LTM(W)
#         W, _ = np.linalg.qr(W)
#         system.forward(1, False)
    
#     # We continue but now Q and R are stored to compute CLV later
#     Phi_list, R_list1 = [W], []
#     if traj:
#         history = np.zeros((n_B+1, system.dim))
#         history[0,:] = system.x
#     if check:
#         copy = system.copy()
#     for i in range(n_B):
#         W = system.next_LTM(W)
#         W, R = np.linalg.qr(W)
#         Phi_list.append(W)
#         R_list1.append(R)
#         system.forward(1, False)
#         if traj:
#             history[i+1,:] = system.x
    
#     # Now we only store R to compute A- later
#     R_list2 = []
#     for _ in range(n_C):
#         W = system.next_LTM(W)
#         W, R = np.linalg.qr(W)
#         R_list2.append(R)
#         system.forward(1, False)
    
#     # Generate A make it converge to A-
#     A = np.triu(np.random.rand(p,p))
#     for R in reversed(R_list2):
#         C = np.diag(1. / np.linalg.norm(A, axis = 0))
#         B = A @ C
#         A = np.linalg.solve(R, B)
#     del R_list2

#     # Compute CLV
#     CLV = [Phi_list[-1] @ A]
#     for Q, R in zip(reversed(Phi_list[:-1]), reversed(R_list1)):
#         C = np.diag(1. / np.linalg.norm(A, axis = 0))
#         B = A @ C
#         A = np.linalg.solve(R, B)
#         CLV_t = Q @ A
#         CLV.append(CLV_t / np.linalg.norm(CLV_t, axis = 0))
#     del R_list1
#     del Phi_list
#     CLV.reverse()

#     if traj:
#         if check:
#             return CLV, history, copy
#         else:
#             return CLV, history
#     else:
#         if check:
#             return CLV, copy
#         else:
#             return CLV
        
# %% # Compute CLV
# CLV_, traj, checking_ds = CLV(Lorenz63, 3, 0, 10**5, 10**6, 10**5, True, check = True)
# # Check CLV
# LCE_check = np.zeros((Lorenz63.dim,))
# for i in range(len(CLV_)):
#     W = CLV_[i]
#     init_norm = np.linalg.norm(W, axis = 0)
#     W = checking_ds.next_LTM(W)
#     norm = np.linalg.norm(W, axis = 0)
#     checking_ds.forward(1, False)
#     LCE_check += np.log(norm / init_norm) / checking_ds.dt
# LCE_check = LCE_check / len(CLV_)

# print("Average of first local Lyapunov exponent: {:.3f}".format(LCE_check[0]))
# print("Average of second local Lyapunov exponent: {:.3f}".format(LCE_check[1]))
# print("Average of third local Lyapunov exponent: {:.3f}".format(LCE_check[2]))