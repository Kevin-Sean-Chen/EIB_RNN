% Chaotic_2D_EI_network
close all
clc

%% setup network
L = 50;
N = L^2;
tau_e = 0.005;  % in seconds
sig_e = 0.1;  % in between 0-1
tau_i = 3*tau_e;
sig_i = 2*tau_e;
%%%% emperical note for parameters %%%
%%% 5, 0.14  ### grid parameter
%%% 15, 0.2  ### chaos parameter
%%% 10, 0.11 ### waves/strips
%%% 8,  0.2  ### blinking

%% scaling for EI network

rescale = 2. ##(N*sig_e*np.pi*1)**0.5 #1
# Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
# Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# Wie = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
# Wii = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
# mu_e = .7*rescale #*(N*sig_i*np.pi*1)**0.5 *rescale  #1e-8#.001*1  # offset
# mu_i = .7*rescale #*(N*sig_i*np.pi*1)**0.5 *rescale  #1e-8#.001*1

Wee = 1.*(N**2*sig_e**2*np.pi*1)**0.5 *rescale  # recurrent weights
Wei = -2.*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
Wie = .99*(N**2*sig_e**2*np.pi*1)**0.5 *rescale
Wii = -1.8*(N**2*sig_i**2*np.pi*1)**0.5 *rescale
mu_e = 1.*rescale
mu_i = .8*rescale