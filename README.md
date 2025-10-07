# EIB_RNN
Excitation-Inhibition Balanced RNN models, with two-dimensional connectivity

## Basic formulism
$$\tau_e \frac{\partial h_e(x,t)}{\partial t} = -h_e + W_{ee}g_e*\phi(h_e) + W_{ei}g_i*\phi(h_i) + \mu_e$$

$$\tau_i \frac{\partial h_i(x,t)}{\partial t} = -h_i + W_{ie}g_e*\phi(h_e) + W_{ii}g_i*\phi(h_i) + \mu_i$$

where $\phi$ is now ReLU, $g$ as spatial Gaussian kernels, and weights are chosen to suffice balance condition and later scaled with connectivitiy $\sqrt(K)$

## Current investigation
- The role of balance and $K$ in spatiotemporal pattern
- Computation and input-driven patterns
- Extensions: Disorder/non-locality, learning/development, adaptation 
