import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

### TO-DO ###
# how to combine this with other functions?
#############
class Relu2DReservoirRNN(nn.Module):
    def __init__(self, N, T, output_dim, device, relu2D_params):
        super().__init__()
        self.N = N
        self.T = T
        self.output_dim = output_dim
        self.device = device
        self.relu2D_params = relu2D_params
        # Readout weights
        self.W_out = nn.Parameter(torch.randn(N*N, output_dim, device=device) * 0.1)

    def forward(self, input_pattern):
        # input_pattern: [N, N, T]
        N = self.N
        T = self.T
        re = torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
        ri = torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
        re_all = []
        for t in range(T):
            # Add input as external drive to excitatory population
            u = self.relu2D_params['u']  # Read baseline u
            u[0] = u[0]*0 + input_pattern[:, :, t].cpu().numpy()*10  # Add input 2D array at time t
            re, ri = relu2D_step(
                re.cpu().numpy(), ri.cpu().numpy(), N,
                self.relu2D_params['dt'], 1, self.relu2D_params['ntype'],
                self.relu2D_params['K'], self.relu2D_params['tau'],
                u, self.relu2D_params['J0'], self.relu2D_params['sigma'],
                self.relu2D_params['J2'], self.relu2D_params['J3']
            )
            re = torch.tensor(re, device=self.device, dtype=torch.float32)
            ri = torch.tensor(ri, device=self.device, dtype=torch.float32)
            re_all.append(torch.relu(re)) ### testing with input nonlinearity

        re_all = torch.stack(re_all, dim=-1)  # [N, N, T]
        # Readout: flatten spatial, shape [N*N, T]
        readout = re_all.reshape(N*N, T)
        out = readout.T @ self.W_out  # [T, output_dim]
        return out.T, re_all  # [output_dim, T], [N, N, T]