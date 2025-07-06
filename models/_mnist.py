"""
Contains the NLMs, Synapse and Backbone Models for MNIST
"""
import math
import torch
import torch.nn as nn


class BackBone(nn.Module):
    def __init__(self, d_input: int):
        super(BackBone).__init__()

        self.d_input = d_input
        self.simple = nn.Sequential(
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W) -> (B, d_input, h, w)
        # NOTE If h==H & w==W then each "patch" is just a pixel, hence easy to visualize the attention.
        # NOTE We output the number of channels equal to the CTM d_input even though we are applying kV projection there because typically in transformers also while calculating the KV the token embeddings dimensions are kept same

        return self.simple(x)


class Identity(nn.Module):
    """Identity Module."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Squeeze(nn.Module):
    """Squeeze Module."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class SuperLinear(nn.Module):
    """SuperLinear Layer: Implements Neuron-Level Models (NLMs) for the CTM."""
    def __init__(self, in_dims, out_dims, N):
        super().__init__()
        self.in_dims = in_dims
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))

    def forward(self, x):
            out = torch.einsum('BDM,MHD->BDH', x, self.w1) + self.b1
            out = out.squeeze(-1)
            return out


class NLM(nn.Module):
    def __init__(self, d_model:int, d_memory: int, memory_hidden_dims: int = 8):
        super(NLM).__init__()

        self.d_model = d_model
        self.d_memory = d_memory

        self.history_processor = nn.Sequential(
            SuperLinear(in_dims=d_memory, out_dims=2 * memory_hidden_dims, N=d_model),
            nn.GLU(),
            SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model),
            nn.GLU(),
            Squeeze(-1)
        )

    def forward(self, x: torch.Tensor):
        return self.history_processor(x)


class Synapses(nn.Module):
    def __init__(self, d_model: int):
        self.d_model = d_model

        self.simple = nn.Sequential(
            nn.LazyLinear(2*d_model), # [d_model + d_input] -> [2*d_model]
            nn.GLU(),
            nn.LayerNorm(d_model)
        )
    def forward(self, x: torch.Tensor):
        return self.simple(x)

