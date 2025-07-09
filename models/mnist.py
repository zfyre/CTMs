"""
Contains the NLMs, Synapse and Backbone Models for MNIST
"""
import math
import torch
import torch.nn as nn
from typing import Tuple

class UNET(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super(UNET, self).__init__()

        self.conv_channels = [64, 128, 256, 512]

        self.conv1 = nn.ModuleList([
            nn.Sequential(
                nn.LazyConv2d(channels, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3),
                nn.ReLU()
            )
            for channels in self.conv_channels
        ])
        self.bottleneck = nn.Sequential(
            nn.Sequential(
                nn.LazyConv2d(1024, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, kernel_size=3),
                nn.ReLU()
            )
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2) # (H, W) -> (H//2, W//2)
        self.up_conv = nn.ModuleList([
            nn.LazyConvTranspose2d(channels, kernel_size=2, stride=2) 
            for channels in reversed(self.conv_channels)
        ])
        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.LazyConv2d(channels, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3),
                nn.ReLU()
            )
            for channels in reversed(self.conv_channels)
        ])
        self.fc = nn.LazyConv2d(in_channels, kernel_size=1)

    def _get_crop_idx(self, init_shape: Tuple[int, int], fin_shape: Tuple[int, int]):
        start = tuple((init_i - fin_i) // 2 for init_i, fin_i in zip(init_shape, fin_shape))
        end = tuple(start_i + fin_i for start_i, fin_i in zip(start, fin_shape))
        return start, end

    def forward(self, x: torch.Tensor):

        skip_conn = []

        # Down
        for conv in self.conv1:
            x_skip = conv(x)
            skip_conn.append(x_skip)
            x = self.maxpool(x_skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Up
        for (conv, conv_transpose, x_skipped) in zip(self.conv2, self.up_conv, reversed(skip_conn)):

            # Upsample the current x
            upsampled_x = conv_transpose(x)

            # Cropped the x_skipped to upsampled_x dim TODO: Better would to rather resize, what if the upsampled is bigger?!
            _, _, h, w = upsampled_x.shape
            _, _, H, W = x_skipped.shape
            start, end = self._get_crop_idx(init_shape=(H,W), fin_shape=(h, w))
            x_skipped_cropped = x_skipped[:, :, (start[0]):(end[0]), (start[1]):(end[1])]

            # Concatenate along the channel dims & update
            x = torch.concat([
                upsampled_x,
                x_skipped_cropped
            ], dim=1)

            # Apply the final conv
            x = conv(x)

        # Final Layer
        x = self.fc(x)

        return x

class BackBone(nn.Module):
    def __init__(self, d_input: int):
        super(BackBone, self).__init__()

        self.d_input = d_input
        self.simple = nn.Sequential( # Across these layers the Height and Weight remains same by design
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
        super(SuperLinear, self).__init__()
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
        super(NLM, self).__init__()

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
        super(Synapses, self).__init__()
        self.d_model = d_model
        
        self.simple = nn.Sequential(
            nn.LazyLinear(2*d_model), # [d_model + d_input] -> [2*d_model]
            nn.GLU(),
            nn.LayerNorm(d_model)
        )
    def forward(self, x: torch.Tensor):
        return self.simple(x)

