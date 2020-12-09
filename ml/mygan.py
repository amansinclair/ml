"""Barebones implementation of StyleGAN2 Generator Network. Simplified operations used throughout.
'Analyzing and Improving the Image Quality of StyleGAN' - https://arxiv.org/pdf/1912.04958.pdf and 
'A Style-Based Generator Architecture for Generative Adversarial Networks' - https://arxiv.org/pdf/1812.04948.pdf.
Implementation based on https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py,
https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb and
https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import pickle

import numpy as np
from pathlib import Path
import time


class Generator(nn.Module):
    """A rigid implementation of StyleGan2."""

    def __init__(self, img_size, latent_dim, n_map_layers, n_channels):
        super().__init__()
        self.n_layers = (int(np.log2(img_size) - 1) * 2) - 1
        self.latent_dim = latent_dim
        self.mapping = Mapping(latent_dim, n_map_layers, self.n_layers)
        self.synthesis = Synthesis(latent_dim, n_channels, self.n_layers, img_size)

    def forward(self, batch_size=1):
        latent_z = torch.randn(batch_size, self.latent_dim)
        latent_w = self.mapping(latent_z)
        return self.synthesis(latent_w)


class Mapping(nn.Module):
    """Encodes latent vector Z to W and projects to match number of synthesizer layers."""

    def __init__(self, latent_dim, n_map_layers, n_style_layers):
        super().__init__()
        self.n_style_layers = n_style_layers
        blocks = [FCLayer(latent_dim) for i in range(n_map_layers)]
        self.layers = nn.Sequential(PixelNormLayer(), *blocks)

    def forward(self, latent_z):
        latent_w = self.layers(latent_z)
        latent_w = latent_w.expand(self.n_style_layers, -1, -1)
        return latent_w


class PixelNormLayer(nn.Module):
    """Norm Layer taken from 'progressive growing of gans for improved quality, stability, and variation' - https://arxiv.org/pdf/1710.10196.pdf."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class FCLayer(nn.Module):
    """Fully connected layer combined with leaky relu."""

    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.fc(x))


class Synthesis(nn.Module):
    """StyleGan2 Synthesis network with the same number of channels in all blocks."""

    def __init__(self, latent_dim, n_channels, n_layers, img_size):
        super().__init__()
        self.n_layers = n_layers
        self.img_size = img_size
        self.n_channels = n_channels
        self.blocks = []
        for i in range(n_layers):
            upsample = bool(i % 2)
            self.blocks.append(SynBlock(latent_dim, n_channels, upsample))
        self.to_rgb = RGBConv(n_channels)

    def forward(self, latent_w, noise=None):
        """latent_w[style_layers, bs, latent_dim], noise[style_layers, bs, img**2]."""
        batch_size = latent_w.shape[1]
        x = torch.zeros((batch_size, self.n_channels, 4, 4))
        if noise == None:
            noise = torch.randn(len(self.blocks), batch_size, self.img_size ** 2)
        for block, w, n in zip(self.blocks, latent_w, noise):
            x = block(x, w, n)
        rgb = self.to_rgb(x)
        return rgb


class SynBlock(nn.Module):
    """A single conv layer with optional upscale."""

    def __init__(self, latent_dim, n_channels, upsample=True):
        super().__init__()
        self.upsample = Upsample() if upsample else None
        self.A = nn.Linear(latent_dim, n_channels)
        self.B = nn.Linear(1, n_channels)
        self.conv = Conv2dMod(n_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, latent_w, noise):
        """latent_w[bs, latent_dim], noise[bs, img**2]."""
        if self.upsample:
            x = self.upsample(x)
        bs, ch, h, w = x.shape
        noise_size = h * w
        style = self.A(latent_w)
        noise = self.B(noise[:, :noise_size].unsqueeze(-1)).permute(0, 2, 1)
        noise = noise.reshape(bs, -1, h, w)
        x = self.conv(x, style)
        return self.act(x + noise)


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, x):
        return self.upsample(x)


class Conv2dMod(nn.Module):
    """Block with Mod, Demod, Conv operations. Channels in and out are the same."""

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.weight = torch.nn.Parameter(torch.randn(n_channels, n_channels, 3, 3))
        nn.init.kaiming_normal_(
            self.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
        )
        self.bias = torch.nn.Parameter(torch.zeros(n_channels))

    def forward(self, x, style):
        """style[bs, n_channels]."""
        bs, _, im_size, _ = x.shape
        mod_weights = self.weight * style.view(bs, 1, self.n_channels, 1, 1)
        rstd = torch.rsqrt((mod_weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
        demod_weights = mod_weights * rstd.view(bs, self.n_channels, 1, 1, 1)
        demod_weights = demod_weights.view(bs * self.n_channels, self.n_channels, 3, 3)
        x = x.view(1, bs * self.n_channels, im_size, im_size)
        out = F.conv2d(x, demod_weights, padding=1, groups=bs)
        return out.view(bs, self.n_channels, im_size, im_size)


class RGBConv(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


g = Generator(64, 32, 4, 8)
img = g()
print(img.shape)
