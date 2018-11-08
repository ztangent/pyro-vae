"""MNIST example for the multimodal VAE.
Adapted from https://github.com/mhw32/multimodal-vae-public/"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn

import pyro.distributions as dist

from mvae import MVAE

class Swish(nn.Module):
    """ReLu replacement: https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)

class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.fc1   = nn.Linear(784, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc31  = nn.Linear(512, n_latents)
        self.fc32  = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.fc1   = nn.Linear(n_latents, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, 512)
        self.fc4   = nn.Linear(512, 784)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)  # NOTE: no sigmoid here. See train.py

class TextEncoder(nn.Module):
    """Parametrizes q(z|y).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()
        self.fc1   = nn.Embedding(10, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc31  = nn.Linear(512, n_latents)
        self.fc32  = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.fc1   = nn.Linear(n_latents, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, 512)
        self.fc4   = nn.Linear(512, 10)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)

class MnistMVAE(MVAE):

    def __init__(self, z_dim, z_prior_loc=0.0, z_prior_scale=1.0,
                 lambda_image=1.0, lambda_text=1.0, use_cuda=False):
        super(MnistMVAE, self).__init__(z_dim,
                                        z_prior_loc=z_prior_loc,
                                        z_prior_scale=z_prior_scale,
                                        name="mnist_mvae")
        self.add_modality('image', 784, dist.Bernoulli,
                          ImageEncoder(z_dim), ImageDecoder(z_dim),
                          lambda_image)
        self.add_modality('text', 10, dist.OneHotCategorical,
                          TextEncoder(z_dim), TextDecoder(z_dim),
                          lambda_text)
        if use_cuda:
            self.use_cuda = True
            self.cuda()
