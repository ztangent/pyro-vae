"""Multimodal Deep Markov Model written in Pyro."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

class ProductOfExperts(nn.Module):
    """
    Return parameters for product of independent Gaussian experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param loc: M x D for M experts
    @param scale: M x D for M experts
    """
    def forward(self, loc, scale, eps=1e-8):
        scale = scale + eps # numerical constant for stability
        # precision of i-th Gaussian expert (T = 1/sigma^2)
        T = 1. / scale
        product_loc = torch.sum(loc * T, dim=0) / torch.sum(T, dim=0)
        product_scale = 1. / torch.sum(T, dim=0)
        return product_loc, product_scale

class MDMM(nn.Module):
    """
    This class encapsulates the parameters (neural networks), models & guides
    needed to train a multimodal deep Markov model.

    @param z_dim: integer
                  size of the tensor representing the latent random variable z
                  
    """
    def __init__(self, z_dim, z_trans, modalities={},
                 use_cuda=False, name="mdmm"):
        super(MVAE, self).__init__()
        self.name = name
        self.z_dim = z_dim
        self.z_trans = z_trans
        self.z_init = nn.Parameter(torch.zeros(z_dim))
        self.experts = ProductOfExperts()
        self.modalities = []
        self.dims = dict()
        self.dists = dict()
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.loss_mults = dict()
        for name, params in modalities.iteritems():
            self.add_modality(name, *params)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def add_modality(self, name, dims, dist, encoder, decoder, loss_mult=1.0):
        self.modalities.append(name)
        self.dims[name] = dims
        self.dists[name] = dist
        self.encoders[name] = encoder
        self.decoders[name] = decoder
        self.loss_mults[name] = loss_mult
        if self.use_cuda:
            encoder.cuda()
            decoder.cuda()
                
    def model(self, inputs={}, seq_lengths=torch.tensor([]),
              annealing_beta=1.0):
        # Extract batch size and max sequence length
        batch_size = len(seq_lengths)
        t_max = int(max(seq_lengths))
        
        # Register this pytorch module and all of its sub-modules with pyro
        pyro.module(self.name, self)

        # Initialize the previous latent state to z_init
        z_prev = self.z_init.expand(batch_size, self.z_dim)
        
        # Use iarange to vectorize over mini-batch
        with pyro.iarange("data", batch_size):
            for t in range(t_max):
                # Construct mask for current timestep
                mask = (t < seq_lengths).float()
                
                # Get Gaussian parameters for z_cur
                z_loc, z_scale = self.z_trans(z_prev)
                            
                # Sample z_cur
                z_dist = dist.Normal(z_loc, z_scale).mask(mask).independent(1)
                with poutine.scale(scale=annealing_beta):
                    z_cur = pyro.sample("latent_{}".format(t), z_dist)

                for m in self.modalities:
                    # Decode the latent code z for each modality
                    m_dist_params = self.decoders[m].forward(z_cur)
                    m_dist = self.dists[m](*m_dist_params).mask(mask).\
                             independent(1)
                    # Score against observed inputs if given
                    if m not in inputs:
                        continue
                    with poutine.scale(scale=self.loss_mults[m]):
                        pyro.sample("obs_{}_{}".format(m, t), m_dist,
                                    obs=inputs[m][:,t,:])

                # Continue to next time step
                z_prev = z_cur
