"""Multimodal VAE written in Pyro, with arbitrary number of modalities.

Adapted from https://github.com/wenxuanliu/multimodal-vae/ and 
https://github.com/mhw32/multimodal-vae-public.
"""

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

class MVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks), models & guides
    needed to train a multimodal variational auto-encoder.

    @param z_dim: integer
                  size of the tensor representing the latent random variable z
                  
    """
    def __init__(self, z_dim, z_prior_loc=0.0, z_prior_scale=1.0,
                 modalities={}, use_cuda=False, name="mvae"):
        super(MVAE, self).__init__()
        self.name = name
        self.z_dim = z_dim
        self.z_prior_loc = z_prior_loc
        self.z_prior_scale = z_prior_scale
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
            
    def forward(self, inputs, batch_size=1):
        z_loc, z_scale  = self.infer(inputs)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
        outputs = {m: self.decoders[m].forward(z) for m in self.modalities}
        return outputs

    def infer(self, inputs={}, batch_size=1):
        # Initialize the universal prior
        z_loc = z_prior_loc * torch.zeros(1, batch_size, self.z_dim)
        z_scale = z_prior_scale * torch.ones(1, batch_size, self.z_dim)
        if self.use_cuda:
            z_loc, z_scale = z_loc.cuda(), z_scale.cuda()

        # Compute posteriors for each modality present in inputs
        for m in self.modalities:
            if m not in inputs or inputs[m] is None:
                continue
            z_loc_m, z_scale_m = self.encoders[m].forward(inputs[m])
            z_loc = torch.cat((z_loc, z_loc_m.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, z_scale_m.unsqueeze(0)), dim=0)
            
        # Product of experts to combine Gaussians
        z_loc, z_scale = self.experts(z_loc, z_scale)
        return z_loc, z_scale
    
    def model(self, inputs={}, batch_size=0, annealing_beta=1.0):
        # Register this pytorch module and all of its sub-modules with pyro
        pyro.module(self.name, self)
                
        with pyro.iarange("data", batch_size):
            # Initialize the universal prior
            z_loc = z_prior_loc * torch.zeros(1, batch_size, self.z_dim)
            z_scale = z_prior_scale * torch.ones(1, batch_size, self.z_dim)
            if self.use_cuda:
                z_loc, z_scale = z_loc.cuda(), z_scale.cuda()
            
            # Sample from prior
            z_dist = dist.Normal(z_loc, z_scale).independent(1)
            with poutine.scale(scale=annealing_beta):
                z = pyro.sample("latent", z_dist)

            outputs = {}
            for m in self.modalities:
                # Decode the latent code z for each modality
                m_dist_params = self.decoders[m].forward(z)
                m_dist = self.dists[m](*m_dist_params).independent(1)
                outputs[m] = m_dist_params
                # Score against observed inputs if given
                if m not in inputs:
                    continue
                m_obs = "obs_" + m
                with poutine.scale(scale=self.loss_mults[m]):
                    pyro.sample(m_obs, m_dist,
                                obs=inputs[m].reshape(-1, self.dims[m]))
                                
        # Return the output distributions for visualization, etc.
        return outputs
    
    def guide(self, inputs={}, batch_size=0, annealing_beta=1.0):
        # Register this pytorch module and all of its sub-modules with pyro
        pyro.module(self.name, self)
        
        with pyro.iarange("data", batch_size):
            # Initialize the universal prior
            z_loc = z_prior_loc * torch.zeros(1, batch_size, self.z_dim)
            z_scale = z_prior_scale * torch.ones(1, batch_size, self.z_dim)
            if self.use_cuda:
                z_loc, z_scale = z_loc.cuda(), z_scale.cuda()

            # Compute posteriors for each modality present in inputs
            for m in self.modalities:
                if m not in inputs or inputs[m] is None:
                    continue
                z_loc_m, z_scale_m = self.encoders[m].forward(inputs[m])
                z_loc = torch.cat((z_loc, z_loc_m.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, z_scale_m.unsqueeze(0)), dim=0)
            
            # Product of experts to combine Gaussians
            z_loc, z_scale = self.experts(z_loc, z_scale)

            # Sample the latent code z
            with poutine.scale(scale=annealing_beta):
                pyro.sample("latent",
                            dist.Normal(z_loc, z_scale).independent(1))
    
