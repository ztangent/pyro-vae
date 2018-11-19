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

from utils import ProductOfExperts

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
        self.encoders = dict() # nn.ModuleDict not available in torch 0.4.0
        self.decoders = dict()
        self.loss_mults = dict()
        self.use_logits = dict()
        for name, params in modalities.iteritems():
            self.add_modality(name, *params)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def add_modality(self, name, dims, dist, encoder, decoder,
                     loss_mult=1.0, use_logits=False):
        self.modalities.append(name)
        self.dims[name] = dims
        self.dists[name] = dist
        self.encoders[name] = encoder
        self.add_module("{}_enc".format(name), encoder)
        self.decoders[name] = decoder
        self.add_module("{}_dec".format(name), decoder)
        self.loss_mults[name] = loss_mult
        self.use_logits[name] = use_logits
        if self.use_cuda:
            encoder.cuda()
            decoder.cuda()
            
    def forward(self, inputs={}, batch_size=1):
        # Sample z conditioned on the inputs
        z_loc, z_scale  = self.infer(inputs, batch_size)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
        # Decode z and reconstruct each of the modalities
        outputs, params = {}, {}
        with pyro.iarange("data", batch_size):
            for m in self.modalities:
                params[m] = self.decoders[m].forward(z)
                if type(params[m]) is tuple:
                    # Unpack parameters if there are multiple
                    m_dist = self.dists[m](*params[m]).independent(1)
                elif self.use_logits[m]:
                    # Use logits for numerical stability
                    m_dist = self.dists[m](logits=params[m]).independent(1)
                else:
                    # Assume only one parameter
                    m_dist = self.dists[m](params[m]).independent(1)
                m_obs = "obs_" + m
                outputs[m] = pyro.sample(m_obs, m_dist)
        return outputs, params

    def infer(self, inputs={}, batch_size=1):
        # Initialize the universal prior
        z_loc = self.z_prior_loc * torch.ones(1, batch_size, self.z_dim)
        z_scale = self.z_prior_scale * torch.ones(1, batch_size, self.z_dim)
        if self.use_cuda:
            z_loc, z_scale = z_loc.cuda(), z_scale.cuda()

        with pyro.iarange("data", batch_size):
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
            z_loc = self.z_prior_loc * torch.ones(batch_size, self.z_dim)
            z_scale = self.z_prior_scale * torch.ones(batch_size, self.z_dim)
            if self.use_cuda:
                z_loc, z_scale = z_loc.cuda(), z_scale.cuda()
            
            # Sample from prior
            z_dist = dist.Normal(z_loc, z_scale).independent(1)
            with poutine.scale(scale=annealing_beta):
                z = pyro.sample("latent", z_dist)

            for m in self.modalities:
                # Decode the latent code z for each modality
                m_dist_params = self.decoders[m].forward(z)
                # Score against observed inputs if given
                if m not in inputs:
                    continue
                if type(m_dist_params) is tuple:
                    # Unpack parameters if there are multiple
                    m_dist = self.dists[m](*m_dist_params).independent(1)
                elif self.use_logits[m]:
                    # Use logits for numerical stability
                    m_dist = self.dists[m](logits=m_dist_params).independent(1)
                else:
                    # Assume only one parameter
                    m_dist = self.dists[m](m_dist_params).independent(1)
                m_obs = "obs_" + m
                with poutine.scale(scale=self.loss_mults[m]):
                    pyro.sample(m_obs, m_dist, obs=inputs[m])
            
    def guide(self, inputs={}, batch_size=0, annealing_beta=1.0):
        # Register this pytorch module and all of its sub-modules with pyro
        pyro.module(self.name, self)
        
        with pyro.iarange("data", batch_size):
            # Initialize the universal prior
            z_loc = self.z_prior_loc * torch.ones(batch_size, self.z_dim)
            z_scale = self.z_prior_scale * torch.ones(batch_size, self.z_dim)
            z_loc, z_scale = z_loc.unsqueeze(0), z_scale.unsqueeze(0)
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
    
