"""Multimodal Deep Markov Model written in Pyro."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from utils import ProductOfExperts

class MDMM(nn.Module):
    """
    This class encapsulates the parameters (neural networks), models & guides
    needed to train a multimodal deep Markov model.

    @param z_dim: integer
                  size of the tensor representing the latent random variable z
                  
    """
    def __init__(self, z_dim, z_forward, z_backward, modalities={},
                 use_cuda=False, name="mdmm"):
        super(MDMM, self).__init__()
        self.name = name
        self.z_dim = z_dim
        self.z_forward = z_forward
        self.z_backward = z_backward
        self.z_init = nn.Parameter(torch.zeros(z_dim))
        self.z_end = nn.Parameter(torch.zeros(z_dim))
        self.z_infer = nn.Parameter(torch.zeros(z_dim))
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
                
    def model(self, inputs={}, masks={}, seq_lengths=torch.tensor([]),
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
                # Mask out timesteps greater than the sequence length
                mask = (t < seq_lengths).float()
                
                # Get Gaussian parameters for z_cur
                z_loc, z_scale = self.z_forward(z_prev)
                            
                # Sample z_cur
                z_dist = dist.Normal(z_loc, z_scale).mask(mask).independent(1)
                with poutine.scale(scale=annealing_beta):
                    z_cur = pyro.sample("latent_{}".format(t), z_dist)

                for m in self.modalities:
                    # Score against observed inputs only if modality is present
                    if m not in inputs or inputs[m] is None:
                        continue
                    # Mask out timesteps where modality is absent
                    mask_m = m in masks ? mask * masks[m][:,t] : mask
                    # Decode the latent code z for each modality
                    m_dist_params = self.decoders[m].forward(z_cur)
                    m_dist = self.dists[m](*m_dist_params).mask(mask_m).\
                             independent(1)
                    input_m = inputs[m][:,t,:]
                    with poutine.scale(scale=self.loss_mults[m]):
                        pyro.sample("obs_{}_{}".format(m, t), m_dist,
                                    obs=input_m)

                # Continue to next time step
                z_prev = z_cur

    def guide_est(self, inputs={}, masks={}, seq_lengths=torch.tensor([]),
                  annealing_beta=1.0):
        # Extract batch size and max sequence length
        batch_size = len(seq_lengths)
        t_max = int(max(seq_lengths))

        # Create buffer for backward z-estimates
        z_est = torch.tensor([batch_size, t_max, self.z_dim])
        if self.use_cuda:
            z_est.cuda()
        
        # Initialize the next latent state estimate to z_end
        z_next = self.z_end.expand(batch_size, self.z_dim)

        # Backward pass to infer estimates of latent states
        for t in range(t_max-1, -1, -1):
            # Construct mask for current timestep
            mask_z = (t < seq_lengths).float()
            mask_all = torch.tensor(mask_z).unsqueeze(0)
            
            # Get Gaussian parameters for z_cur
            z_loc, z_scale = self.z_backward(z_next)
            z_loc, z_scale = z_loc.unsqueeze(0), z_scale.unsqueeze(0)
            
            # Compute posteriors for each modality present in inputs
            for m in self.modalities:
                if m not in inputs or inputs[m] is None:
                    continue
                # Mask out timesteps where modality is absent
                mask_m = m in masks ? mask_z * masks[m][:,t] : mask_z
                mask_all = torch.cat((mask_all, mask_m.unsqueeze(0)), dim=0)
                # Get mean and variance from encoder for modality m
                input_m = inputs[m][:,t,:]
                z_loc_m, z_scale_m = self.encoders[m].forward(input_m)
                z_loc = torch.cat((z_loc, z_loc_m.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, z_scale_m.unsqueeze(0)), dim=0)
            
            # Product of experts to combine Gaussians
            z_loc, z_scale = self.experts(z_loc, z_scale, mask_all)

            # Reuse z_next for batches where t > seq_length
            z_loc = z_loc * mask_z.unsqueeze(-1)
            z_loc = z_loc + z_next * (1 - mask_z.unsqueeze(-1))
            
            # Store z_loc as estimate of latent state
            z_est[:, t, :] = z_loc
            z_next = z_loc

        return z_est
                
    def guide(self, inputs={}, masks={}, seq_lengths=torch.tensor([]),
              annealing_beta=1.0):
        # Extract batch size and max sequence length
        batch_size = len(seq_lengths)
        t_max = int(max(seq_lengths))

        # Register this pytorch module and all of its sub-modules with pyro
        pyro.module(self.name, self)

        # Compute backward latent state estimates
        z_est = self.guide_est(inputs, masks, seq_lengths, annealing_beta)

        # Initialize the latent state to z_infer
        z_prev = self.z_infer.expand(batch_size, self.z_dim)
        
        # Use iarange to vectorize over mini-batch        
        with pyro.iarange("data", batch_size):
            for t in range(t_max):
                # Construct mask for current timestep
                mask_z = (t < seq_lengths).float()
                mask_all = torch.tensor(mask_z).unsqueeze(0)
                
                # Get Gaussian parameters from p(z_cur | z_prev)
                z_loc, z_scale = self.z_forward(z_prev)
                z_loc, z_scale = z_loc.unsqueeze(0), z_scale.unsqueeze(0)
                
                # Compute posteriors for each modality present in inputs
                for m in self.modalities:
                    if m not in inputs or inputs[m] is None:
                        continue
                    # Mask out timesteps where modality is absent
                    mask_m = m in masks ? mask_z * masks[m][:,t] : mask_z
                    mask_all = torch.cat((mask_all,mask_m.unsqueeze(0)),dim=0)
                    # Get mean and variance from encoder for modality m
                    input_m = inputs[m][:,t,:]
                    z_loc_m, z_scale_m = self.encoders[m].forward(input_m)
                    z_loc = torch.cat((z_loc,z_loc_m.unsqueeze(0)),dim=0)
                    z_scale = torch.cat((z_scale,z_scale_m.unsqueeze(0)),dim=0)

                # Concatenate Gaussian parameters from q(z_cur | z_next)
                if t < t_max-1:
                    z_next = z_est[:,t+1,:]

                    # Mask out batches where t+1 > seq_length
                    mask_n = (t+1 < seq_lengths).float()
                    mask_all = torch.cat((mask_all,mask_n.unsqueeze(0)),dim=0)

                    z_loc_n, z_scale_n = self.z_backward(z_next)
                    z_loc = torch.cat((z_loc,z_loc_n.unsqueeze(0)),dim=0)
                    z_scale = torch.cat((z_scale,z_scale_n.unsqueeze(0)),dim=0)

                # Product of experts to combine Gaussians
                z_loc, z_scale = self.experts(z_loc, z_scale, mask_all)

                # Sample z_cur
                z_dist = dist.Normal(z_loc, z_scale).\
                         mask(mask_z).independent(1)
                with poutine.scale(scale=annealing_beta):
                    z_cur = pyro.sample("latent_{}".format(t), z_dist)
