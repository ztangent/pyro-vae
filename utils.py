from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

class ProductOfExperts(nn.Module):
    """
    Return parameters for product of independent Gaussian experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param loc: M x B X D for M experts, B batches, and D latent dimensions
    @param scale: M x B X D for M experts, B batches, and D latent dimensions
    @param masks: M X B for M experts and B batches
    """
    def forward(self, loc, scale, masks=None, eps=1e-8):
        scale = scale + eps # numerical constant for stability
        # Precision matrix of i-th Gaussian expert (T = 1/sigma^2)
        T = 1. / scale
        if masks is not None:
            T = T * masks.unsqueeze(-1)
        product_loc = torch.sum(loc * T, dim=0) / torch.sum(T, dim=0)
        product_scale = 1. / torch.sum(T, dim=0)
        return product_loc, product_scale
