"""MNIST example for the multimodal VAE.
Adapted for Pyro from https://github.com/wenxuanliu/multimodal-vae/"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms

import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(1, parent_dir)

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
        self.fc1   = nn.Linear(10, 512)
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
    """MVAE for MNIST data.
    @param z_dim: integer
                  number of latent dimensions
    """

    def __init__(self, z_dim, z_prior_loc=0.0, z_prior_scale=1.0,
                 lambda_image=1.0, lambda_text=50.0, use_cuda=False):
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=64,
                        help='size of the latent embedding')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--supervision', type=float, default=1.0, metavar='F',
                        help='fraction of complete examples (default: 1.0)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--anneal_len', type=int, default=200, metavar='N',
                        help='number of epochs to anneal beta (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--l_image', type=float, default=1.0, metavar='W',
                        help='image regularization (default: 1.0)')
    parser.add_argument('--l_text', type=float, default=50.0, metavar='W',
                        help='text regularization (default: 50.0)')
    parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                        help='how many epochs to wait before saving')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Create loaders for MNIST (auto-downloads if necessary)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    # Construct multimodal VAE
    mvae = MnistMVAE(z_dim=args.z_dim, use_cuda=args.cuda,
                     lambda_image=args.l_image, lambda_text=args.l_text)

    # Setup optimizer and inference algorithm
    optimizer = Adam({'lr': args.lr})
    svi = SVI(mvae.model, mvae.guide, optimizer, loss=Trace_ELBO())

    # Training loop
    def train(epoch):
        # Anneal beta linearly from 0 to 1 over anneal_len
        annealing_beta = min(epoch / args.anneal_len, 1.0)
        # Accumulate loss for each modality and joint loss
        joint_loss, image_loss, text_loss = 0.0, 0.0, 0.0
        for batch_num, (image, text) in enumerate(train_loader):
            # Flatten images
            image = image.view(-1, 784)
            # Convert labels to one-hot
            text = text.view(-1, 1)
            text = torch.zeros(text.shape[0], 10).scatter_(1, text, 1)
            if args.cuda:
                image, text = image.cuda(), text.cuda()
            # Minimize ELBO term for complete example
            joint_loss += svi.step(inputs={'image': image, 'text': text},
                                   batch_size=train_loader.batch_size,
                                   annealing_beta=annealing_beta)
            # Minimize ELBO term for each modality
            image_loss += svi.step(inputs={'image': image},
                                   batch_size=train_loader.batch_size,
                                   annealing_beta=annealing_beta)
            text_loss += svi.step(inputs={'text': text},
                                  batch_size=train_loader.batch_size,
                                  annealing_beta=annealing_beta)
            print(batch_num, joint_loss, image_loss, text_loss)
        # Average losses and print
        joint_loss /= len(train_loader)
        image_loss /= len(train_loader)
        text_loss /= len(train_loader)
        print('Epoch: {}\tLoss: {:.4f}\tImage: {:.4f}\tText:{:.4f}'.\
              format(epoch, joint_loss, image_loss, text_loss))
        return joint_loss, image_loss, text_loss

    # Evaluation over test set
    def evaluate():
        # Accumulate loss for each modality and joint loss
        joint_loss, image_loss, text_loss = 0.0, 0.0, 0.0
        for batch_num, (image, text) in enumerate(test_loader):
            # Flatten images
            image = image.view(-1, 784)
            # Convert labels to one-hot
            text = text.view(-1, 1)
            text = torch.zeros(text.shape[0], 10).scatter_(1, text, 1)
            if args.cuda:
                image, text = image.cuda(), text.cuda()
            # Compute ELBO term for complete example
            joint_loss += svi.step(inputs={'image': image, 'text': text},
                                   batch_size=test_loader.batch_size)
            # Compute ELBO term for each modality
            image_loss += svi.step(inputs={'image': image},
                                   batch_size=test_loader.batch_size)
            text_loss += svi.step(inputs={'text': text},
                                  batch_size=test_loader.batch_size)
        # Average losses and print
        joint_loss /= len(train_loader)
        image_loss /= len(train_loader)
        text_loss /= len(train_loader)
        print('Test\tLoss: {:.4f}\tImage: {:.4f}\tText:{:.4f}'.\
              format(joint_loss, image_loss, text_loss))
        return joint_loss, image_loss, text_loss

    # Train and save best model
    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        joint_loss, image_loss, text_loss = evaluate()
        total_loss = joint_loss + image_loss + text_loss

        if total_loss < best_loss:
            best_loss = total_loss
            path = os.path.join("./mnist_models", "best.save") 
            pyro.get_param_store().save(path)

        if epoch % args.save_freq == 0:
            path = os.path.join("./mnist_models",
                                "epoch_{}.save".format(epoch)) 
            pyro.get_param_store().save(path)
            
