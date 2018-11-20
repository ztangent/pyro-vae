"""MNIST example for the multimodal VAE.
Adapted for Pyro from https://github.com/wenxuanliu/multimodal-vae/"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import random as r
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(1, parent_dir)

from mvae import MVAE, MVAE_ELBO

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
        return self.fc31(h), torch.exp(self.fc32(h))

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
        return F.logsigmoid(self.fc4(h))

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
        return self.fc31(h), torch.exp(self.fc32(h))


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
        return F.log_softmax(self.fc4(h), dim=1)

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
                          lambda_image, use_logits=True)
        self.add_modality('text', 10, dist.OneHotCategorical,
                          TextEncoder(z_dim), TextDecoder(z_dim),
                          lambda_text, use_logits=True)
        if use_cuda:
            self.use_cuda = True
            self.cuda()        

def image_transform(x):
    """Flatten and normalize image to [0, 1]."""
    x = transforms.functional.to_tensor(x).view(784)
    x = (x > 0.5).float()
    return x

def text_transform(y):
    """Convert labels to one-hot vector."""
    y = torch.zeros(10).scatter_(0, y, 1.0)
    return y

def load_sample(loader, target=None, n_samples=1):
    # Collate data from loader
    image, text = torch.zeros([0, 784]), torch.zeros([0, 10])
    i_samples = 0
    for batch_num, (i, t) in enumerate(loader):
        # Convert texts from one-hot back to integers
        labels = t.nonzero()[:,1]
        # Filter samples with the target label
        if target is not None:
            i = i[labels == target]
            t = t[labels == target]
        # Collate samples
        image = torch.cat((image, i[0:n_samples-i_samples,:]), dim=0)
        text = torch.cat((text, t[0:n_samples-i_samples,:]), dim=0)
        i_samples = image.shape[0]
        if i_samples == n_samples:
            break
    return image, text

def gen_sample(mvae, loader, n_samples, args):
    if args.target and (args.target < 0 or args.target > 9):
        print("Target label must be from 0--9.")
        return
    inputs = {}
    if args.condition:
        # Condition upon observed images or targets
        image, text = load_sample(loader, args.target, n_samples)
        if args.cuda:
            image, text = image.cuda(), text.cuda()
        if args.condition in ['image', 'both']:
            save_image(image.view(n_samples, 1, 28, 28), "original.png")
            inputs['image'] = image
        if args.condition in ['text', 'both']:
            inputs['text'] = text
    # Generate image and text
    outputs, params = mvae.forward(inputs, batch_size=n_samples)
    # Save generated image
    image = torch.exp(params['image'])
    save_image(image.view(n_samples, 1, 28, 28), "generated.png")
    # Save generated text labels
    text = params['text'].argmax(dim=1)
    with open('./labels.txt', 'w') as fp:
        for t in text:
            fp.write('{}\n'.format(int(t)))

def train(epoch, svi, loader, args):
    """Training loop."""
    # Anneal beta linearly from 0 to 1 over anneal_len
    annealing_beta = min(epoch / args.anneal_len, 1.0)
    # Accumulate loss and number of examples
    loss, data_num = 0.0, 0
    for batch_num, (image, text) in enumerate(loader):
        batch_size = image.shape[0]
        if args.cuda:
            image, text = image.cuda(), text.cuda()
        # Minimize ELBO terms
        loss += svi.step(inputs={'image': image, 'text': text},
                         batch_size=batch_size,
                         annealing_beta=annealing_beta)
        data_num += batch_size
        if batch_num % 50 != 0:
            continue
        # Print average loss at regular intervals
        print('Batch: {:5d}\tLoss: {:10.1f}'.\
              format(batch_num, loss/data_num))
    # Average losses and print
    loss /= data_num
    print('---')
    print('Epoch: {}\tLoss: {:10.1f}'.format(epoch, loss))
    return loss

# Evaluation over test set
def evaluate(mvae, svi, loader, args):
    # Accumulate loss and number of examples
    loss, data_num, correct = 0.0, 0, 0
    for batch_num, (image, text) in enumerate(loader):
        batch_size = image.shape[0]
        if args.cuda:
            image, text = image.cuda(), text.cuda()
        # Evaluate loss
        loss += svi.evaluate_loss(inputs={'image': image, 'text': text},
                                  batch_size=batch_size)
        # Compute labelling accuracy
        outputs, params = mvae.forward(inputs={'image': image},
                                       batch_size=batch_size)
        correct += (params['text'].argmax(dim=1) == text.nonzero()[:,1]).sum()
        data_num += batch_size
    # Compute and print average loss and label accuracy
    loss /= data_num
    accuracy = float(correct) / data_num
    print('Evaluation\tLoss: {:10.1f}\tAccuracy: {}'.format(loss, accuracy))
    return loss, accuracy

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
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training')
    parser.add_argument('--sample', type=int, default=0, metavar='N',
                        help='sample from trained model (default: no samples)')
    parser.add_argument('--condition', type=str, default=None,
                        help='sample conditoned on [image/text/both/none]')
    parser.add_argument('--target', type=int, default=None,
                        help="target label to condition on (default: random)")
    parser.add_argument('--model', type=str,
                        default="./mnist_models/best.save",
                        help='path to trained model')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Create loaders for MNIST (auto-downloads if necessary)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=True, download=True,
                       transform=image_transform,
                       target_transform=text_transform),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=False, download=True,
                       transform=image_transform,
                       target_transform=text_transform),
        batch_size=args.batch_size, shuffle=True)
    # Create path to save models
    if not os.path.exists('./mnist_models'):
        os.makedirs('./mnist_models')
    
    # Construct multimodal VAE
    pyro.clear_param_store()
    mvae = MnistMVAE(z_dim=args.z_dim, use_cuda=args.cuda,
                     lambda_image=args.l_image, lambda_text=args.l_text)
    
    # Setup optimizer and inference algorithm
    optimizer = Adam({'lr': args.lr})
    svi = SVI(mvae.model, mvae.guide, optimizer, loss=MVAE_ELBO())

    # Load trained model to test or sample
    if args.test or args.sample:
        pyro.get_param_store().load(args.model)
        pyro.module(mvae.name, mvae, update_module_params=True)
    
    # Sample from model if flag is set
    if args.sample > 0:
        gen_sample(mvae, train_loader, args.sample, args)
        sys.exit(0)
    
    # Evaluate model if test flag is set
    if args.test or args.sample:
        evaluate(mvae, svi, test_loader, args)
        sys.exit(0)
    
    # Otherwise train and save best model
    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(epoch, svi, train_loader, args)
        loss, accuracy = evaluate(mvae, svi, test_loader, args)

        if loss < best_loss:
            best_loss = loss
            path = os.path.join("./mnist_models", "best.save") 
            pyro.get_param_store().save(path)

        if epoch % args.save_freq == 0:
            path = os.path.join("./mnist_models",
                                "epoch_{}.save".format(epoch)) 
            pyro.get_param_store().save(path)
            
