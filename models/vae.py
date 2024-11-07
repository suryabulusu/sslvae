# repurposed from: https://github.com/wohlert/semi-supervised-pytorch/blob/master/semi-supervised/models/vae.py
# updates: consider joint dist q(z,y|x)

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from samplers import GaussianSample, GumbelSoftmaxSample
from torch.distributions import RelaxedOneHotCategorical

config = Config()


class FeedForward(nn.Module):
    """simple feedforward nn"""

    def __init__(
        self, dims: List[int], activation_fn=F.relu, output_activation_fn=F.relu
    ):
        super().__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == len(self.layers) - 1:
                x = self.output_activation_fn(x)
            else:
                x = self.activation_fn(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        h_dims: List[int],
        z_dim: int,
        num_categories: int,
        tau2: float,
    ):
        """q(z,y|x) variational posterior

        q(z,y|x) = q(y|x) * q(z|y,x)
        params
        - x_dim: dimension of input image
        - h_dims: dimensions of hidden layers
        - num_categories/y_dim: number of categories/classes
        - z_dim: dimension of variational posterior z
        - tau2: concrete posterior temp

        """
        super().__init__()

        # layers for q(y|x)
        self.hidden_layer_y = FeedForward([x_dim] + h_dims)
        self.sample_y = GumbelSoftmaxSample(
            in_features=h_dims[-1], num_categories=num_categories, tau=tau2
        )

        # layers for q(z|x,y); concat y to x
        self.hidden_layer_z = FeedForward([x_dim + num_categories] + h_dims)
        self.sample_z = GaussianSample(in_features=h_dims[-1], out_features=z_dim)

    def forward(self, x: torch.Tensor):
        # set tau in config
        h_y = self.hidden_layer_y(x)
        reparam_y, log_logits_y = self.sample_y(h_y)

        xy = torch.cat([x, reparam_y], dim=-1)
        h_z = self.hidden_layer_z(xy)
        reparam_z, mu, log_var = self.sample_z(h_z)

        # bad code; club them somehow
        # we need mu, log_var, etc. for kld computation
        return reparam_y, log_logits_y, reparam_z, mu, log_var

    def sample(self, x: torch.Tensor):
        # encode images and observe latent reps / classes
        reparam_y, log_logits_y, reparam_z, mu, log_var = self.forward(x)
        hard_y = reparam_y.argmax(dim=1).item()

        return reparam_z, hard_y


class Decoder(nn.Module):
    """p(x|z) data likelihood model / generator"""

    def __init__(self, z_dim: int, h_dims: List[int], x_dim: int):
        super().__init__()
        self.layers = FeedForward(
            [z_dim] + h_dims + [x_dim], output_activation_fn=F.sigmoid
        )
        # a richer model would be pixelcnn + mixture of logistics output

    def forward(self, z: torch.Tensor):
        """takes in latent rep and generates data x"""
        return self.layers(z)


class SSLVAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.x_dim = config.x_dim
        self.z_dim = config.z_dim
        self.h_dims = config.h_dims
        self.num_categories = config.num_categories
        self.tau1 = config.tau1  # temp hyperparam prior = 2/3
        self.tau2 = config.tau2  # temp hyperparam post = 1.0
        self.device = config.device

        self.encoder = Encoder(
            self.x_dim, self.h_dims, self.z_dim, self.num_categories, self.tau2
        )
        self.decoder = Decoder(self.z_dim, list(reversed(self.h_dims)), self.x_dim)

        self.kld = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        reparam_y, log_logits_y, reparam_z, mu, log_var = self.encoder(x)
        recon_x = self.decoder(reparam_z)

        # compute kl for y and z
        # KL[q(z|y,x) || p(z)] where p(z) is N(0, I)
        # https://statproofbook.github.io/P/norm-kl.html
        kl_z = torch.sum(
            -0.5 * log_var + 0.5 * mu.pow(2) + 0.5 * log_var.exp() - 0.5
        ).mean()

        # KL[q(y|x) || p(y)] where p(y) is unif and q(y|x) is concrete dist
        # for 4 classes, tau posterior = 1, tau prior = 2/3
        # compute for one sample!
        qy = RelaxedOneHotCategorical(
            temperature=torch.tensor([self.tau2]), logits=log_logits_y
        ).log_prob(reparam_y)
        # uniform concrete prior
        # log logits = [-logK, -logK, ... , -logK]
        prior_log_logits = -torch.log(
            torch.tensor(self.num_categories, device=self.device)
        )
        prior_log_logits = prior_log_logits.expand_as(log_logits_y)
        py = RelaxedOneHotCategorical(
            temperature=torch.tensor([self.tau1]), logits=prior_log_logits
        ).log_prob(reparam_y)
        kl_y = (qy - py).mean()

        self.kld = kl_z + kl_y
        return recon_x, kl_y + kl_z

    def sample(self, z: torch.Tensor):
        """given latent rep z, generate image"""
        return self.decoder(z)
        # how to inform here about class? :)


# # testing
# sslvae = SSLVAE(config=config)
# x = torch.randn(32, 784) # two images
# x_recon, kl = sslvae(x)
# print(x_recon.shape)
# print(kl) # assertions: check if kl > 0
# # note: kl_y can mess up for small values of batch size cuz we're sampling
