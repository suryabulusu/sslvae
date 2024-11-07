import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import device


class GaussianSample(nn.Module):
    """
    Last layer for Gaussian variational posterior
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_fn = nn.Linear(in_features, out_features)
        self.log_var_fn = nn.Linear(in_features, out_features)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # eps ~ N(0, 1) of shape mu
        eps = torch.randn_like(mu, device=device)
        std = (0.5 * log_var).exp()
        z = mu + std * eps

        return z

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input:
        - penultimate layer of encoded image rep

        returns latent representation z
        - reparametrized sample z = mu + sig*eps
        - mean
        - log-variance
        """
        mu = self.mu_fn(x)
        log_var = F.softplus(self.log_var_fn(x))

        return self.reparametrize(mu, log_var), mu, log_var


class GumbelSoftmaxSample(nn.Module):
    """
    last layer when q(z|x) is a concrete dist

    Straight-through estimator trick
    - forward pass: sample from gumbel sotfmax, take argmax (like categorical)
    - backward pass: pretend it is a concrete dist, take grads
    """

    def __init__(self, in_features: int, num_categories: int, tau: float):
        super().__init__()
        self.in_features = in_features
        self.num_categories = num_categories
        self.tau = tau  # temparature
        # low vals of tau => closer to argmax

        self.logits_fn = nn.Linear(in_features, num_categories)

    def forward(self, x: torch.Tensor):
        """
        returns
        - reparametrized sample
        - log logits
        """

        log_logits = self.logits_fn(x)
        # return loglogits for concrete dist kl computation

        return self.reparametrize(log_logits), log_logits

    def reparametrize(self, log_logits):
        """obtains differentiable sample"""
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(log_logits, device=device) + 1e-8) + 1e-8
        )
        sample = F.softmax((log_logits + gumbel_noise) / self.tau, dim=-1)

        return sample


# testing
# gumbel_sampler = GumbelSoftmaxSample(in_features=128, num_categories=10)
# x = torch.randn(32, 128)  # Batch of 32 samples
# sample, probs = gumbel_sampler(x)
# print(sample.shape, probs.shape, device)
