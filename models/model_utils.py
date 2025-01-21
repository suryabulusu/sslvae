import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    """check for gpu availability"""
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"  # apple M1/M2
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


device = get_device()


def load_encoder_decoder(model_name: str):
    """load saved models for analysis"""
    print(f"loading model: {model_name}")


# clever straight-through estimator implementation
def straight_through(theta, tau=1.0):
    # sample D ~ softmax(theta) forward pass
    pi_0 = F.softmax(theta, dim=-1)
    D = torch.multinomial(pi_0, num_samples=1)
    D = F.one_hot(D.squeeze(), num_classes=theta.size(-1)).float()

    # setup gumbel prob for backward
    pi_1 = F.softmax(theta / tau, dim=-1)

    # setup a single line fn handling both fwd n bwd
    D = pi_1 - pi_1.detach() + D.detach()
    # fwd => D = pi_1 - pi_1 + D = D
    # bwd => D = pi_1; so dD/dx = dpi_1/dx

    return D


def reinmax_st(theta, tau=2.0):
    # sample D ~ softmax(theta) fwd pass
    pi_0 = F.softmax(theta, dim=-1)
    D = torch.multinomial(pi_0, num_samples=1)
    D = F.one_hot(D.squeeze(), num_classes=theta.size(-1)).float()

    # bwd pass set up two new probs pi_1 and pi_2
    pi_1 = (D + F.softmax(theta / tau, dim=-1)) / 2
    pi_1 = F.softmax(((torch.log(pi_1) - theta).detach() + theta), dim=-1)
    # fwd => pi_1 = F.softmax(F.log(pi_1)) = pi_1 stable

    pi_2 = 2 * pi_1 - pi_0 / 2
    D = pi_2 - pi_2.detach() + D.detach()
    # fwd => D = pi_2 - pi_2 + D = D
    # bwd = D = pi_2
    # dpi_2 / dtheta = 2 * dpi_1 / dtheta - 1/2 * pi_0 / dtheta
    # dpi_1 / dtheta = dpi_1 / dinternal * dinternal / dtheta 
    # internal = theta due to detach.. -> dinternal / dtheta = I
    # dpi_1 / dinternal = it is typical softmax jacobian

    return D