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