import torch
import torch.nn as nn


def get_device():
    """check for gpu availability"""
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps" # apple M1/M2
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


device = get_device()


def load_encoder_decoder(
    model_name: str
):
    """load saved models for analysis"""
    print(f"loading model: {model_name}")
