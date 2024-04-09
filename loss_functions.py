import torch

def wasserstein_loss(y_real, y_fake):
    return torch.mean(y_real) - torch.mean(y_fake)
