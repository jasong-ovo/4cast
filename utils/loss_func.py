import torch
import torch.nn as nn
import numpy as np



def generate_weights(h, device):
    # steph = 180.0 / h
    # latitude = np.arange(-90, 90, steph).astype(np.int)

    latitude = np.linspace(-90, 90, num=h)
    weights = np.cos(np.deg2rad(latitude))
    weights = weights / weights.mean()
    weights = torch.from_numpy(weights).reshape(1, 1, h, 1).to(device)
    return weights

class rmse_loss(nn.Module):
    def __init__(self, h=32, device='cuda'):
        self.h = h
        self.weights = generate_weights(self.h, device)
        super().__init__()
        
    def forward(self, x, y):
        loss = torch.pow((x-y), 2)
        loss = loss * self.weights
        return torch.mean(loss)

def get_loss_func(name, device):
    """
    source: B, C, H, W
    target: B, C, H, W
    """
    if name == 'mse':
        return nn.MSELoss().to(device)
    elif name == 'rmse':
        return rmse_loss(device=device)
    elif name == 'smse':
        pass
    else:
        return NotImplementedError