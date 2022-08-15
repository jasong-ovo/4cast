import numpy as np
import torch


def generate_weights(h, device):
    # steph = 180.0 / h
    # latitude = np.arange(-90, 90, steph).astype(np.int)

    latitude = np.linspace(-90, 90, num=h)
    weights = np.cos(np.deg2rad(latitude))
    weights = weights / weights.mean()
    weights = torch.from_numpy(weights).reshape(1, 1, h, 1).to(device)
    return weights

def compute_weithed_metric(metric, fake, real, std, device, mean_dims=[1,2,3], eps=1e-8):
    if metric == 'rmse':
        return compute_weighted_rmse(fake, real, std, device, mean_dims, eps=1e-8)
    elif metric == 'acc':
        return compute_weighted_acc(fake, real, std, device, mean_dims, eps=1e-8)
    elif metric == 'mae':
        return compute_weighted_mae(fake, real, std, device, mean_dims, eps=1e-8)
    else: return None

def compute_weighted_rmse(fake, real, std, device, mean_dims=[1,2,3], eps=1e-8):
    error = fake - real + eps
    bs, ch, h, w = fake.shape
    # latitude = np.linspace(-90, 90, num=h)
    weights = generate_weights(h, device)
    wrmse = torch.sqrt(((error)**2 * weights).mean(mean_dims)) * std
    return wrmse

def compute_weighted_acc(fake, real, std, device, mean_dims=[1,2,3], eps=1e-8):
    bs, ch, h, w = fake.shape
    real = real.float()
    fake = fake.float()
    weights = generate_weights(h, device)
    fake_prime = fake - fake.mean() + eps
    real_prime = real - real.mean() + eps


    cov = torch.sum(weights * fake_prime * real_prime, dim=mean_dims)
    cor = torch.sum(weights* fake_prime ** 2, dim=mean_dims,) * torch.sum(weights* real_prime ** 2, dim=mean_dims,)
    cor = torch.sqrt(cor.float())
    # acc = torch.sum(w * fake_prime * real_prime) / torch.sqrt(torch.sum(w* fake_prime ** 2))
    acc = cov / cor
    return acc


    return None

def compute_weighted_mae(fake, real,std, device, mean_dims=[1,2,3], eps=1e-8):
    bs, ch, h, w = fake.shape
    weights = generate_weights(h, device)
    error = fake - real + eps
    mae = (torch.abs(error) * weights).mean(mean_dims) * std
    return 

def compute_latitude_mse(fake, real, std, device, mean_dims=[1,2,3], eps=1e-8):
    error = fake - real + eps
    bs, ch, h, w = fake.shape
    # latitude = np.linspace(-90, 90, num=h)
    # weights = generate_weights(h, device)
    # wrmse = torch.sqrt(((error)**2 * weights).mean(mean_dims)) * std
    # import pdb;pdb.set_trace()
    total_mse = torch.sqrt(error**2).sum(dim=[1, 2, 3]).reshape(bs, -1)# B
    mse_latitude = torch.sqrt(((error)**2)).sum(dim=[1, 3]) #B, H
    relative_mse_latitude = mse_latitude / total_mse     #B ,H
    return mse_latitude, relative_mse_latitude
