import numpy as np
import torch


def compute_weighted_rmse(fake, real):
    error = fake - real
    bs, ch, h, w = fake.shape
    latitude = np.linspace(-90, 90, num=h)
    weights = np.cos(np.deg2rad(latitude))
    weights = weights / weights.mean()
    wrmse = np.sqrt(((error)**2 * weights).mean())
    return wrmse

def compute_weighted_acc(fake, real):
    return None
