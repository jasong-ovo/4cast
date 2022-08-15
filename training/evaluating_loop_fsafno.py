import os
import time
import copy
import pickle
import pynvml
import psutil
import torch
import torch.distributed as dist
import timm
import dnnlib
import numpy as np
from torch_utils import misc
from torch_utils import training_stats
from datasets.dataset import init_dataset
from torch.utils.data.distributed import DistributedSampler
from datasets.sampler import InfiniteSampler, RASampler
from models.module import AFNONet_FS
from utils.io import load_std
from utils.dist import get_avaliable_memory
from utils.metric import compute_weithed_metric

from configs.era5 import vnames
import warnings
import timm.optim
from timm.scheduler import create_scheduler
from torch.nn.parallel import DistributedDataParallel
warnings.filterwarnings("ignore", message="Argument interpolation should be")
warnings.filterwarnings("ignore", message="leaker semaphore objects")
# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
@torch.no_grad()
def evaluating_loop(
        rank                    = 0,        # Rank of the current process in [0, num_gpus[.
        run_dir                 = '.',      # Output directory.
        enames = [],
        metrics = [],
        num_gpus                = 8,        # Number of GPUs participating in the training.
        length                  = 21,       # prediction length
        resume                  = None,     # Resume from given network pickle
        batch_size              = 64,       # Total batch size for one training iteration
        batch_gpu               = 8,        # Total batch size for each gpu
        random_seed             = 0,        # Global random seed.
        network_kwargs          = {},       # Options for network.
        dataset_kwargs          = {},       # Options for dataset set.
        data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
        evaluation_kwargs       = {},       # Options for evaluation.
        visualize_kwargs        = {},       # Options for visualization.
):
    vis_rank = visualize_kwargs.vis_rank
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    in_chans = network_kwargs.in_chans
    t, h, w = network_kwargs.clip_size
    idxs = [vnames.index(name) for name in enames]
    stds = load_std(enames)
    for i, ename in enumerate(enames):
        if rank == vis_rank:
            print(f'std {ename} {stds[i]:.4f}')

    # import pdb; pdb.set_trace()
    evaluating_set = init_dataset(dataset_kwargs)
    evaluating_set_sampler = RASampler(evaluating_set, rank=rank, num_replicas=num_gpus, shuffle=False, seed=random_seed)
    evaluating_set_dataloader = torch.utils.data.DataLoader(evaluating_set, batch_gpu,  sampler=evaluating_set_sampler, **data_loader_kwargs)
    dict_score = {}
    for ename in enames:
        for metric in metrics:
            for seq in range(1, length+1):
                dict_score[f"{ename}-{metric}-{seq:02d}"] = []

    if rank == vis_rank:
        print()
        print('Num images: ', len(evaluating_set))
        print('Clip shape:', network_kwargs.clip_size)
        print('sequential length:', length)
        print('Metrics:', metrics)
        print('Variables:', enames)
        print()
    model = AFNONet_FS(**network_kwargs).to(device).eval()
    if rank == vis_rank:
        print(f'Resuming from "{resume}"')
        resume_data = torch.load(resume, map_location=device)
        model.load_state_dict(resume_data['model'])
    if rank == vis_rank:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [model]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=vis_rank)
    for name, module in [('model', model)]:
        if module is not None:
            if num_gpus > 1:
                misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
    if rank == vis_rank:
        print(f'Model consistency checking passed!')
    # print('index', idxs)

    for step, batch in enumerate(evaluating_set_dataloader):
        # import pdb;pdb.set_trace()
        batch = batch.to(device, non_blocking=True).half()
        xs, xts = batch[:, :6, :, :, :], batch[:, 6:, :, :, :]

        for seq in range(1, length+1):
            xt = xts[:, seq-1, :, :, :].half().to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                xt_hat = model(xs)
            for i, idx in enumerate(idxs):
                ename = enames[i]
                std = stds[i]
                fake = xt_hat[:, idx:idx+1, :, :]
                real = xt[:, idx:idx+1, :, :]
                for r, metric in enumerate(metrics):
                    score = compute_weithed_metric(metric, fake, real, std, device, )
                    dict_score[f"{ename}-{metric}-{seq:02d}"].append(score.clone())
                    # if rank == vis_rank:
                    #     print(step, seq, ename, metric, score.shape, score.device)
            xs[:, :5, :, :, :] = xs[:, 1:6, :, :, :]
            xs[:, 5, :, :, :] = xt_hat
    for ename in enames:
        for metric in metrics:
            for seq in range(1, length + 1):
                score = dict_score[f"{ename}-{metric}-{seq:02d}"]
                score = torch.cat(score, dim=0)
                # if rank == vis_rank:
                #     print(ename, metric, seq, f"{score.mean().item()}:.4f")
                if num_gpus > 1:
                    torch.distributed.all_reduce(score)
                score /= num_gpus
                # score = score.reshape()
                score = torch.mean(score)
                # dict_score[f"{ename}-{metric}-{seq:02d}"] = score
                if rank == vis_rank:
                    print(ename, metric, seq, f"{score.item():.4f}")

        # if rank == vis_rank:
        #     # xs, xt = [x.half().to(device, non_blocking=True) for x in batch]
        #     print('step', step, xs.shape)


    return None