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
from models.module import AFNONet
from utils.io import load_std
from utils.dist import get_avaliable_memory
from utils.metric import compute_latitude_mse

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
    h, w = network_kwargs.img_size
    idxs = [vnames.index(name) for name in enames]
    stds = load_std(enames)
    for i, ename in enumerate(enames):
        if rank == vis_rank:
            print(f'std {ename} {stds[i]:.4f}')

    evaluating_set = init_dataset(dataset_kwargs)
    evaluating_set_sampler = RASampler(evaluating_set, rank=rank, num_replicas=num_gpus, shuffle=False, seed=random_seed)
    evaluating_set_dataloader = torch.utils.data.DataLoader(evaluating_set, batch_gpu,  sampler=evaluating_set_sampler, **data_loader_kwargs)
    dict_score = {}
    metrics = ['la_mse']
    for ename in enames:
        for metric in metrics:
            for seq in range(1, length+1):
                dict_score[f"{ename}-{metric}-{seq:02d}"] = []
                dict_score[f"rel_{ename}-{metric}-{seq:02d}"] = []

    if rank == vis_rank:
        print()
        print('Num images: ', len(evaluating_set))
        print('Image shape:', network_kwargs.img_size)
        print('sequential length:', length)
        print('Metrics:', metrics)
        print('Variables:', enames)
        print()
    model = AFNONet(**network_kwargs).to(device).eval()
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
        batch = batch.half().to(device, non_blocking=True).permute(1, 0, 2, 3, 4)
        xs = batch[0]

        for seq in range(1, length+1):
            xt = batch[seq].half().to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                xt_hat = model(xs)
            for i, idx in enumerate(idxs):
                ename = enames[i]
                std = stds[i]
                fake = xt_hat[:, idx:idx+1, :, :]
                real = xt[:, idx:idx+1, :, :]
                for r, metric in enumerate(metrics):
                    la_mse, rel_la_mse = compute_latitude_mse(fake, real, std, device, mean_dims=[1,2,3], eps=1e-8)
                    dict_score[f"{ename}-{metric}-{seq:02d}"].append(la_mse.clone())
                    dict_score[f"rel_{ename}-{metric}-{seq:02d}"].append(rel_la_mse.clone())
            xs = xt_hat
    rel_dict = {}
    for ename in enames:
        rel_dict[f'rel_{ename}'] = []

    for ename in enames:
        for metric in metrics:
            for seq in range(1, length + 1):
                la_score = dict_score[f"{ename}-{metric}-{seq:02d}"]
                la_score = torch.cat(la_score, dim=0) #B, H
                # if rank == vis_rank:
                #     print(ename, metric, seq, f"{score.mean().item()}:.4f")
                if num_gpus > 1:
                    torch.distributed.all_reduce(la_score)
                la_score /= num_gpus
                # score = score.reshape()
                la_score = torch.mean(la_score, dim=0)
                # dict_score[f"{ename}-{metric}-{seq:02d}"] = score
                rel_la_score = dict_score[f"rel_{ename}-{metric}-{seq:02d}"]
                rel_la_score = torch.cat(rel_la_score, dim=0) #B, H
                # if rank == vis_rank:
                #     print(ename, metric, seq, f"{score.mean().item()}:.4f")
                if num_gpus > 1:
                    torch.distributed.all_reduce(rel_la_score)
                rel_la_score /= num_gpus
                # score = score.reshape()
                rel_la_score = torch.mean(rel_la_score, dim=0)
                if rank == vis_rank:
                    print(ename, seq)
                    print(la_score)
                    print(rel_la_score)
                    rel_dict[f'rel_{ename}'].append(rel_la_score.unsqueeze(dim=0))
            # import pdb; pdb.set_trace()
            rels = torch.cat(rel_dict[f'rel_{ename}'], dim=0)
            rels = rels.cpu().numpy()
            np.save(f'rel_{ename}.npy', rels)

        # if rank == vis_rank:
        #     # xs, xt = [x.half().to(device, non_blocking=True) for x in batch]
        #     print('step', step, xs.shape)


    return None