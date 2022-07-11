'''
# test the usage of timm

'''
import os
import torch
client_config_file = "~/petreloss.conf"

# class DistilledVisionTransformer(VisionTransformer):
from datasets.dataset import ERA5CephDataset
from datasets.sampler import RASampler
from utils.args import get_args_parser
from utils.dist import init_distributed_mode


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    args.distributed = True
    args.rank = 0
    args.world_size = 1
    args.gpu = 8

    # torch.cuda.set_device(range(args.gpu))
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

    dataset = ERA5CephDataset()
    sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=1, rank=0, shuffle=False
                )
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    for i, item in enumerate(data_loader):
        print(i, item.shape, item.device)
        if i > 10: break


