

import os
import torch
from torch_utils import training_stats, custom_ops
import re
import json
import dnnlib
import tempfile
import click
from configs.param import get_args



@click.command()
# basic_kwargs.
@click.option('--outdir',    help='Where to save the results', metavar='DIR',   default='/mnt/lustre/chenzhuo1/IF_WKP',type=str)
@click.option('--resume',    help='Resume from given network pickle', metavar='[PATH|URL]', default=None,      type=str)
@click.option('--desc',      help='String to include in result dir name', metavar='STR',                       type=str)
@click.option('--seed',      help='Random seed', metavar='INT',                 default=1,                     type=click.IntRange(min=0), show_default=True)

@click.option('--gpus',      help='Number of GPUs to use', metavar='INT',       default=8,                     type=click.IntRange(min=1))
@click.option('--batch',     help='Total batch size', metavar='INT',            default=80,                    type=click.IntRange(min=1))
# training_set_kwargs
@click.option('--dataset',   help='Dataset class',                              default='ERA5GDataset',         type=str)
@click.option('--dir_data',  help='Path of training data',                      default='/mnt/lustre/chenzhuo1/era5G32x64',         type=str)
@click.option('--workers',   help='DataLoader worker processes', metavar='INT', default=32,                     type=click.IntRange(min=0),  show_default=True)
@click.option('--drop_last', help='Forget unfull batch', metavar='BOOL',        default=True,                  type=bool, show_default=True)
@click.option('--pin_memory',help='pin_memory', metavar='BOOL',                 default=True,                  type=bool, show_default=True)


def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)


if __name__ == "__main__":
    main()
