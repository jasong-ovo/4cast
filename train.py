import os
import torch
from torch_utils import training_stats, custom_ops
import re
import json
import dnnlib
import tempfile
import click
from training.training_loop import training_loop
from configs.param import get_args

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop(rank=rank, **c)
#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    # print(f'Training duration:   {c.total_kimg} kimg')
    # print(f'Dataset path:        {c.training_set_kwargs.path}')
    # print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    # print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    # print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    # print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=True)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()
# basic_kwargs.
@click.option('--outdir',    help='Where to save the results', metavar='DIR',   default='/mnt/lustre/chenzhuo1/IF_WKP',type=str)
@click.option('--cfg',       help='Base configuration',                         default='era5R32x64',                type=str)

# training_kwargs.
@click.option('--gpus',      help='Number of GPUs to use', metavar='INT',       default=8,                     type=click.IntRange(min=1))
@click.option('--batch',     help='Total batch size', metavar='INT',            default=256,                    type=click.IntRange(min=1))
@click.option('--epoch_t',   help='Epoch of pretraining', metavar='INT',        default=80,                    type=click.IntRange(min=0))
@click.option('--epoch_f',   help='Epoch of finetunng', metavar='INT',          default=50,                    type=click.IntRange(min=0))
@click.option('--resume',    help='Resume from given network pickle', metavar='[PATH|URL]', default=None,      type=str)
@click.option('--desc',      help='String to include in result dir name', metavar='STR',                       type=str)
@click.option('--dry_run',   help='do nothing', metavar='BOOL',                 default=False,                 type=click.BOOL,  show_default=True)
@click.option('--seed',      help='Random seed', metavar='INT',                 default=1,                     type=click.IntRange(min=0), show_default=True)
@click.option('--lr',        help='learning rate ',  metavar='FLOAT',           default=5e-4,                  type=click.FloatRange(min=0))
@click.option('--weight_decay',help='weight_decay',  metavar='FLOAT',           default=0.05,                  type=click.FloatRange(min=0))

# scheduler_kwargs.
@click.option('--sched',     help='scheduler type', metavar='SCHEDULER',        default='cosine',              type=str)
@click.option('--lr_noise',  help='learning rate ',  metavar='FLOAT',           default=None,                  type=click.FloatRange(min=0))
@click.option('--min_lr',    help='learning rate low bound',  metavar='FLOAT',  default=1e-5,                  type=click.FloatRange(min=0))
@click.option('--warmup_lr', help='learning rate ',  metavar='FLOAT',           default=5e-4,                  type=click.FloatRange(min=0))
@click.option('--warmup_epochs', help='Random seed', metavar='INT',             default=10,                     type=click.IntRange(min=0), show_default=True)
@click.option('--cooldown_epochs', help='Random seed', metavar='INT',           default=10,                     type=click.IntRange(min=0), show_default=True)


# training_set_kwargs
@click.option('--dataset',   help='Dataset class',                              default='ERA5GDataset',         type=str)
@click.option('--dir_data',  help='Path of training data',                      default='/mnt/lustre/chenzhuo1/era5G32x64',         type=str)
@click.option('--workers',   help='DataLoader worker processes', metavar='INT', default=24,                     type=click.IntRange(min=0),  show_default=True)
@click.option('--drop_last', help='Forget unfull batch', metavar='BOOL',        default=True,                  type=bool, show_default=True)
@click.option('--pin_memory',help='pin_memory', metavar='BOOL',                 default=False,                  type=bool, show_default=True)

# network_kwargs
@click.option('--resh',      help='Base configuration',                         default=32,                     type=int)
@click.option('--resw',      help='Base configuration',                         default=64,                     type=int)
@click.option('--patch_size',help='Base configuration',                         default=8,                      type=int)
@click.option('--in_chans',  help='Base configuration',                         default=20,                     type=int)
@click.option('--out_chans', help='Base configuration',                         default=20,                     type=int)
@click.option('--embed_dim', help='Base configuration',                         default=768,                    type=int)
@click.option('--depth',     help='Base configuration',                         default=12,                     type=int)
@click.option('--mlp_ratio', help='Base configuration',                         default=4,                      type=int)
# visualize_kwargs
@click.option('--vis_loss_step',help='print loss frequency', metavar='INT',     default=40,                    type=click.IntRange(min=0),  show_default=True)
@click.option('--vis_rank', help='visualization rank', metavar='INT',           default=0,                     type=click.IntRange(min=0),  show_default=True)
@click.option('--dump_epoch', help='visualization rank', metavar='INT',         default=20,                     type=click.IntRange(min=0),  show_default=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch // opts.gpus
    c.epoch_t = opts.epoch_t
    c.epoch_f = opts.epoch_f
    c.resume = opts.resume
    c.lr = opts.lr
    c.training_set_kwargs = dnnlib.EasyDict(class_name=opts.dataset, root=opts.dir_data, mode='train', length=1, crop_coord=None)
    c.random_seed = opts.seed
    c.data_loader_kwargs = dnnlib.EasyDict(prefetch_factor=2)
    c.data_loader_kwargs.num_workers = opts.workers
    c.data_loader_kwargs.drop_last = opts.drop_last
    c.data_loader_kwargs.pin_memory = opts.pin_memory

    c.network_kwargs = dnnlib.EasyDict()
    c.network_kwargs.img_size = [opts.resh, opts.resw]
    c.network_kwargs.patch_size = opts.patch_size
    c.network_kwargs.in_chans = opts.in_chans
    c.network_kwargs.out_chans = opts.out_chans
    c.network_kwargs.embed_dim = opts.embed_dim
    c.network_kwargs.depth = opts.depth
    c.network_kwargs.mlp_ratio = opts.mlp_ratio

    c.scheduler_kwargs = dnnlib.EasyDict()
    c.scheduler_kwargs.sched = opts.sched
    c.scheduler_kwargs.epochs = max(opts.epoch_t - opts.warmup_epochs, 1)
    c.scheduler_kwargs.lr_noise = opts.lr_noise
    c.scheduler_kwargs.min_lr = opts.min_lr
    c.scheduler_kwargs.warmup_lr = opts.warmup_lr
    c.scheduler_kwargs.warmup_epochs = opts.warmup_epochs
    c.scheduler_kwargs.cooldown_epochs = opts.cooldown_epochs



    c.visualize_kwargs = dnnlib.EasyDict()
    c.visualize_kwargs.vis_loss_step = opts.vis_loss_step
    c.visualize_kwargs.vis_rank = opts.vis_rank
    c.visualize_kwargs.dump_epoch = opts.dump_epoch


    desc = f'{opts.cfg:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)
    c.network_kwargs = get_args()

if __name__ == "__main__":
    main()