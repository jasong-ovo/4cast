import argparse

#uniform_drop=False, drop_rate=0., drop_path_rate=0., norm_layer=None, dropcls=0)
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--epoch_t', default=1, type=int)
    parser.add_argument('--epoch_f', default=1, type=int)
    parser.add_argument('--cooldown_epochs', default=1, type=int)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--vis_loss_step', default=1, type=int)
    parser.add_argument('--vis_rank', default=1, type=int)
    # parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--output_dir', default='/mnt/lustre/chenzhuo1/C4_WKP/afno1', help='path where to save, empty for no saving')

    # training parameters
    parser.add_argument('--in_h', type=int, default=32, metavar='N', help='height of input')
    parser.add_argument('--in_w', type=int, default=64, metavar='N', help='weight of input')
    parser.add_argument('--in_ch', type=int, default=20, metavar='N', help='channel of input')
    parser.add_argument('--out_ch', type=int, default=20, metavar='N', help='channel of output')

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str, help='Name of model to train')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',  help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

    # fno parameters
    parser.add_argument('--fno-bias', action='store_true')
    parser.add_argument('--fno-blocks', type=int, default=4)
    parser.add_argument('--fno-softshrink', type=float, default=0.00)
    parser.add_argument('--double-skip', action='store_true')
    parser.add_argument('--tensorboard-dir', type=str, default=None)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--checkpoint-activations', action='store_true')
    parser.add_argument('--autoresume', action='store_true')

    # attention parameters
    parser.add_argument('--num-attention-heads', type=int, default=1)

    # long short parameters
    parser.add_argument('--ls-w', type=int, default=4)
    parser.add_argument('--ls-dp-rank', type=int, default=16)

    return parser


def get_args():
    parser = argparse.ArgumentParser('GFNet training and evaluation script', parents=[get_args_parser()])
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = parser.parse_args()
    return _GLOBAL_ARGS