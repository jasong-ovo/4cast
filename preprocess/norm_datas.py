import numpy as np

means = np.load("/mnt/cache/gongjunchao/workdir/IfGAN/means.npy")
stds = np.load("/mnt/cache/gongjunchao/workdir/IfGAN/stds.npy")

train_data = np.load('/mnt/petrelfs/gongjunchao/32x64/train_norm.npy')
valid_data = np.load('/mnt/petrelfs/gongjunchao/32x64/valid_norm.npy')
test_data = np.load('/mnt/petrelfs/gongjunchao/32x64/test_norm.npy')

import pdb; pdb.set_trace() #np.argwhere(np.isnan(orig_data))
# srun -p ai4science --gres=gpu:0 -c 96 -x SH-IDC1-10-140-0-169 python norm_datas.py