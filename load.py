import numpy as np
import os
import pickle 
def read_npy_from_buffer(path):
    buf = bytearray(os.path.getsize(path))
    with open(path, 'rb') as f:
        f.readinto(buf)
    return np.frombuffer(buf, dtype=np.half).reshape(720,1440)

# norm_path = "/mnt/lustre/share_data/ai4earth/era5_mean_std_new/10m_u_component_of_wind.pkl"
# with open(norm_path, 'rb') as f:
#     vars = pickle.load(f)
# mean = vars['mean']
# std = vars['std']

# Ashape = (720, 1440)
# a = read_npy_from_buffer("/mnt/cache/gongjunchao/workdir/IfGAN/10m_u_component_of_wind-1988-0000.npy")

# b = np.load("grib_u10_wind-1988-0000.npy")
old_std_file = '/mnt/lustre/share_data/ai4earth/era5_mean_std'
new_std_file = '/mnt/lustre/share_data/ai4earth/era5_mean_std_new'

orig_validdata_file = '/mnt/petrelfs/gongjunchao/32x64/valid.npy'
orig_traindata_file = '/mnt/petrelfs/gongjunchao/32x64/train.npy'

old_norm_path = os.path.join(old_std_file, '10m_u_component_of_wind.pkl')
new_norm_path = os.path.join(new_std_file, '500h_geopotential.pkl')

with open(old_norm_path, 'rb') as f:
    old = pickle.load(f)
with open(new_norm_path, 'rb') as f:
    new = pickle.load(f)

train_data = np.load("/mnt/lustre/share_data/ai4earth/era5G32x64_set/train_data.npy") #train_data[1, 16]*new['std'] + new['mean']
# a = np.load("/mnt/lustre/share_data/ai4earth/era5G32x64/1982/1982-0010.npy") #a[0]*old['std']+old['mean']
orig_train_data = np.load(orig_traindata_file)
orig_val_data = np.load(orig_validdata_file)

orig_data = np.concatenate([orig_train_data, orig_val_data], axis=0)
# print((orig_train_data-new['mean'])/new['std'])
import pdb; pdb.set_trace()
# means = np.mean(orig_data, axis=(0, 2, 3)) # 1序，第12个属性850Z出现Nan np.argwhere(np.isnan(orig_data))
# means = np.mean(np.mean(orig_data, axis=(2, 3)), axis=0)
# vars = np.var(orig_data, axis=(0, 2, 3))
##srun -p ai4science --gres=gpu:0 -c 96 -x SH-IDC1-10-140-0-169 python load.py