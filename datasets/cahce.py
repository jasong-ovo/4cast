# cache dataset 
import numpy as np
import os

from tqdm.notebook import tqdm

# data = np.load("/nvme/zhangtianning/datasets/era5G32x64/1979/1979-0000.npy")

# all_data = []
# for year in range(2018, 2022):
#     array_list=[]
#     file_path   = f"/mnt/lustre/share_data/ai4earth/era5G32x64_new/{year}"
#     file_length = len(os.listdir(file_path))
#     for i in range(file_length):
#         array_list.append(np.load(f"/mnt/lustre/share_data/ai4earth/era5G32x64_new/{year}/{year}-{i:04d}.npy"))
#     array_list = np.stack(array_list)
#     all_data.append(array_list)
# all_data = np.concatenate(all_data)
# np.save('/mnt/lustre/share_data/ai4earth/test_data.npy', all_data)

# all_data = []
# for year in range(2016, 2018):
#     array_list=[]
#     file_path   = f"/mnt/lustre/share_data/ai4earth/era5G32x64_new/{year}"
#     file_length = len(os.listdir(file_path))
#     for i in range(file_length):
#         array_list.append(np.load(f"/mnt/lustre/share_data/ai4earth/era5G32x64_new/{year}/{year}-{i:04d}.npy"))
#     array_list = np.stack(array_list)
#     all_data.append(array_list)
# all_data = np.concatenate(all_data)
# np.save('/mnt/lustre/share_data/ai4earth/valid_data.npy', all_data)

all_data = []
for year in range(1979, 2016):
    array_list=[]
    file_path   = f"/mnt/lustre/share_data/ai4earth/era5G32x64_new/{year}"
    file_length = len(os.listdir(file_path))
    for i in range(file_length):
        array_list.append(np.load(f"/mnt/lustre/share_data/ai4earth/era5G32x64_new/{year}/{year}-{i:04d}.npy"))
    array_list = np.stack(array_list)
    all_data.append(array_list)
all_data = np.concatenate(all_data)
np.save('/mnt/lustre/share_data/ai4earth/train_data.npy', all_data)
#np.save("/nvme/zhangtianning/datasets/era5G32x64/test_data",all_data)