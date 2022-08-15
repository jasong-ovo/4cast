import os
import sys
import io
import numpy as np
import xarray as xr
import torch
import pickle
from torchvision import datasets, transforms

from configs.ceph import client_config_file, vnames_short, Shape
from configs.era5 import Years
from utils.augs import identity_aug, longtitude_aug, frame_aug, latitude_flip_aug

import petrel_client
from petrel_client.client import Client


# def read_npy_from_ceph(client, url, Ashape=Shape):
#     array_ceph = client.get(url)
#     array_ceph = np.frombuffer(array_ceph, dtype=np.half).reshape(Ashape)
#     return array_ceph

def init_dataset(dataset_opts):
    dataset_name = dataset_opts.class_name
    if dataset_name == 'ERA5CephDataset':
        raise NotImplementedError
        # dataset = ERA5CephDataset(**dataset_opts)
    elif dataset_name == 'ERA5GDataset':
        dataset = ERA5GDataset(**dataset_opts)
    elif dataset_name == 'ERA5Dataset':
        pass
        #dataset = ERA5Dataset(**dataset_opts)
    return dataset

# class  ERA5CephDataset(datasets.ImageFolder):
#     def __init__(self, class_name='ERA5Dataset', root='s3://era5npy', ispretrain=True, crop_coord=None, mode='train', length=1):
#         self.client = Client(conf_path="~/.petreloss.conf")
#         self.root = root
#         self.years = Years['train']
#         self.crop_coord = crop_coord
#         self.file_list = self.init_file_list()
#         self.mode = mode
#         self.length = length

#         # self.transform = transforms.Compose(
#         #     [
#         #         transforms.to
#         #     ]
#         # )

#     def __len__(self):
#         return len(self.file_list)

#     def init_file_list(self):

#         file_list = []
#         for year in self.years:

#             if year == 1988:
#                 max_item = 1373
#             elif year % 4 == 0:
#                 max_item = 1464
#             else:
#                 max_item = 1460
#             for hour in range(max_item):
#                 file_list.append([year, hour])
#         return file_list

#     def _load_array(self, item):

#         year, hour = self.file_list[item]
#         arrays = []
#         # print(year, hour)
#         # import pdb; pdb.set_trace()
#         for name in vnames:
#             url = f"{self.root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
#             array = read_npy_from_ceph(self.client, url)
#             arrays.append(array[np.newaxis, :, :])
#         arrays = np.concatenate(arrays, axis=0)
#         # print(arrays.shape)
#         if not self.crop_coord is None:
#             l, r, u, b = self.crop_coord
#             arrays = arrays[:, u:b, l:r]
#         arrays = torch.from_numpy(arrays)

#         return arrays

#     def __getitem__(self, item):
#         item = min(item, len(self.file_list)-self.length)
#         array_seq = []
#         for i in range(self.length+1):
#             array_seq.append(self._load_array(item+i))
#         return array_seq



# smalldataset_path={
#     'train': "/mnt/lustre/share_data/ai4earth/era5G32x64_set/train_data.npy",
#     'valid': "/mnt/lustre/share_data/ai4earth/era5G32x64_set/valid_data.npy",
#     'test':  "/mnt/lustre/share_data/ai4earth/era5G32x64_set/test_data.npy"
# }
smalldataset_path={
    'train': '/mnt/petrelfs/gongjunchao/32x64/train_norm.npy',
    'valid': '/mnt/petrelfs/gongjunchao/32x64/valid_norm.npy',
    'test':  '/mnt/petrelfs/gongjunchao/32x64/test_norm.npy'
}
vnames = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',

]
def load_small_dataset_in_memory(split):
    return torch.Tensor(np.load(smalldataset_path[split]))

class ERA5GDataset(datasets.ImageFolder):
    def __init__(self, class_name='ERA5GDataset', root='/mnt/lustre/chenzhuo1/era5G32x64',  mode='train', length=1, 
                crop_coord=None, in_length=1, std_trans=False,  use_longtitude_aug=False, use_latitude_aug=False,
                insert_frame_aug=False):
        """
        args:
            length: pred_length (int).
            in_length: length of input frames (int).
        """
        self.mode = mode
        self.vnames = vnames
        self.crop_coord = crop_coord
        self.length = length
        self.datas = load_small_dataset_in_memory(mode) # (N, C, H, W)
        self.in_length = in_length
        self.insert_frame_aug = insert_frame_aug
        

        if std_trans:
            # import pdb; pdb.set_trace()
            trans_weights = np.load('/mnt/cache/gongjunchao/workdir/IfGAN/datasets/new2old.npy')
            self.datas = self.datas * trans_weights.reshape(1, -1, 1, 1)
            print("###################################### warning ######################################")
            print("###################################### std_transfered ######################################")
            print("###################################### warning ######################################")


        self.transform = [identity_aug()]
        if use_longtitude_aug:
            self.transform.append(longtitude_aug())
        if use_latitude_aug:
            self.transform.append(latitude_flip_aug())

        self.transform = transforms.Compose(
           self.transform
        )

        self.frame_aug = frame_aug()

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        if  self.insert_frame_aug:
            total_length = self.length + self.in_length + 1
            item = min(item, self.datas.shape[0] - total_length)
            tensor_seq = self.datas[item:item+total_length , :, :, :]
            tensor_seq = self.frame_aug(tensor_seq)
        else:
            total_length = self.length + self.in_length
            item = min(item, self.datas.shape[0] - total_length)
            tensor_seq = self.datas[item:item+total_length , :, :, :]
        
        return self.transform(tensor_seq)
        if self.use_transform:
            return self.transform(tensor_seq)
        else:
            return tensor_seq
        # return self.transform(tensor_seq)
        # array_seq = []
        # for i in range(self.length+1):
        #     array_seq.append()
        # return array_seq

if __name__ == "__main__":
    # srun -p ai4science --gres=gpu:0 -c 32 -x SH-IDC1-10-140-0-169 python dataset.py
    # datas = load_small_dataset_in_memory('test') #N_test, C, H, W
    data0 = np.load('/mnt/lustre/share_data/ai4earth/era5G32x64_new/2018/2018-0000.npy') #C, H, W
    data1 = np.load('2018-0000.npy')
    print(data1[19, 1, 4], data0[19, 1, 4]) # test set的第0帧
    # YearsAll = range(1979, 2022)
    # Years4Train = range(1979, 2016)
    # Years4Valid = range(2016, 2018)
    # Years4Test = range(2018, 2022)
    import pdb; pdb.set_trace()
    

