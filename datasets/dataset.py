import os
import sys
import io
import numpy as np
import xarray as xr
import torch
import pickle
from torchvision import datasets, transforms

from configs.ceph import client_config_file, vnames, vnames_short, Shape
from configs.era5 import Years
import petrel_client
from petrel_client.client import Client


def read_npy_from_ceph(client, url, Ashape=Shape):
    array_ceph = client.get(url)
    array_ceph = np.frombuffer(array_ceph, dtype=np.half).reshape(Ashape)
    return array_ceph

def init_dataset(dataset_opts):
    dataset_name = dataset_opts.class_name
    if dataset_name == 'ERA5CephDataset':
        dataset = ERA5CephDataset(**dataset_opts)
    elif dataset_name == 'ERA5GDataset':
        dataset = ERA5GDataset(**dataset_opts)
    elif dataset_name == 'ERA5Dataset':
        dataset = ERA5Dataset(**dataset_opts)
    return dataset

class  ERA5CephDataset(datasets.ImageFolder):
    def __init__(self, class_name='ERA5Dataset', root='s3://era5npy', ispretrain=True, crop_coord=None):
        self.client = Client(conf_path="~/petreloss.conf")
        self.root = root
        self.years = Years['train']
        self.crop_coord = crop_coord
        self.file_list = self.init_file_list()

        # self.transform = transforms.Compose(
        #     [
        #         transforms.to
        #     ]
        # )

    def __len__(self):
        return len(self.file_list)

    def init_file_list(self):

        file_list = []
        for year in self.years:

            if year == 1988:
                max_item = 1373
            elif year % 4 == 0:
                max_item = 1464
            else:
                max_item = 1460
            for hour in range(max_item):
                file_list.append([year, hour])
        return file_list

    def __getitem__(self, item):

        year, hour = self.file_list[item]
        arrays = []
        for name in vnames:
            url = f"{self.root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            array = read_npy_from_ceph(self.client, url)
            array = array[np.newaxis, :, :]
            arrays.append(array)
        arrays = np.concatenate(arrays, axis=0)
        if not self.crop_coord is None:
            l, r, u, b = self.crop_coord
            arrays = arrays[:, u:b, l:r]
        arrays = torch.from_numpy(arrays)

        return arrays

class ERA5Dataset(datasets.ImageFolder):
    def __init__(self, class_name='ERA5Dataset', root='/mnt/lustre/chenzhuo1/era5dx8',  mode='train', length=1, crop_coord=None):
        self.root = root
        self.mode = mode

        self.years = Years[mode]
        self.crop_coord = crop_coord
        self.file_list = self.init_file_list()
        self.length = length

        # self.transform = transforms.Compose(
        #     [
        #         transforms.to
        #     ]
        # )

    def __len__(self):
        return len(self.file_list)

    def init_file_list(self):
        file_list = []
        for year in self.years:
            # if year == 1988:
            #     max_item = 1373
            if year % 4 == 0:
                max_item = 1464
            else:
                max_item = 1460
            for hour in range(max_item):
                file_list.append([year, hour])
        return file_list

    def _load_array(self, item):

        year, hour = self.file_list[item]
        arrays = []
        for name in vnames:
            url = f"{self.root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            # array = read_npy_from_ceph(self.client, url)
            array = np.load(url)
            # print(name, np.mean(array))
            array = array[np.newaxis, :, :]
            if "geopotential" in name or 'surface_pressure' in name or 'mean_sea_level_pressure'in name:
                array = array / 1000
            arrays.append(array)
        #     print(name, np.mean(array))
        # print()
        arrays = np.concatenate(arrays, axis=0)
        if not self.crop_coord is None:
            l, r, u, b = self.crop_coord
            arrays = arrays[:, u:b, l:r]
        arrays = torch.from_numpy(arrays, )
        # print(arrays.shape)
        return arrays


    def __getitem__(self, item):
        item = min(item, len(self.file_list)-self.length)
        array_seq = []
        for i in range(self.length+1):
            array_seq.append(self._load_array(item+i))
        return array_seq


class ERA5GDataset(datasets.ImageFolder):
    def __init__(self, class_name='ERA5GDataset', root='/mnt/lustre/chenzhuo1/era5G32x64',  mode='train', length=1, crop_coord=None):
        self.root = root
        self.mode = mode
        self.vnames = vnames
        self.years = Years[mode]
        self.crop_coord = crop_coord
        self.file_list = self.init_file_list()
        self.length = length


        # self.transform = transforms.Compose(
        #     [
        #         transforms.to
        #     ]
        # )

    def __len__(self):
        return len(self.file_list)


    def init_file_list(self):
        file_list = []
        for year in self.years:
            # if year == 1988:
            #     max_item = 1373
            if year % 4 == 0:
                max_item = 1464
            else:
                max_item = 1460
            for hour in range(max_item):
                file_list.append([year, hour])
        return file_list

    def _load_array(self, item):

        year, hour = self.file_list[item]
        url = f"{self.root}/{year}/{year}-{hour:04d}.npy"
        arrays = np.load(url)
        if not self.crop_coord is None:
            l, r, u, b = self.crop_coord
            arrays = arrays[:, u:b, l:r]
        arrays = torch.from_numpy(arrays)
        # print(arrays.shape)
        return arrays


    def __getitem__(self, item):
        item = min(item, len(self.file_list)-self.length-1)
        array_seq = []
        for i in range(self.length+1):
            array_seq.append(self._load_array(item+i))
        return array_seq
