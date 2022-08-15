import os
import sys
sys.path.append('/mnt/lustre/gongjunchao/IfGAN')
import io
import torch
import timm
import numpy as np

import xarray as xr
import pickle

import petrel_client
from petrel_client.client import Client
import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
client_config_file = "/mnt/lustre/share/pymc/mc.conf"

vnames = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',
]
vnames_short = [
    'u10', 'v10', 't2m', 'sp', 'msl',
    'u', 'v', 'z',
    't', 'u', 'v', 'z', 'r',
    't', 'u', 'v', 'z', 'r',
    'z',
    'tcwv'
]
years = range(1979, 2022)
# years = range(1979, 1990)
# years = range(1990, 2000)
# years = range(2000, 2010)
# years = range(2010, 2022)
Ashape = (720, 1440)


def preprocess(year):
    client = Client(conf_path="~/.petreloss.conf")

    if year % 4 == 0:
        max_item = 1464
    else:
        max_item = 1460

    dir_save = '/mnt/lustre/gongjunchao/era5G32x64/{:d}'.format(year)
    os.makedirs(dir_save, exist_ok=True)
    pbar = tqdm.tqdm(range(max_item))
    pbar.set_description('{:d}'.format(year))

    sH = 720
    sW = 1440
    tH = 32
    tW = 64
    steph = sH / float(tH)
    stepw = sW / float(tW)

    x = np.arange(0, sW, stepw).astype(np.int)
    # y = np.arange(0, sH, steph).astype(np.int)
    # x = np.linspace(0, sW-1, tW, dtype=np.int)
    y = np.linspace(0, sH-1, tH, dtype=np.int)
    x, y = np.meshgrid(x, y)
    # import pdb; pdb.set_trace()
    for i in pbar:
        arrays = []
        for iname, vname in enumerate(vnames):
            # url = f's3://era5npy/{vname}/{year}/{vname}-{year}-{i:04d}.npy'
            url = f's3://era5npy/{vname}/{year}/{vname}-{year}-{i:04d}.npy'
            array_ceph = client.get(url, update_cache=True)
            array_ceph = np.frombuffer(array_ceph, dtype=np.half).reshape(Ashape)
            array = array_ceph[y, x]
            array = array[np.newaxis, :, :]
            arrays.append(array)
        arrays = np.concatenate(arrays, axis=0).astype(np.half)
        amean = np.mean(arrays)
        # if np.isnan(np.isnan(amean)):
        if np.isnan(amean):
            print('*' * 80)
            print('*' * 80)
            print(year, i + 1, amean)
            print('*' * 80)
            print('*' * 80)
            exit()

        if (i+1) % 100 == 0:
            amin = np.min(arrays)
            amax = np.max(arrays)
            amean = np.mean(arrays)
            print(year, i+1, arrays.shape, amax, amin, amean)
            if np.isnan(amean) or np.isinf(amean):
                exit()

        path_save = os.path.join(dir_save, '{:d}-{:04d}.npy'.format(year, i))
        np.save(path_save, arrays)



if __name__ == '__main__':
    tasks = []
    # preprocess(1979)
    with ThreadPoolExecutor(max_workers=12) as t:
        for year in years:
            task = t.submit(preprocess, year)
            time.sleep(1)
            tasks.append(task)
    #srun -p ai4science --gres=gpu:0 -c 32 -x SH-IDC1-10-140-0-169 python resize_npy.py
