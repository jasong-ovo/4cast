import os
import sys
sys.path.append('/mnt/lustre/gongjunchao/IfGAN')
import io
import torch
import timm
import numpy as np
import pickle
import xarray as xr
import time
import petrel_client
from petrel_client.client import Client
import tqdm
from concurrent.futures import ThreadPoolExecutor
client_config_file = "/mnt/lustre/share/pymc/mc.conf"
import cfgrib

user='gongjunchao'

# surface_pressure inf, mean_sea_level_pressure inf, 50h_geopotential inf
vnames = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',
    # '850h_geopotential',
    # 'surface_pressure', 'mean_sea_level_pressure', '50h_geopotential',
    # '850h_geopotential'
]
vnames_short = [
    'u10', 'v10', 't2m', 'sp', 'msl',
    'u', 'v', 'z',
    't', 'u', 'v', 'z', 'r',
    't', 'u', 'v', 'z', 'r',
    'z',
    'tcwv'
    # 'z',
    # 'sp', 'msl', 'z',
    # 'z',
]

# years = range(1979, 2022)
years = range(1979, 2022)

Ashape = (721, 1440)



def preprocess(vname_pair, year):
    vname, vname_short = vname_pair
    url = f's3://era5/{vname}/{vname}-{year}.grib'
    fname = f'/mnt/lustre/gongjunchao/era5/{vname:s}-{year:d}_ceph.grib'
    dir_save = f'/mnt/lustre/gongjunchao/era5/{vname:s}/{year:d}'
    os.makedirs(dir_save, exist_ok=True)

    if os.path.exists(fname):
        pass
    else:
        data = client.get(url)
        with open(fname, 'wb') as f:
            f.write(data)

    # import pdb; pdb.set_trace()
    ds = xr.open_dataset(fname, engine="cfgrib", cache=False)
    ds_array = ds.data_vars[vname_short]
    N, H, W = ds_array.shape
    # print(vname, N, H, W)
    # print(ds)
    sH = 720
    sW = 1440
    tH = 32
    tW = 64
    steph = sH / float(tH)
    stepw = sW / float(tW)
    x = np.arange(0, sW, stepw).astype(int)
    y = np.linspace(0, sH-1, tH, dtype=int)
    x, y = np.meshgrid(x, y)
    # exit()
    pbar = tqdm.tqdm(range(N))
    pbar.set_description(f'{vname} - {vname_short} - {year}')
    for i in pbar:
        clip_array = ds_array[i]
        clip_array = np.array(clip_array, dtype=float)
        # clip_array = np.resize(clip_array, (32, 64))
        lowRe_array = clip_array[y, x]
        path_save = f'/mnt/lustre/gongjunchao/era5/{vname:s}/{year:d}/{vname:s}-{year:d}-{i:04d}.npy'
        # import pdb; pdb.set_trace()
        # clip_array = np.resize(clip_array, (64, 128))
        # path_save = f'/mnt/lustre/gongjunchao/era5R64x128/{vname:s}/{year:d}/{vname:s}-{year:d}-{i:04d}.npy'
        np.save(path_save, lowRe_array)
        # print(vname, year, i, np.mean(clip_array))

        # name_save = os.path.join(dir_save, f'{vname:s}-{year:d}-{i:04d}.npy')
        # np.save(name_save, clip_array)

        # url_save = f's3://era5npy/{vname}/{year}/{vname}-{year}-{i:04d}.npy'
        # buff = io.BytesIO(clip_array)
        # client.put(uri=url_save, content=buff)

        # clip_array_ceph = client.get(url_save)
        # clip_array_ceph = np.frombuffer(clip_array_ceph, dtype=np.half).reshape(Ashape)
        # print(clip_array[:-4, :-8])
        # print(clip_array_ceph[:-4, :-8])
        # np.savetxt(name_save, clip_array)

        # print(clip_array.shape)
        # print(type(clip_array))
    os.remove(fname)




if __name__ == '__main__':
    client = Client(conf_path="~/.petreloss.conf")
    tasks = []
    # preprocess(('10m_u_component_of_wind', 'u10'), 1988)
    with ThreadPoolExecutor(max_workers=10) as t:
        for vname_pair in zip(vnames, vnames_short):
            for year in years:
                task = t.submit(preprocess, vname_pair, year)
                # preprocess(vname_pair, year)
                time.sleep(1)
                tasks.append(task)
    # srun -p ai4science --gres=gpu:0 -c 32 -x SH-IDC1-10-140-0-169 python grib2npy.py


