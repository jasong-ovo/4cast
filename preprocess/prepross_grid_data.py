'''
# split grid file ]1460, 721, 1440] to numpy array files [ 721, 1440]

'''


import os
import sys
sys.path.append('/mnt/lustre/chenzhuo1/IfGAN')
import io
import torch
import timm
import numpy as np

import xarray as xr

import petrel_client
from petrel_client.client import Client
import tqdm

client_config_file = "/mnt/lustre/share/pymc/mc.conf"

vnames = [
    # '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    # '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    # '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    # '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    # '50h_geopotential',
    # 'total_column_water_vapour',
    '850h_geopotential',
]
vnames_short = [
    # 'u10', 'v10', 't2m', 'sp', 'msl',
    # 'u', 'v', 'z',
    # 't', 'u', 'v', 'z', 'r',
    # 't', 'u', 'v', 'z', 'r',
    # 'z',
    # 'tcwv'
    'z',

]

# years = range(1979, 2022)
years = range(1988, 1989)

Ashape = (721, 1440)

def preprocess(vname_pair, year):
    vname, vname_short = vname_pair
    url = f's3://era5/{vname}/{vname}-{year}.grib'
    fname = f'/mnt/lustre/chenzhuo1/era5/{vname:s}-{year:d}_ceph.grib'

    if os.path.exists(fname):
        pass
    else:
        data = client.get(url)
        with open(fname, 'wb') as f:
            f.write(data)

    ds = xr.open_dataset(fname, engine="cfgrib", cache=False)
    ds_array = ds.data_vars[vname_short]
    N, H, W = ds_array.shape
    print(vname, N, H, W)
    print(ds)


    # exit()
    # dir_save = f'/mnt/lustre/chenzhuo1/era5/{vname:s}'
    # os.makedirs(dir_save, exist_ok=True)
    pbar = tqdm.tqdm(range(N))
    pbar.set_description(f'{vname} - {vname_short} - {year}')
    for i in tqdm.tqdm(range(N)):
        clip_array = ds_array[i]
        clip_array = np.array(clip_array, dtype=np.half)
        # name_save = os.path.join(dir_save, f'{vname:s}-{year:d}-{i:04d}.npy')
        # np.save(name_save, clip_array)

        url_save = f's3://era5npy/{vname}/{year}/{vname}-{year}-{i:04d}.npy'
        buff = io.BytesIO(clip_array)
        client.put(uri=url_save, content=buff)

        # clip_array_ceph = client.get(url_save)
        # clip_array_ceph = np.frombuffer(clip_array_ceph, dtype=np.half).reshape(Ashape)
        # print(clip_array[:-4, :-8])
        # print(clip_array_ceph[:-4, :-8])
        # np.savetxt(name_save, clip_array)

        # print(clip_array.shape)
        # print(type(clip_array))
    os.remove(fname)




if __name__ == '__main__':
    client = Client(conf_path="~/petreloss.conf")
    for vname_pair in zip(vnames, vnames_short):
        for year in years:
            preprocess(vname_pair, year)
        # break
        #     exit()






