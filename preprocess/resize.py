import os
import sys
sys.path.append('/mnt/lustre/chenzhuo1/IfGAN')
import time
import numpy as np
from configs.ceph import client_config_file, vnames, vnames_short, Shape
from configs.ceph import Years, Years4Train, Years4Test, Years4Valid


from concurrent.futures import ThreadPoolExecutor
from petrel_client.client import Client


client = Client(conf_path="~/petreloss.conf")
# dir_save = '/mnt/lustre/chenzhuo1/era5R32x64'

def read_npy_from_ceph(client, url, Ashape=Shape):
    array_ceph = client.get(url)
    array_ceph = np.frombuffer(array_ceph, dtype=np.half).reshape(Ashape)
    return array_ceph


def resize_npy(vname, year):
    dir_source = "s3://era5npy/{:s}/{:d}".format(vname, year)
    if year % 4 == 0: nums = 1464
    else: nums = 1460
    dir_target = '/mnt/lustre/chenzhuo1/era5R32x64'
    # dir_target = '/mnt/lustre/chenzhuo1/era5R64x128'
    dir_save = "{:s}/{:s}/{:d}".format(dir_target, vname, year)
    os.makedirs(dir_save, exist_ok=True)

    print(vname, year, nums)
    for i in range(nums):

        url = "{:s}/{:s}-{:d}-{:04d}.npy".format(dir_source, vname, year, i)
        path_save = "{:s}/{:s}-{:d}-{:04d}.npy".format(dir_save, vname, year, i)
        array_ceph = read_npy_from_ceph(client, url, Ashape=Shape)
        # print("array_ceph", array_ceph.shape)
        # array_ceph = array_ceph[::8, ::8]

        array_ceph = np.resize(array_ceph, (32, 64))
        # array_ceph = np.resize(array_ceph, (64, 128))
        np.save(path_save, array_ceph)
        # print("array_ceph", array_ceph.shape)



    return year



if __name__ == '__main__':
    tasks = []
    print("start")
    with ThreadPoolExecutor(max_workers=8) as t:
        for vname in ["850h_geopotential"]:
            for year in range(1988, 1989):
                task = t.submit(resize_npy, vname, year)
                time.sleep(1)
                tasks.append(task)
    # "s3://era5npy/10m_u_component_of_wind/1981/10m_u_component_of_wind-1981-0.npy"
    # url = 's3://era5npy/10m_u_component_of_wind/1983/10m_u_component_of_wind-1983-0000.npy'
    # array_ceph = read_npy_from_ceph(client, url, Ashape=Shape)
    # print("array_ceph", array_ceph.shape)