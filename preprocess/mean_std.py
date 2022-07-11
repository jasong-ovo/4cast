import numpy as np
import tqdm
import time
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

vnames = [
    # '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    # '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    # '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    # '500h_temperature',
    '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',
]
vnames_short = [
    # 'u10', 'v10', 't2m', 'sp', 'msl',
    # 'u', 'v', 'z',
    # 't', 'u', 'v', 'z', 'r',
    # 't',
    'u', 'v', 'z', 'r',
    'z',
    'tcwv'
]

years = range(1979, 2022)
# vmeans = [0.01, 0.01, 1, 1000, 1000,
#           0.01, 0.01, 1000,
#           1, 0.01, 0.01, 1000, 1,
#           1, 0.01, 0.01, 1000, 1,
#           1000,
#           0.1
# ]

def compute_mean_std(vname_pair):
    name, vname_short = vname_pair
    dir_save = '/mnt/lustre/chenzhuo1/era5_mean_std'
    os.makedirs(dir_save, exist_ok=True)
    dir_data = '/mnt/lustre/chenzhuo1/era5R32x64/'
    # pbar = tqdm.tqdm(years)
    # pbar.set_description('{:s}'.format(name))
    # print()
    arrays = []
    for year in years:
        if year % 4 == 0:
            max_item = 1464
        else:
            max_item = 1460
        for i in range(0, max_item, 10):
            url = f"{dir_data}/{name}/{year}/{name}-{year}-{i:04d}.npy"
            array = np.load(url)
            array = np.resize(array, (32, 64))
            array = array[np.newaxis, :, :]
            arrays.append(array)
    arrays = np.concatenate(arrays, axis=0).astype(np.float64)

    value_std = np.std(arrays)

    value_mean = np.mean(arrays)
    value_dict = {
        "mean": value_mean,
        "std": value_std
    }
    path_save = os.path.join(dir_save, '{:s}.pkl'.format(name))
    with open(path_save, 'wb') as f:
        pickle.dump(value_dict, f)
    print(name, arrays.shape, value_mean, value_std, np.min(arrays), np.max(arrays))

if __name__ == '__main__':
    # tasks = []
    # with ThreadPoolExecutor(max_workers=1) as t:
    #     for vname in vnames:
    #         task = t.submit(compute_mean_std, vname)
    #         time.sleep(1)
    #         tasks.append(task)

    for vname_pair in zip(vnames, vnames_short):
        compute_mean_std(vname_pair)
        time.sleep(1)