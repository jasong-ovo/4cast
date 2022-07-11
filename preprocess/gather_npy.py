import numpy as np
import tqdm
import time
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

vnames = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',
]
vnames_short = [
    # 'u10', 'v10', 't2m', 'sp', 'msl',
    # 'u', 'v', 'z',
    # 't', 'u', 'v', 'z', 'r',
    # 't', 'u', 'v', 'z', 'r',
    # 'z',
    # 'tcwv'
    # 'z',
    # 'sp', 'msl', 'z',
]

# years = range(1979, 2022)
years = range(1979, 1980)
years = range(1980, 1990)
years = range(1990, 2000)
years = range(2000, 2010)
years = range(2010, 2020)
years = range(2020, 2022)
# vmeans = [0.01, 0.01, 1, 1000, 1000,
#           0.01, 0.01, 1000,
#           1, 0.01, 0.01, 1000, 1,
#           1, 0.01, 0.01, 1000, 1,
#           1000,
#           0.1
# ]

def preprocess(year):
    if year % 4 == 0:
        max_item = 1464
    else:
        max_item = 1460

    mius = []
    vars = []
    for vname in vnames:
        path_miu_var = "/mnt/lustre/chenzhuo1/era5_mean_std/{:s}.pkl".format(vname)
        with open(path_miu_var, 'rb') as f:
            # pickle.dump(value_dict, f)
            data = pickle.load(f)
        miu = np.float(data['mean'])
        var = np.float(data['std'])
        mius.append(miu)
        vars.append(var)

    dir_save = '/mnt/lustre/chenzhuo1/era5G32x64/{:d}'.format(year)
    os.makedirs(dir_save, exist_ok=True)
    dir_data = '/mnt/lustre/chenzhuo1/era5R32x64/'
    pbar = tqdm.tqdm(range(max_item))
    pbar.set_description('{:d}'.format(year))

    for i in pbar:
        arrays = []
        # mean_array = []
        for iname, name in enumerate(vnames):
            url = f"{dir_data}/{name}/{year}/{name}-{year}-{i:04d}.npy"
            array = np.load(url)
            array = np.resize(array, (32, 64))
            array = array[np.newaxis, :, :]
            array = array.astype(np.float)
            array = (array - mius[iname]) / vars[iname]
            arrays.append(array)

        arrays = np.concatenate(arrays, axis=0).astype(np.half)
        # print(i, np.mean(arrays), np.min(arrays),np.max(arrays))
        # arrays = np.array(arrays, dtype=np.half)
        path_save = os.path.join(dir_save, '{:d}-{:04d}.npy'.format(year, i))
        np.save(path_save, arrays)

    # mean_arrays = np.array(mean_arrays)
    # mean_arrays = np.mean(mean_arrays, axis=0)
    # desc = "Year: {:d} | mean:{.2f} max:{.2f} min:{.2f}".format(year, np.mean(mean_arrays), np.max(mean_arrays),
    #                                                             np.min(mean_arrays))
    # print(desc)
    # for i in range(len(vnames)):
    #     print(vnames[i], mean_arrays[i])
    # print(mean_array)




if __name__ == '__main__':
    tasks = []
    with ThreadPoolExecutor(max_workers=10) as t:
        for year in years:
            task = t.submit(preprocess, year)
            time.sleep(1)
            tasks.append(task)