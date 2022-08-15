import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import tqdm


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



def preprocess(year):
    root_path = '/mnt/lustre/gongjunchao/era5'
    dir_save = os.path.join(root_path, '32x64', str(year))
    os.makedirs(dir_save, exist_ok=True)
    
    if year % 4 == 0:
        N = 1464
    else:
        N = 1460

    pbar = tqdm.tqdm(range(N))
    pbar.set_description(f'{year}')
    for day in pbar:
        datas = []
        for vname in vnames:
            read_path = os.path.join(root_path, vname, str(year), f'{vname:s}-{year:d}-{day:04d}.npy')
            data = np.load(read_path)
            datas.append(data.reshape(1, 32, 64))
        datas = np.concatenate(datas, axis=0)
        # import pdb; pdb.set_trace()
        np.save(os.path.join(dir_save, f'{year:d}-{day:04d}.npy'), datas)
        # c, h, w = datas.shape
        
if __name__ == "__main__":
    tasks = []
    with ThreadPoolExecutor(max_workers=10) as t:
        for year in years:
            task = t.submit(preprocess, year)
                # preprocess(vname_pair, year)
            time.sleep(1)
            tasks.append(task)
    #srun -p ai4science --gres=gpu:0 -c 96 -x SH-IDC1-10-140-0-169 python generate_mem.py
