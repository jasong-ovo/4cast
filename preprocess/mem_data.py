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

Years = {
    'train': range(1979, 2016),
    'valid': range(2016, 2018),
    'test': range(2018, 2022),
    'all': range(1979, 2022)

}



def preprocess(mode):
    years = Years[mode]
    root_path = '/mnt/petrelfs/gongjunchao/32x64'
    datas = []
    for year in years:
        if year % 4 == 0:
            N = 1464
        else:
            N = 1460
        pbar = tqdm.tqdm(range(N))
        pbar.set_description(f'{year}')
        # import pdb;pdb.set_trace()
        for i, day in enumerate(pbar):
            # if i > 100:
            #     break
            data_path = f'{year:d}-{day:04d}.npy'
            data = np.load(os.path.join(root_path, str(year), data_path))
            data = np.expand_dims(data, axis=0)
            datas.append(data)
        # break
    datas = np.concatenate(datas, axis=0)
    np.save(os.path.join(root_path, f'{mode}.npy'), datas)

        
if __name__ == "__main__":
    tasks = []
    # modes = ['train', 'valid', 'test']
    modes = ['train']
    preprocess('train')
    # with ThreadPoolExecutor(max_workers=10) as t:
    #     for mode in modes:
    #         task = t.submit(preprocess, mode)
    #             # preprocess(vname_pair, year)
    #         time.sleep(1)
    #         tasks.append(task)
    # srun -p ai4science --gres=gpu:0 -c 24 -x SH-IDC1-10-140-0-169 python mem_data.py