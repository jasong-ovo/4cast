import numpy as np
import os
import pickle

means = np.load("means.npy")
stds = np.load('stds.npy')

vnames = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',

]

save_dir = '/mnt/petrelfs/gongjunchao/32x64/means_stds/'
os.makedirs(save_dir, exist_ok=True)

for i, vname in enumerate(vnames):
    path = os.path.join(save_dir, vname+'.pkl')
    content = {'mean':means[i], 'std':stds[i]}
    print("####################")
    print(vname)
    print(content)
    with open(path, 'wb') as f:
        pickle.dump(content, f)

