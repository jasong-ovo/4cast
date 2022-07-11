import pickle
import os
import numpy as np
if __name__ == '__main__':
    # path_miu_var = "/mnt/lustre/chenzhuo1/era5_mean_std/10m_u_component_of_wind.pkl"
    # path_miu_var = "/mnt/lustre/chenzhuo1/era5_mean_std/surface_pressure.pkl"
    dir_data = "/mnt/lustre/chenzhuo1/era5_mean_std/"
    files = os.listdir(dir_data)
    for file in sorted(files):
        path_miu_var = os.path.join(dir_data, file)
        with open(path_miu_var, 'rb') as f:
            # pickle.dump(value_dict, f)
            data = pickle.load(f)
            # miu = data['mean']
            miu = np.float(data['mean'])
            var = np.float(data['std'])
        print(file,  ' mean', miu, 'std', var)
    print(len(files))
