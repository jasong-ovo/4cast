import os
import cdsapi
import time
from concurrent.futures import ThreadPoolExecutor

# vname = 'geopotential'
# vname = 'u_component_of_wind'
# vname = 'v_component_of_wind'
# vname = 'temperature'
# vname = 'relative_humidity'


vname = 'geopotential'
# sname = '50h_geopotential'
# sname = '500h_geopotential'
sname = '850h_geopotential'
# sname = '1000h_geopotential'
# sname = '1000h_u_component_of_wind'
# sname = '1000h_v_component_of_wind'
# sname = '850h_u_component_of_wind'
# sname = '850h_v_component_of_wind'
# sname = '500h_u_component_of_wind'
# sname = '500h_v_component_of_wind'
# sname = '500h_temperature'
# sname = '500h_relative_humidity'
# sname = '850h_temperature'
# sname = '850h_relative_humidity'

# vname = 'total_column_water_vapour'
# sname = 'total_column_water_vapour'
#
dir_save = f'/mnt/lustre/chenzhuo1/era5/{sname}'
os.makedirs(dir_save, exist_ok=True)

def request(year):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        # 'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': vname,
            'pressure_level': '850',
            'year': [
                f'{year:d}',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
            'format': 'grib',
        },
        # f'C:\ERA5\{vname:s}-{year:d}.grib')
        f'{dir_save}/{sname:s}-{year:d}.grib')
    return year

if __name__ == '__main__':
    tasks = []
    max_workers = 2
    with ThreadPoolExecutor(max_workers=max_workers) as t:
        for year in range(1987, 1988):
            task = t.submit(request, year)
            tasks.append(task)
