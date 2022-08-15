'''
Fixed settings for ceph files

'''

client_config_file = "/mnt/lustre/share/pymc/mc.conf"

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

# YearsAll = range(1979, 2022)
# Years4Train = range(1979, 2016)
# Years4Valid = range(2016, 2018)
# Years4Test = range(2018, 2022)
# Years = {
#     'train': Years4Train,
#     'valid': Years4Valid,
#     'test': Years4Test
# }

Shape = (720, 1440)
Shapev1 = (32, 64)