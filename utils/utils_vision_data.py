import cdsapi
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from datetime import datetime
from utils.data_processing import *
import os
import warnings; warnings.simplefilter('ignore')


#All the following functions are used for processing vision data from ERA5



def process_netcdf(filepath, param):
    '''
    input: netcdf filepath and the specific corresponding parameter in str format (eg. 'z', 'u', 'v'...)
    '''
    nc = netCDF4.Dataset(filepath, mode='r')
    nc.variables.keys()
    #lat = nc.variables['latitude'][:]
    #lon = nc.variables['longitude'][:]
    #time_var = nc.variables['time']
    #dtime = netCDF4.num2date(time_var[:],time_var.units)
    grid = nc.variables[param][:]
    #transform into np.array format and reshape something in (1,grid_size,grid_size) into (grid_size,grid_size)
    grid = np.array(grid).reshape(grid.shape[1],grid.shape[2], grid.shape[3])
    return grid


def get_storms(extraction = False, min_wind = 30, min_steps= 20, max_steps=60, path = "since1980.csv"):
    '''
    returns an array of elements of type [datetime, lat, lon]
    set extraction to True if used for downloading data and False if used to convert netcdf files to tensor
    '''
    data = prepare_data2(path = path, min_wind = min_wind, min_steps= min_steps, max_steps=max_steps, one_hot = False, secondary = False)
    e = data.transpose((2,0,1))
    d = e.reshape(e.shape[0]*e.shape[1],3)
    for t in d:
        try:
            t[0] = datetime.strptime(t[0], "%Y-%m-%d %H:%M:%S")
        except:
            pass
    if extraction :
        f = d.reshape(e.shape[0],e.shape[1],3)
        return f
    return d



def get_timestep_vision(time, lat, lon):
    '''
    given a datetime and latitute, longitude returns a processed array obtained after donwload
    '''
    filepath = get_filename(['700', '500', '225'], ['geopotential', 'u_component_of_wind', 'v_component_of_wind'], time, lat, lon)
    u, v, z = process_netcdf(filepath, 'u'), process_netcdf(filepath, 'v'), process_netcdf(filepath, 'z')
    return np.array([u, v, z])


def get_storm_vision(storm, epsilon = 0):
    '''
    given a storm (list of timesteps with time and lat/lon), returns the vision array
    epsilon is a parameter in case there is a scenario whith not correct grid size
    '''
    l = np.zeros((len(storm), 3, 3, 25, 25))
    for i in range(len(storm)):
        time, lat, lon = storm[i]
        try :
            l[i]=get_timestep_vision(time, lat, lon)
        except:
            try :
                b = get_timestep_vision(time, lat, lon)
                print(b.shape)
                print(time, lat, lon)
                get_data(['700', '500', '225'], ['geopotential', 'u_component_of_wind', 'v_component_of_wind'], time, lat, lon, grid_size = 25, force = True, epsilon = epsilon)
            except:
                pass
    return l


def extract_vision(data, epsilon):
    '''
    processes all the data to get the vision array
    '''

    vision = []
    for storm in data:
        vision.append(get_storm_vision(storm, epsilon))
    return np.array(vision)


def get_filename(pressure, params, time, lat, lon):
    '''
    returns filename to save the netcdf file
    '''
    params_str = '_'.join(map(str, params))
    pressure_str = '_'.join(map(str, pressure))
    year, month, day, hour = str(time.year), str(time.month), str(time.day), str(time.hour)
    return 'data_era/'+params_str+'/eradata_'+pressure_str+'hPa'+'_'+year+'_'+month+'_'+day+'_'+hour+'_'+'coord'+'_'+str(lat)+'_'+str(lon)+'.nc'



def get_area(lat, lon, grid_size, e = 0.0):
    '''
    input : center of the storm, with lat and lon ; grid_size and error parameter in case
    output: returns a centered squared grid of size grid_size degrees
    '''
    val = grid_size // 2
    return [lat + val + e, lon - val, lat - val - e/10, lon + val]


def get_data(pressure_level, params, time, lat, lon, grid_size=25, degbypix=1.0, force=False, epsilon=0):
    '''
    pressure_level is the the pressure level we wish to get the data.
    params has to be in format e.g: 'geopotential' or 'u_component_of_wind' or 'v_component_of_wind'
    grid_size should be odd
    '''
    if not os.path.exists(get_filename(pressure_level, params, time, lat, lon)) or force:
        c = cdsapi.Client()
        year, month, day, hour = str(time.year), str(time.month), str(time.day), str(time.hour)

        c.retrieve('reanalysis-era5-pressure-levels', {
            'variable': params,
            'pressure_level': pressure_level,
            'product_type': 'reanalysis',
            'year': year,
            'month': month,
            'day': day,
            'area': get_area(lat, lon, grid_size, epsilon),  # North, West, South, East. Default: global
            'grid': [degbypix, degbypix],
            # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
            'time': hour,
            'format': 'netcdf'  # Supported format: grib and netcdf. Default: grib
        }, get_filename(pressure_level, params, time, lat, lon))
    else:
        print("Already downloaded", get_filename(pressure_level, params, time, lat, lon))


def download_all2(data):
    i = 0
    for storm in data:
        for t in storm:
            time, lat, lon = t[0], t[1], t[2]
            try:
                get_data(['700', '500', '225'], ['geopotential', 'u_component_of_wind', 'v_component_of_wind'], time, lat, lon, grid_size = 25)
            except:
                print("False request.")
        i+=1
        print("Storm ", i, " completed.")
    print("Download complete.")


def create_dataset(min_wind, min_steps, max_steps, vision_name, y_name, path = './data/last3years.csv', save_path = 'data/'):
    '''
    create_dataset(30, 16, 120, 'vision_data_30_16_120_3years.npy', 'y_30_16_120_3years.npy')
    :param min_wind:
    :param min_steps:
    :param max_steps:
    :param vision_name:
    :param y_name:
    :param path:
    :return: nothing but creates the datasets
    '''
    data = get_storms(min_wind = min_wind, min_steps = min_steps, max_steps = max_steps, path = path, extraction=True)
    vision_data = extract_vision(data, epsilon=0)
    y, _, y_with_timestamp = prepare_tabular_data_vision(min_wind=min_wind,
                                        min_steps=min_steps,
                                        max_steps=max_steps,
                                        path = path)
    np.save(save_path + vision_name, vision_data, allow_pickle=True)
    np.save(save_path + y_name, y, allow_pickle=True)
