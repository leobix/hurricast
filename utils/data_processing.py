from __future__ import print_function
import pandas as pd
import tropycal.tracks as tracks
import math

import torch
import numpy as np

import warnings
warnings.filterwarnings('ignore')

dtype = torch.float
device = torch.device("cpu")


#convert columns to numeric values
#and interpolate missing values
def numeric_data(data):
    for i in ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND', 'STORM_SPEED', 'STORM_DIR']:
        data[i]=pd.to_numeric(data[i],errors='coerce').astype('float64')
        data[i]=data[i].interpolate(method='linear')
    return data

def smooth_features(df):
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], format= '%Y-%m-%d %H:%M:%S')
    df['cos_day'] = np.cos(2 * np.pi * df['ISO_TIME'].dt.day / 365)
    df['sin_day'] = np.sin(2 * np.pi * df['ISO_TIME'].dt.day / 365)
    df['COS_STORM_DIR'] = np.cos(2 * np.pi * df['STORM_DIR'] / 360)
    df['SIN_STORM_DIR'] = np.sin(2 * np.pi * df['STORM_DIR'] / 360)
    df['COS_LAT'] = np.cos(2 * np.pi * df['LAT'] / 360)
    df['SIN_LAT'] = np.sin(2 * np.pi * df['LAT'] / 360)
    df['COS_LON'] = np.cos(2 * np.pi * df['LON'] / 360)
    df['SIN_LON'] = np.sin(2 * np.pi * df['LON'] / 360)
    df = df.drop('STORM_DIR', axis=1)

    #df.loc[(df['ISO_TIME'].dt.hour <=11) & (df['ISO_TIME'].dt.hour >=0),'sign_day'] = 1
    return df


#to use in the future: computes the wind category
def sust_wind_to_cat_one_hot(wind):
    # maximum sustained wind in kt (knot)
    if wind<=33: cat='TD' # <=33
    elif wind<=63.:  cat='TS'
    elif wind <=82.: cat='H1'
    elif wind <=95.: cat='H2'
    elif wind <=112.: cat='H3'
    elif wind <=136.: cat='H4'
    elif wind > 136. : cat='H5'
    else: cat = 'nan'

    return cat

def sust_wind_to_cat_val(wind):
    # maximum sustained wind in kt (knot)
    if wind<=33: cat= 0 # <=33
    elif wind<=63.:  cat=1
    elif wind <=82.: cat=2
    elif wind <=95.: cat=3
    elif wind <=112.: cat=4
    elif wind <=136.: cat=5
    elif wind > 136. : cat=6
    else: cat = 7

    return cat

def add_one_hot(df, df_col, prefix):
    dummies = pd.get_dummies(df_col, prefix=prefix)
    df_out = pd.concat([df, dummies], axis=1)
    print("one-hot added for ", df_col.name)
    return df_out

def sort_storm(data, min_wind, min_steps = 5, max_steps = 120):
    '''function to create dictionary of storm matrices
    arguments:
    data we want to cut
    min_wind: the minimum wind speed to store data
    '''
    #get unique storm_id:
    SID=pd.unique(data['SID']).tolist()
    #remove empty SID
    #if not dropna: SID.remove(' ')
    #create empty dictionary
    dict0={}
    ind = 0
    for i in range(len(SID)):
        #get data of a particular SID
        M = data.loc[data['SID'] == SID[i]]
        #cut off using min wind speed
        #TODO : cut everything before, ie look for the right date
        try:
            t = M.index[M['WMO_WIND']>= min_wind][0]
            t0 = M.index[0]
        except:
            t = 0
        N = M.loc[M['WMO_WIND'] >= min_wind]
        #save matrix in dict0
        if N.shape[0] > min_steps:
            ind+=1
            dict0.update({ind:M.iloc[t-t0:max_steps+t-t0]})
    print("The dictionary of storms has been created.")
    return dict0

#instead of padding with 0, pad with latest values in loop
def pad_traj(dict0, max_steps, nan = False):
    dict1={}
    for t in dict0:
        num_steps = dict0[t].shape[0]
        steps2add = max_steps - num_steps
        if steps2add > 0:
            if nan:
                dict1[t] = pd.concat([dict0[t], pd.DataFrame([[np.nan] * dict0[t].shape[1]]*steps2add, columns=dict0[t].columns)], ignore_index=True)
            else:
                dict1[t] = pd.concat([dict0[t], pd.DataFrame([[0] * dict0[t].shape[1]]*steps2add, columns=dict0[t].columns)], ignore_index=True)
                #In fact it happens to be easier to make the change afterwards with repad
                #dict1[t] = pd.concat([dict0[t], pd.DataFrame([dict0[t].tail(1)]*steps2add, columns=dict0[t].columns)], ignore_index=True)
        else:
            dict1[t] = dict0[t][:max_steps]
    print("The trajectories have now been padded.")
    return dict1

def add_displacement_lat_lon2(dict0):
    dict1={}
    #loop over each dataframe
    for i in dict0:
        df=dict0[i]
        #reset index
        df.reset_index(inplace=True, drop=True)
        lst_lat = [0]
        lst_lon = [0]
        for j in range(1,len(df)):
            lat_j, lon_j = df['LAT'][j], df['LON'][j]
            if lat_j==0 and lon_j == 0:
                d_lat = 0
                d_lon = 0
            else:
                d_lat = df['LAT'][j] - df['LAT'][j-1]
                d_lon = df['LON'][j] - df['LON'][j-1]
            lst_lat.append(d_lat)
            lst_lon.append(d_lon)
        df['DISPLACEMENT_LAT'] = lst_lat
        df['DISPLACEMENT_LON'] = lst_lon
        dict1[i]=df
    return dict1


def add_displacement_distance_km(dict0):
    dict1 = {}
    # loop over each dataframe
    for i in dict0:
        df = dict0[i]
        #         #reset index
        df.reset_index(inplace=True, drop=True)
        lst_lat = [0]
        lst_lon = [0]
        for j in range(1, len(df)):
            lat_j, lon_j = df['LAT'][j], df['LON'][j]
            if lat_j == 0 and lon_j == 0:
                d_lat = 0
                d_lon = 0
            else:
                d_lat = df['LAT'][j] - df['LAT'][j - 1]
                d_lon = df['LON'][j] - df['LON'][j - 1]
            lst_lat.append(d_lat)
            lst_lon.append(d_lon)
        df['DISPLACEMENT_LAT'] = lst_lat
        df['DISPLACEMENT_LON'] = lst_lon
        dict1[i] = df
    return dict1

#function to calculate tensor shape
    #input: dictionary of storm data
def tensor_shape(dict0):
    #number of storms
    num_storms=len(dict0) - 1
    #number of features
    num_features=dict0[next(iter(dict0))].shape[1]

    #to compute min and max number of steps
    t_max = 0 #initialise
    t_min = 1000
    t_hist = []
    for i in dict0:
        t0 = dict0[i].shape[0]
        t_hist.append(t0)
        if  t0 > t_max:
            t_max = t0
        if t0 < t_min:
            t_min = t0
    print("There are %s storms with %s features, and maximum number of steps is %s and minimum is %s." %(num_storms,num_features,t_max, t_min))
    return num_storms, num_features, t_max, t_min, t_hist

#create a tensor
def create_tensor(data, number_of_storms):
    tensor = data[1]
    for i in range(2,number_of_storms,1):
        tensor=np.dstack((tensor, data[i]))
    #return list of features
    p_list = data[1].columns.tolist()
    print("The tensor has now been created.")
    return tensor, p_list

def repad(t):
    for i in range(t.shape[0]):
        if t[i][2][-1] == 0:
            ind = np.argmin(t[i][2])
            for j in range(ind,t.shape[2]):
                t[i,:,j]=t[i,:,ind-1]
    return t

#function to get forecast of hurricane based on name and year
def get_forecast(hurdat, name, year, pred=24): #pred: hours prediction
    try:
        storm = hurdat.get_storm((name, year))
        forecast = storm.get_operational_forecasts()
        #choose models
        model_list = set(['NAM','AP01','HWRF','SHIP', 'AEMI','CLAP5','EMXI','CMC']).intersection(forecast.keys())
        #create empty df
        df_out = pd.DataFrame(columns=['datetime'])
        for model in model_list:
            df_model = pd.DataFrame()
            for time in forecast[model]:
                df = pd.DataFrame(forecast[model][time])
                temp = df.loc[df['fhr']==pred]
                #select columns
                temp = temp[['lat','lon','vmax','mslp']]
                temp = temp.add_prefix(str(model)+'_'+str(pred)+'_')
                temp['datetime'] = pd.to_datetime(time, format = '%Y%m%d%H')
                df_model = pd.concat([df_model, temp], axis=0)
            df_out = df_out.merge(df_model, on='datetime', how='outer')
        df_out = df_out.sort_values(by='datetime')
    except:
        print('no forecast for', name, year)
        df_out = pd.DataFrame(columns=['datetime'])
    return df_out

def join_forecast(df, pred=24):
    print('joining forecast data')
    #create a dataframe to store joined data
    df_all= pd.DataFrame()

    # read hurdat data set for both basins: north atlantic and east pacific
    hurdat = tracks.TrackDataset(basin='both',source='hurdat',include_btk=False)

    #get list of storm names and year
    df['NAME'] = df['NAME'].str.lower()
    df['YEAR'] = df['ISO_TIME'].dt.year
    storm_list = df[['SID','NAME','YEAR','BASIN']].drop_duplicates(['SID']) #get list of storm names and year
    storm_list.reset_index(inplace=True, drop=True)

    for i in range(len(storm_list)):
        SID = storm_list['SID'][i]
        name = storm_list['NAME'][i]
        year= storm_list['YEAR'][i]
        basin = storm_list['BASIN'][i]

        #get stat data for particular storm
        df_stat = df.loc[df['NAME']==name]

        #no forecast for these basins
        if basin in ['SP','SI','NI','WP']:
            df_all = pd.concat([df_all, df_stat], axis=0)
        else:
            #try to get forecast for particular storm
            df_forecast = get_forecast(hurdat, name, year, pred)
            if df_forecast.shape[0]>0:
                #join with dataframe
                df_forecast['NAME']= name
                df_joined = df_stat.merge(df_forecast, how='left', left_on=['ISO_TIME','NAME'], right_on=['datetime','NAME'])
                df_joined.drop('datetime', axis=1, inplace=True)
                df_all = pd.concat([df_all, df_joined], axis=0)
            else:
                df_all = pd.concat([df_all, df_stat], axis=0)
    #drop added columns
    df_all.drop(['NAME','YEAR'], inplace=True, axis=1)
    df_all.reset_index(inplace=True, drop=True)
    print("The dataframe of storms with forecast has been created.")
    return df_all

def prepare_tabular_data_vision(path="./data/last3years.csv", min_wind=34, min_steps=20,
                  max_steps=120, get_displacement=True, forecast=True, predict_period = 24, one_hot = True):
    data = pd.read_csv(path)
    data.drop(0, axis=0, inplace=True) #drop secondary column names
    # select interesting columns
    df0 = data[['SID','BASIN','NAME', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND', 'STORM_SPEED', 'STORM_DIR']]
    # transform data from String to numeric
    df0 = numeric_data(df0)
    # smooth cos & sign of day
    df0 = smooth_features(df0)
    # add wind category
    df0['wind_category'] = df0.apply(lambda x: sust_wind_to_cat_val(x['WMO_WIND']), axis=1)
    if forecast:
        #join forecast
        df0 = join_forecast(df0, predict_period)
        print('df0 columns :', df0.columns)
    if one_hot:
        #adding BASIN and NATURE feature as a one hot
        df0 = add_one_hot(df0, data['BASIN'], 'basin')
        # df0 = add_one_hot(df0, data['NATURE'], 'nature')
        #add category one_hot
        #df0 = add_one_hot(df0, df0['wind_category'], 'category')

    #drop category column
    df0.drop(['BASIN'], axis=1, inplace=True)

    print('df0 columns :', df0.columns)
    # get a dict with the storms with a windspeed and number of timesteps greater to a threshold
    storms = sort_storm(df0, min_wind, min_steps)
    # pad the trajectories to a fix length
    d = pad_traj(storms, max_steps)
    # print(d)
    if get_displacement:
        d = add_displacement_lat_lon2(d)
    # print the shape of the tensor
    m, n, t_max, t_min, t_hist = tensor_shape(d)
    # create the tensor
    t, p_list = create_tensor(d, m)

    #put t in format storm * timestep * features
    e = t.transpose((2, 0, 1))
    for tt in e:
        try:
            tt[0] = datetime.strptime(tt[0], "%Y-%m-%d %H:%M:%S")
        except:
            pass
    return e[:, :, 1:], d, e



# -------- old codes: ------ #
# def prepare_data(path = "/data/last3years.csv", max_wind_change = 12, min_wind = 50, min_steps = 15, max_steps = 120, secondary = False, one_hot=False, dropna = False):
#     data = pd.read_csv(path)
#     #select interesting columns
#     df0 = select_data(data)
#     #transform data from String to numeric
#     df0 = numeric_data(df0)
#     #if dropna: df0 = df0.dropna()
#     #add one_hot columns:
#     if one_hot:
#         #add one-hot storm category
#         #df0 = add_storm_category_val(df0)
#         df0 = add_storm_category_one_hot(df0)
#         #transform basin and nature of the storm into one-hot vector
#         df0 = add_one_hot(data, df0)
#     if secondary:
#         #add the max-wind-change column
#         df0 = get_max_wind_change(df0, max_wind_change)
#
#     #get a dict with the storms with a windspeed greater to a threshold
#     storms = sort_storm(df0, min_wind, min_steps)
#     #pad the trajectories to a fix length
#     d = pad_traj(storms, max_steps)
#     #print(d)
#     if secondary:
#         #d = add_displacement_distance(d)
#         d = add_displacement_lat_lon2(d)
#     #print the shape of the tensor
#     m, n, t_max, t_min, t_hist = tensor_shape(d)
#     #create the tensor
#     t, p_list = create_tensor(d, m)
#     #delete id and number of the storms
#     t2 = torch.Tensor(t[:,3:,:].astype('float64'))
#     #match feature list
#     p_list = p_list[3:]
#     #transpose time and sample
#     t3 = torch.transpose(t2,0,2)
#     #replace 0 by latest values in the tensor
#     t3 = repad(t3)
#     return t3, p_list


# #allows to keep only specific columns
def select_data(data):
     return data[['SID', 'NUMBER', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND', 'STORM_SPEED', 'STORM_DIR']]

def get_distance_km(lon1, lat1, lon2, lat2):
     '''
     Using haversine formula (https://www.movable-type.co.uk/scripts/latlong.html)
     '''
     R=6371e3 # meters (earth's radius)
     phi_1=math.radians(lat1)
     phi_2 = math.radians(lat2)
     delta_phi=math.radians(lat2-lat1)
     delta_lambda=math.radians(lon2-lon1)
     a=np.power(math.sin(delta_phi/2),2) + math.cos(phi_1)*math.cos(phi_2)\
       * np.power(math.sin(delta_lambda/2),2)
     c= 2 * math.atan2(math.sqrt(a),math.sqrt(1-a))

     return R*c/1000.
#
# #compute the displacement from t=0
# def add_displacement_distance(dict0):
#     dict1={}
#     #loop over each dataframe
#     for i in dict0:
#         df=dict0[i]
#         #reset index
#         df.reset_index(inplace=True, drop=True)
#         #calculate difference from t=0
#         df['DISPLACEMENT'] = 0
#         for j in range(1,len(df)):
#             d = get_distance_km(df['LON'][j-1], df['LAT'][j-1], df['LON'][j], df['LAT'][j])
#             if d > 500: d=0
#             df['DISPLACEMENT'][j] = d
#         dict1[i]=df
#     return dict1


# #Geographical difference features: i.e. feature_1(t) = feature(t)-feature(0)
#     # features: LAT, LON, DIST2LAND
# def geo_diff(dict0):
#     dict1={}
#     #loop over each dataframe
#     for i in dict0:
#         df=dict0[i]
#         #reset index
#         df.reset_index(inplace=True, drop=True)
#         #calculate difference from t=0
#         df['LAT_1']= df['LAT'] - df['LAT'][0]
#         df['LON_1']= df['LON'] - df['LON'][0]
#         df['DIST2LAND_1']= df['DIST2LAND'] - df['DIST2LAND'][0]
#         #substitute back to the dictionary
#         dict1[i]=df
#     return dict1

# #This code allows to get the maximum wind change in the last X hours.
# def get_max_change(data, time, i):
#     t = time//3
#     try:
#         val = max(data['WMO_WIND'][i-t:i])-min(data['WMO_WIND'][i-t:i])
#     except:
#         val = 'NaN'
#     return val
#
# #please specify a multiple of 3h for the time
# def get_max_wind_change(data, time):
#     df = data
#     df['max_wind_change']=[get_max_change(data, time, i) for i in range(len(data))]
#     return df


# def add_storm_category_one_hot(data):
#     df = pd.DataFrame()
#     df['storm_category'] = [sust_wind_to_cat_one_hot(data['WMO_WIND'][i]) for i in range(len(data))]
#     storm_cat = pd.get_dummies(df['storm_category'],prefix='storm_category')
#     #storm_cat
#     storm_cat.drop('storm_category_nan', axis=1, inplace=True)
#     frames = [data, storm_cat]
#     df0 = pd.concat(frames, axis = 1)
#     #df0.drop('storm_category', axis=1)
#     print("Storm category is now added and one-hot.")
#     return df0

# def add_storm_category_val(df):
#     df['storm_category'] = df['WMO_WIND'].apply(sust_wind_to_cat_val) #add storm category
#     return df
