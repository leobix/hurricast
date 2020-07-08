import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#------main function ------- # 

def prepare_data(since_1980=False, basin = 'EP'):
    if since_1980 == True:
        path = "ibtracs.since1980.list.v04r00.csv"
    else:
        path = "ibtracs.last3years.list.v04r00.csv"
    #read csv 
    df= pd.read_csv(path) 
    df= df.drop(df.index[0]) #drop first row  
    df = df.replace(' ', np.nan) #fill space with nan
    
    #drop columns with nan threashold > 0.6
    df = df.dropna(axis=1, thresh=int(0.5*len(df)))
    
    df = select_data(df) #select useful columns 
    df = numeric_data(df) #formate to numeric 
    
    #select based on basin 
    storm_dict = storm_to_dict(df, basin, min_steps=30, min_wind=30)
    
    #create tensor based on storm_dict 
    tensor, p_list  = dict_to_tensor(storm_dict)
    return tensor, p_list 



#------subfunctions------# 

#allows to keep only specific columns
def select_data(data):
    return data[['SID', 'BASIN','ISO_TIME', 'DIST2LAND', 'LAT', 'LON',  'USA_WIND',
       'USA_PRES', 'USA_SSHS', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW',
       'USA_R34_NW', 'USA_POCI', 'USA_ROCI', 'USA_RMW', 'STORM_SPEED',
       'STORM_DIR']]#, 'BASIN', 'NATURE']]

#convert columns to numeric values
def numeric_data(data):
    for i in ['DIST2LAND', 'LAT', 'LON',  'USA_WIND',
       'USA_PRES', 'USA_SSHS', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW',
       'USA_R34_NW', 'USA_POCI', 'USA_ROCI', 'USA_RMW', 'STORM_SPEED',
       'STORM_DIR']:
        data[i]=pd.to_numeric(data[i],errors='coerce').astype('float64')
        data[i]=data[i].interpolate(method='linear')
        data[i]=data[i].fillna(method='ffill')
        data[i]=data[i].fillna(method='bfill')
    return data

#select storms belong to certain basin, storm them in a dictionary 
def storm_to_dict(data, basin='EP', min_steps=30, min_wind=30): 
    #create empty dictionary
    dict0={}
    ind = 0
    #groupby SID 
    grouped = data.groupby('SID')
    for name, df in grouped:
#         df.reset_index(inplace=True)#reset index 
        if df['BASIN'].iloc[0]==basin:  #select only particular basin 
            if df.shape[0]>=min_steps: #select only if enough data is present 
                df = df.loc[df['USA_WIND']>=min_wind] #keep only greater than min speed
                df = df[['DIST2LAND', 'LAT', 'LON',  'USA_WIND',
       'USA_PRES', 'USA_SSHS', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW',
       'USA_R34_NW', 'USA_POCI', 'USA_ROCI', 'USA_RMW', 'STORM_SPEED',
       'STORM_DIR']] #keep only numerical columns 
                dict0.update({ind:df})
                ind+=1
    print('number of storms in basin %s is %s' %(basin, len(dict0)))
    return dict0  


#create a tensor
def dict_to_tensor(storm_dict):
    d0, d1, d2 = get_tensor_shape(storm_dict) #d2 = max time step among hurricanes 
    tensor = np.zeros((d0,d1,d2)) #pad with zero 
    
    for i in storm_dict:
        m = storm_dict[i]
        d2 = m.shape[0]
        tensor[i,:,:d2] = m.to_numpy().T
    
    #return list of features 
    p_list = storm_dict[0].columns.tolist()
    return tensor, p_list


#calculate maximum time steps among hurricanes 
def get_tensor_shape(dict0):
    tmax= 0 
    #get the max time steps: 
    for i in dict0:
        if  dict0[i].shape[0]> tmax :
            tmax = dict0[i].shape[0]
    
    #dimensions 
    d0 = len(dict0)
    d1 = dict0[0].shape[1]
    d2=tmax
    print('tensor shape is ', [d0,d1,d2])
    return d0, d1, d2


