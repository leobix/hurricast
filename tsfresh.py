#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tsfresh import extract_features
from tsfresh import extract_relevant_features

window_size = 16

X_train = np.load('data/X_train_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle = True)
X_test = np.load('data/X_test_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle = True)

tgt_intensity_train = np.load('data/y_train_intensity_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)
tgt_intensity_test = np.load('data/y_test_intensity_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)

def prepare_fresh(X):
    data = X.reshape(X.shape[0], window_size, X.shape[1]//window_size)
    n, t, p = data.shape
    new_data = np.zeros((n, t, p+2))
    for i in range(n):
        for j in range(t):
            new_data[i,j,2:] = data[i,j]
            new_data[i,j,0] = int(i)
            new_data[i,j,1] = int(j)
    new_data = new_data.reshape(n*t, -1)
    new_data = pd.DataFrame(new_data).rename(columns={0: "storm_id", 1: "time"})
    new_data = new_data.astype({"storm_id": 'int32', "time": 'int32'})
    for i in range(p+2):
        new_data = new_data.rename(columns={i:"feat_"+str(i-1)})
    return new_data

names = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND',
         'STORM_SPEED', 'STORM_DIR', 'storm_category', 'cat_basin_EP', 'cat_basin_NI',
'cat_basin_SI', 'cat_basin_SP', 'cat_basin_WP', 'cat_nature_DS', 'cat_nature_ET',
'cat_nature_MX', 'cat_nature_NR', 'cat_nature_SS', 'cat_nature_TS',
'cat_UNKNOWN', 
'STORM_DISPLACEMENT_X', 'STORM_DISPLACEMENT_Y']

names_all = names * window_size

for i in range(len(names_all)):
    names_all[i] += '_' + str(i//22)

    
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train.columns = names_all
X_test.columns = names_all

cols = [c for c in X_train.columns if c.lower()[:3] != 'cat']

X_train = np.array(X_train[cols])
X_test = np.array(X_test[cols])

a = prepare_fresh(np.concatenate((X_train, X_test), axis = 0))
features_filtered_direct = extract_relevant_features(a, pd.Series(np.concatenate((tgt_intensity_train, tgt_intensity_test), axis = 0)), column_id='storm_id', column_sort='time')

np.save('data/features_filtered_direct_w' + str(window_size) +'.npy', np.array(features_filtered_direct), allow_pickle = True)