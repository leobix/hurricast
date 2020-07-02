#!/usr/bin/env python3

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

window_size = 16

X_train = np.load('data/X_train_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)
X_test = np.load('data/X_test_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)

tgt_intensity_train = np.load('data/y_train_intensity_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)
tgt_intensity_test = np.load('data/y_test_intensity_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)

tgt_intensity_cat_train = np.load('data/y_train_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)
tgt_intensity_cat_test = np.load('data/y_test_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)

fresh = np.load('data/test_features_filtered_direct_w' + str(window_size) + '.npy', allow_pickle=True)

names = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND',
         'STORM_SPEED', 'STORM_DIR', 'storm_category', 'cat_basin_EP', 'cat_basin_NI',
         'cat_basin_SI', 'cat_basin_SP', 'cat_basin_WP', 'cat_nature_DS', 'cat_nature_ET',
         'cat_nature_MX', 'cat_nature_NR', 'cat_nature_SS', 'cat_nature_TS',
         'cat_UNKNOWN',
         'STORM_DISPLACEMENT_X', 'STORM_DISPLACEMENT_Y']

names_all = names * window_size

for i in range(len(names_all)):
    names_all[i] += '_' + str(i // 22)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train.columns = names_all
X_test.columns = names_all

cols = [c for c in X_train.columns if c.lower()[-1] == '0' or c.lower()[:3] != 'cat']

X_train = np.array(X_train[cols])
X_test = np.array(X_test[cols])
n = tgt_intensity_train.shape[0]
train = np.concatenate((fresh[:n], X_train), axis=1)
test = np.concatenate((fresh[n:], X_test), axis=1)

for i in range(3, 6)
    xgb = XGBRegressor(max_depth=i, n_estimators=100)
    xgb.fit(train, tgt_intensity_train)
    yhat = xgb.predict(test)
    print("XGB score depth", i, ": ", mean_absolute_error(tgt_intensity_test, yhat))


for i in range(3, 6)
    xgb = XGBClassifier(max_depth=i, n_estimators=100)
    xgb.fit(train, tgt_intensity_cat_train)
    yhat = xgb.predict(test)
    print("XGB score depth", i, ": ", accuracy_score(tgt_intensity_cat_test, yhat))