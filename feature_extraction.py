#!/usr/bin/env python3

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

from julia import Julia
Julia(sysimage='../sys.so')
from interpretableai import iai

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

cols = [c for c in X_train.columns if c.lower()[-2] == '_0' or c.lower()[:3] != 'cat']

X_train = np.array(X_train[cols]) #X_train = X_train[cols]
X_test = np.array(X_test[cols]) #X_test = X_test[cols]
n = tgt_intensity_train.shape[0]
train = np.concatenate((fresh[:n], X_train), axis=1)
test = np.concatenate((fresh[n:], X_test), axis=1)


(train_X, train_y), (test_X, test_y) = iai.split_data('regression', train, tgt_intensity_train, seed=1)

grid = iai.GridSearch(
    iai.OptimalFeatureSelectionRegressor(
        random_seed=1,
    ),
    sparsity=range(50, 500, 50),
)
grid.fit(train, tgt_intensity_train)

y_hat_intensity = grid.predict(test)
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test, y_hat_intensity)*1.852)

numeric_weights, categoric_weights = grid.get_prediction_weights()
numeric_weights


grid = iai.GridSearch(
    iai.OptimalFeatureSelectionRegressor(
        random_seed=1,
    ),
    sparsity=range(110, 150, 10),
)

grid.fit(X_train, tgt_intensity_train)
y_hat_intensity = grid.predict(X_test)
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test, y_hat_intensity)*1.852)

numeric_weights, categoric_weights = grid.get_prediction_weights()

X_train = X_train[numeric_weights.keys()]
X_test = X_test[numeric_weights.keys()]
xgb = XGBRegressor(max_depth=5, n_estimators=150, learning_rate = 0.15)
xgb.fit(X_train, tgt_intensity_train)
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test, xgb.predict(X_test))*1.852)


xgb = XGBClassifier(max_depth=5, n_estimators=150, learning_rate = 0.15)
xgb.fit(X_train, tgt_intensity_cat_train)
print("Accuracy: ", accuracy_score(tgt_intensity_test, xgb.predict(test))*1.852)


#df.Species = df.Species.astype('category')


