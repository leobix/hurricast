#!/usr/bin/env python3

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import  GridSearchCV

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

fresh = pd.DataFrame(np.load('data/test_features_filtered_direct_w' + str(window_size) + '.npy', allow_pickle=True))

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

X_train = X_train[cols]
X_test = X_test[cols]

#cat_cols = [c for c in X_train.columns if c.lower()[:3] == 'cat']

#for col in cat_cols:
    #X_train[col] = X_train[col].astype('category')
    #X_test[col] = X_test[col].astype('category')

n = tgt_intensity_train.shape[0]
#train = pd.concat([fresh[:n], X_train], axis=1)
#test = pd.concat([fresh[n:], X_test], axis=1)
train = pd.DataFrame(np.concatenate((np.array(fresh[:n]), np.array(X_train)), axis = 1))
test = pd.DataFrame(np.concatenate((np.array(fresh[n:]), np.array(X_test)), axis = 1))
train.columns = [str(i) for i in range(train.shape[1])]
test.columns = train.columns


#(train_X, train_y), (test_X, test_y) = iai.split_data('regression', train, tgt_intensity_train, seed=1)

##### FRESH + TABULAR #####
grid = iai.GridSearch(
    iai.OptimalFeatureSelectionRegressor(
        random_seed=1,
    ),
    sparsity=range(600, 1000, 100),
)
grid.fit(train, tgt_intensity_train)

y_hat_intensity = grid.predict(test)
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test, y_hat_intensity)*1.852)

numeric_weights, categoric_weights = grid.get_prediction_weights()


X_train_sparse = train[numeric_weights.keys()]
X_test_sparse = test[numeric_weights.keys()]
xgb = XGBRegressor(max_depth=5, n_estimators=100, learning_rate = 0.08, min_child_weight = 0.5, subsample = 0.6)
xgb.fit(X_train_sparse, tgt_intensity_train)
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test, xgb.predict(X_test_sparse))*1.852)

###### TABULAR REGRESSION ######
grid = iai.GridSearch(
    iai.OptimalFeatureSelectionRegressor(
        random_seed=1,
    ),
    sparsity=range(90, 150, 20),
)

grid.fit(X_train, tgt_intensity_train)
y_hat_intensity = grid.predict(X_test)
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test, y_hat_intensity)*1.852)

numeric_weights, categoric_weights = grid.get_prediction_weights()

X_train_sparse = X_train[numeric_weights.keys()]
X_test_sparse = X_test[numeric_weights.keys()]


xgb = XGBRegressor(max_depth=5, n_estimators=140, learning_rate = 0.15, min_child_weight = 2, subsample = 0.8)
xgb.fit(X_train_sparse, tgt_intensity_train)
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test, xgb.predict(X_test_sparse))*1.852)

xgb = XGBClassifier(max_depth=5, n_estimators=140, learning_rate = 0.15, min_child_weight = 2, subsample = 0.9)
xgb.fit(X_train_sparse, tgt_intensity_cat_train)
print("Accuracy: ", accuracy_score(tgt_intensity_cat_test, xgb.predict(X_test_sparse)))

###### TABULAR CLASSIFICATION ######
grid = iai.GridSearch(
    iai.OptimalFeatureSelectionClassifier(
        random_seed=1,
    ),
    sparsity=range(90, 150, 20),
)

grid.fit(X_train, tgt_intensity_cat_train)

grid.score(X_test, tgt_intensity_cat_test, criterion='misclassification')
#y_hat_intensity = grid.predict(X_test)
#print("MAE intensity: ", mean_absolute_error(tgt_intensity_test, y_hat_intensity)*1.852)

numeric_weights, categoric_weights = grid.get_prediction_weights()

X_train_sparse_class = X_train[numeric_weights.keys()]
X_test_sparse_class = X_test[numeric_weights.keys()]

xgb = XGBClassifier(max_depth=5, n_estimators=140, learning_rate = 0.15, min_child_weight = 2, subsample = 0.9)
xgb.fit(X_train_sparse_class, tgt_intensity_cat_train)
print("Accuracy: ", accuracy_score(tgt_intensity_cat_test, xgb.predict(X_test_sparse_class)))

#64.83
#15.62


##### GRID SEARCH #####

param_test1 = {
 'min_child_weight':range(1,6,2),
 'n_estimators':[100, 140],
 'subsample':[0.6,0.8,1],
 'learning_rate':[0.1, 0.15, 0.2]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.15, n_estimators=140, max_depth=5,
 min_child_weight=1, subsample=0.8, seed=1),
 param_grid = param_test1, n_jobs=4, scoring = 'neg_mean_absolute_error')
gsearch1.fit(X_train_sparse, tgt_intensity_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


