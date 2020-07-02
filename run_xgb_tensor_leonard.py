import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import utils_tensor as utils #import local function


def run_xgb(window_size, max_depth, reduced_ranks = [5,7,10,10]):
    #get training and test tensors

    x_stat_train = np.load('data/X_train_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)
    x_stat_test = np.load('data/X_test_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)

    x_viz_train = np.load('data/X_train_vision_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)
    x_viz_test = np.load('data/X_test_vision_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle = True)

    tgt_intensity_cat_train = np.load('data/y_train_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy',
                                      allow_pickle=True)
    tgt_intensity_cat_test = np.load('data/y_test_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy',
                                     allow_pickle=True)

    # tgt_intensity_cat_baseline_train = np.load('data/y_train_intensity_cat_baseline_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)
    tgt_intensity_cat_baseline_test = np.load(
        'data/y_test_intensity_cat_baseline_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)

    #concat viz with tabular
    X_train, compress_error = utils.concat_stat_viz(x_stat_train, x_viz_train, reduced_ranks)
    X_test, _ = utils.concat_stat_viz(x_stat_test, x_viz_test, reduced_ranks)

    #run XGB on intensity
    xgb = XGBClassifier(max_depth=max_depth, n_estimators=100)
    xgb.fit(X_train, tgt_intensity_cat_train)
    yhat = xgb.predict(X_test)
    score_xgb = accuracy_score(tgt_intensity_cat_test, yhat)
    score_base = accuracy_score(tgt_intensity_cat_test, tgt_intensity_cat_baseline_test)

    return np.round(score_xgb,3) , np.round(score_base,3), compress_error


if __name__ == "__main__":

    #set up empty dataframes
    accuracy = pd.DataFrame(columns={})

    #max_depth
    max_depth_list = [4,5,6]

    #steps_in
    steps_in_list = [8,12,16]

    for max_depth in max_depth_list:
        for steps_in in steps_in_list:
            for steps_out in steps_out_list:
                #run model
                intensity_xgb_score, intensity_base_score = run_xgb(window_size = steps_in, max_depth = max_depth, reduced_ranks = [5,8,10,10])
                #record accuracy
                accuracy = accuracy.append({'past_n_steps': str(steps_in),
                                                  'pred_n_steps': str(steps_out),
                                                  'max_depth': str(max_depth),
                                                  'xgb_intensity_accu': intensity_xgb_score,
                                                  'base_intensity_accu': intensity_base_score,
                                                  'compression_error': compress_error}, ignore_index=True)
                print(accuracy)

    #output results df
    accuracy.to_csv('cluster_results/xgb_tensor_accuracy.csv', index=False)
