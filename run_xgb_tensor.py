from run import Prepro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import utils_tensor as utils #import local function


def run_xgb(vision_data, stat_data, window_size, predict_at, max_depth, reduced_ranks = [5,7,10,10]):
    #get training and test tensors
    ##maybe we don't have to call this for each prediction period, since x is the same, just y different?
    train_test_split=0.8
    train_tensors, test_tensors = Prepro.process(vision_data, stat_data, train_test_split, predict_at = predict_at, window_size=window_size)
    x_viz_train, x_stat_train, tgt_intensity_cat_train, tgt_intensity_cat_baseline_train, tgt_displacement_train, tgt_intensity_train = train_tensors
    x_viz_test, x_stat_test, tgt_intensity_cat_test, tgt_intensity_cat_baseline_test, tgt_displacement_test, tgt_intensity_test = test_tensors

    #concat viz with tabular
    X_train, compress_error =utils.concat_stat_viz(x_stat_train, x_viz_train, reduced_ranks)
    X_test, _ =utils.concat_stat_viz(x_stat_test, x_viz_test, reduced_ranks)

    #run XGB on intensity
    xgb = XGBClassifier(max_depth=max_depth, n_estimators=80)
    xgb.fit(X_train, tgt_intensity_cat_train)
    yhat = xgb.predict(X_test)
    score_xgb = accuracy_score(tgt_intensity_cat_test, yhat)
    score_base = accuracy_score(tgt_intensity_cat_test, tgt_intensity_cat_baseline_test)

    return np.round(score_xgb,3) , np.round(score_base,3), compress_error


if __name__ == "__main__":

    #load data:
    #vision_data = np.load('data/vision_data_30_16_120_3years_test2.npy', allow_pickle = True)
    vision_data = np.load('data/vision_data_50_20_60_3years_v2.npy', allow_pickle = True)
    # vision_data = np.load('../../../Volumes/Samsung_T5/vision_data_50_20_90_1980_v3.npy', allow_pickle = True)

    #stat_data = np.load('data/y_30_16_120_3years_test2.npy', allow_pickle = True)
    stat_data = np.load('data/y_50_20_60_3years_v2.npy', allow_pickle = True)
    # stat_data = np.load('../../../Volumes/Samsung_T5/y_50_20_90_1980_v3.npy', allow_pickle = True)

    #set up empty dataframes
    accuracy = pd.DataFrame(columns={})

    #prediction steps
    steps_out_list= [2,4,6,8,10,12,14,16,18] #[2] #

    #max_depth
    max_depth_list = [4,5,6] #[5]

    #steps_in
    steps_in_list = [8,12,16] #[8]

    for max_depth in max_depth_list:
        for steps_in in steps_in_list:
            for steps_out in steps_out_list:
                #run model
                intensity_xgb_score, intensity_base_score, compress_error= run_xgb(vision_data, stat_data, window_size = steps_in, predict_at = steps_out, max_depth = max_depth, reduced_ranks = [5,8,10,10])
                #record accuracy
                accuracy = accuracy.append({'past_n_steps': str(steps_in),
                                                  'pred_n_steps': str(steps_out),
                                                  'max_depth': str(max_depth),
                                                  'xgb_intensity_accu': intensity_xgb_score,
                                                  'base_intensity_accu': intensity_base_score,
                                                  'compression_error': compress_error}, ignore_index=True) #'dx_xgb_mae':dx_xgb_mae, 'dy_xgb_mae':dy_xgb_mae

    #output results df
    accuracy.to_csv('cluster_results/xgb_tensor_accuracy.csv', index=False)
