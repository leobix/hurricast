from run import Prepro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

#
# import argparse
#
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
# parser.add_argument("--steps-in", type=int, default=48,
#                             help="number of in time steps")
#
# parser.add_argument("--t_list", type=list, default=[1,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48],
                            # help="list of prediction time steps")

def run_models(steps_in, steps_out, max_depth=5):
    train_test_split = 0.8
    predict_at = steps_in #steps_out
    window_size= steps_out #how many timesteps from the past to take ie steps_in

    train_tensors, test_tensors = Prepro.process(vision_data, y, train_test_split, predict_at, window_size)
    x_viz_train, x_stat_train, tgt_intensity_cat_train, tgt_intensity_cat_baseline_train, tgt_displacement_train, tgt_intensity_train = train_tensors
    x_viz_test, x_stat_test, tgt_intensity_cat_test, tgt_intensity_cat_baseline_test, tgt_displacement_test, tgt_intensity_test = test_tensors

    #reshape and concat
    X_train = x_stat_train.reshape(x_stat_train.shape[0], -1)
    X_test = x_stat_test.reshape(x_stat_test.shape[0], -1)
    X_train_vision = x_viz_train.reshape(x_viz_train.shape[0], -1)
    X_test_vision = x_viz_test.reshape(x_viz_test.shape[0], -1)
    X_train_tab_vision = np.concatenate((X_train, X_train_vision), axis = 1)
    X_test_tab_vision = np.concatenate((X_test, X_test_vision), axis = 1)

    #run xgb for intensity
    xgb = XGBClassifier(max_depth, n_estimators=100)
    xgb.fit(X_train, tgt_intensity_cat_train)
    intensity_xgb = xgb.predict(X_test)

    #run random forrest for intensity
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, tgt_intensity_cat_train)
    intensity_rf = rf.predict(X_test)

    #run xbg for displacement x and y
    xgb_x = XGBRegressor(max_depth, n_estimators=100)
    xgb_y = XGBRegressor(max_depth, n_estimators=100)
    xgb_x.fit(X_train, tgt_displacement_train[:,0])
    xgb_y.fit(X_train, tgt_displacement_train[:,1])

    dx_xgb= xgb_x.predict(X_test)
    dy_xgb= xgb_y.predict(X_test)

    # #calculate accuracy score for intensity
    intensity_xgb_score = accuracy_score(tgt_intensity_cat_test, intensity_xgb).round(3)
    intensity_rf_score = accuracy_score(tgt_intensity_cat_test, intensity_rf).round(3)
    intensity_base_score = accuracy_score(tgt_intensity_cat_test, tgt_intensity_cat_baseline_test).round(3)

    #calculate displacement mae
    dx_xgb_mae =  mean_absolute_error(tgt_displacement_test[:,0], dx_xgb).round(3)
    dy_xgb_mae = mean_absolute_error(tgt_displacement_test[:,1], dy_xgb).round(3)

    return intensity_xgb_score, intensity_rf_score, intensity_base_score, dx_xgb_mae, dy_xgb_mae


if __name__ == "__main__":

    #load data:
    #vision_data = np.load('data/vision_data_30_16_120_3years_test2.npy', allow_pickle = True)
    vision_data = np.load('data/vision_data_50_20_60_3years_v2.npy', allow_pickle = True)
    # vision_data = np.load('../../../Volumes/Samsung_T5/vision_data_50_20_90_1980_v3.npy', allow_pickle = True)

    #y = np.load('data/y_30_16_120_3years_test2.npy', allow_pickle = True)
    y = np.load('data/y_50_20_60_3years_v2.npy', allow_pickle = True)
    # y = np.load('../../../Volumes/Samsung_T5/y_50_20_90_1980_v3.npy', allow_pickle = True)

    #set up empty dataframes
    accuracy = pd.DataFrame(columns={})

    #prediction steps
    steps_out_list= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    #max_depth
    max_depth_list = [4,5,6]

    #steps_in
    steps_in_list = [8,12,16]

    for max_depth in max_depth_list:
        for steps_in in steps_in_list:
            for steps_out in steps_out_list:
                #run model
                intensity_xgb_score, intensity_rf_score, intensity_base_score, dx_xgb_mae, dy_xgb_mae = run_models(steps_in, steps_out, max_depth)
                #record accuracy
                accuracy = accuracy.append({'past_n_steps': str(steps_in),
                                                  'pred_n_steps': str(t),
                                                  'max_depth': str(max_depth),
                                                  'xgb_intensity_accu': intensity_xgb_score,
                                                  'rf_intensity_accu': intensity_rf_score,
                                                  'base_intensity_accu': intensity_base_score,
                                                  'dx_xgb_mae':dx_xgb_mae,
                                                  'dy_xgb_mae':dy_xgb_mae}, ignore_index=True)

    #output results df
    accuracy.to_csv('cluster_results/model_accuracy.csv', index=False)
