import sys
sys.path.append('../')
from run import Prepro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from utils import utils_tensor as utils #import local functions
import torch


import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps_in_list", type=list, default=[8,12,16],
                            help="number of in time steps")
parser.add_argument("--steps_out_list", type=list, default=[8, 16],
                            help="list of prediction time steps")
parser.add_argument("--reduced_ranks_list", type=list, default= [[5,7,5,5],[10,7,10,10],[10,7,10,10]]
                            help="list of reduced ranks") #past_n_steps * maps * 25 * 25

def run_xgb(window_size, predict_at, max_depth, reduced_ranks):
    #get training and test tensors
    ##maybe we don't have to call this for each prediction period, since x is the same, just y different?
    train_test_split=0.8

    train_tensors, test_tensors = Prepro.process(vision_data, stat_data, train_test_split, predict_at = predict_at, window_size=window_size)
    x_viz_train, x_stat_train, tgt_intensity_cat_train, tgt_intensity_cat_baseline_train, tgt_displacement_train, tgt_intensity_train = train_tensors
    x_viz_test, x_stat_test, tgt_intensity_cat_test, tgt_intensity_cat_baseline_test, tgt_displacement_test, tgt_intensity_test = test_tensors
    
    x_viz_test = x_viz_test.numpy()
    x_stat_test = x_stat_test.numpy()
    tgt_intensity_cat_test = tgt_intensity_cat_test.numpy()
    tgt_intensity_cat_baseline_test=tgt_intensity_cat_baseline_test.numpy()
    tgt_displacement_test = tgt_displacement_test.numpy()
    tgt_intensity_test = tgt_intensity_test.numpy()
    x_viz_train = x_viz_train.numpy()
    x_stat_train = x_stat_train.numpy()
    tgt_intensity_cat_train = tgt_intensity_cat_train.numpy()
    tgt_intensity_cat_baseline_train=tgt_intensity_cat_baseline_train.numpy()
    tgt_displacement_train = tgt_displacement_train.numpy()
    tgt_intensity_train = tgt_intensity_train.numpy()

    #standardize x and y
    x_viz_train, x_viz_test, x_stat_train, x_stat_test = standardize_x(x_viz_train, x_viz_test, x_stat_train, x_stat_test)
    tgt_intensity_train,tgt_intensity_test, mean_intensity, std_intensity  = standardize_y(tgt_intensity_train,tgt_intensity_test)
    tgt_dx_train,tgt_dx_test, mean_dx, std_dx  = standardize_y(tgt_displacement_train[:,0],tgt_displacement_test[:,0])
    tgt_dy_train,tgt_dy_test, mean_dy, std_dy  = standardize_y(tgt_displacement_train[:,1],tgt_displacement_test[:,1])

    print('reshaping vision data to reduced dimensions...', reduced_ranks)
    #concat viz with tabular
    ##can i save this reduced vision data as .py and don't have to reload everytime?
    X_train, compress_error =utils.concat_stat_viz(x_stat_train, x_viz_train, reduced_ranks)
    X_test, _ =utils.concat_stat_viz(x_stat_test, x_viz_test, reduced_ranks)
    #run XGB on intensity
    print('training xgb on intensity...')
    xgb = XGBRegressor(max_depth=5, n_estimators=80)
    xgb.fit(X_train, tgt_intensity_train)
    yhat = xgb.predict(X_test)
    mae_intensity = mean_absolute_error(tgt_intensity_test*std_intensity + mean_intensity, yhat*std_intensity+mean_intensity)
    #stat only
    xgb = XGBRegressor(max_depth=5, n_estimators=80)
    xgb.fit(x_stat_train, tgt_intensity_train)
    yhat = xgb.predict(x_stat_test)
    mae_intensity_stat = mean_absolute_error(tgt_intensity_test*std_intensity + mean_intensity, yhat*std_intensity+mean_intensity)


    #run XGB on displacement
    xgb = XGBRegressor(max_depth=5, n_estimators=80)
    xgb.fit(X_train, tgt_dx_train)
    yhat = xgb.predict(X_test)
    mae_dx = mean_absolute_error(tgt_dx_test*std_dx + mean_dx, yhat*std_dx+mean_dx)

    xgb = XGBRegressor(max_depth=5, n_estimators=80)
    xgb.fit(X_train, tgt_dy_train)
    yhat = xgb.predict(X_test)
    mae_dy = mean_absolute_error(tgt_dy_test*std_dx + mean_dy, yhat*std_dy+mean_dy)

    #run XGB on intensity category
    xgb = XGBClassifier(max_depth=max_depth, n_estimators=80)
    xgb.fit(X_train, tgt_intensity_cat_train)
    yhat = xgb.predict(X_test)
    score_xgb = accuracy_score(tgt_intensity_cat_test, yhat)
    score_base = accuracy_score(tgt_intensity_cat_test, tgt_intensity_cat_baseline_test)

    return np.round(mae_intensity,3) , np.round(mae_dx,3), np.round(mae_dy,3), np.round(score_xgb,3), np.round(score_base,3), compress_error


#standardize x:
def standardize_x(x_viz_train, x_viz_test, x_stat_train, x_stat_test):
    means = x_viz_train.mean(axis=(0, 1, 3, 4))
    stds = x_viz_train.std(axis=(0, 1, 3, 4))

    means_stat = x_stat_train.mean(axis=(0, 1))
    stds_stat = x_stat_train.std(axis=(0, 1))

    for i in range(len(means)):
        x_viz_train[:, :, i] = (x_viz_train[:, :, i] - means[i]) / stds[i]
        x_viz_test[:, :, i] = (x_viz_test[:, :, i] - means[i]) / stds[i]

    for i in range(len(means_stat)):
        x_stat_train[:, :, i] = (x_stat_train[:, :, i] - means_stat[i]) / stds_stat[i]
        x_stat_test[:, :, i] = (x_stat_test[:, :, i] - means_stat[i]) / stds_stat[i]
    return x_viz_train, x_viz_test, x_stat_train, x_stat_test

#standardize y:
def standardize_y(y_train, y_test):
    y_train = y_train
    y_test = y_test
    mean = y_train.mean()
    std = y_train.std()
    y_train = (y_train-mean)/std
    y_test = (y_test-mean)/std
    return y_train, y_test, mean , std

if __name__ == "__main__":

    #load data:
    #vision_data = np.load('data/vision_data_30_16_120_3years_test2.npy', allow_pickle = True)
    # vision_data = np.load('data/vision_data_50_20_60_3years_v2.npy', allow_pickle = True)
    # vision_data = np.load('../../../Volumes/Samsung_T5/vision_data_50_20_90_1980_v3.npy', allow_pickle = True)

    #stat_data = np.load('data/y_30_16_120_3years_test2.npy', allow_pickle = True)
    # stat_data = np.load('data/y_50_20_60_3years_v2.npy', allow_pickle = True)
    # stat_data = np.load('../../../Volumes/Samsung_T5/y_50_20_90_1980_v3.npy', allow_pickle = True)

    #set up empty dataframes
    accuracy = pd.DataFrame(columns={})

    args = parser.parse_args()
    #steps_in
    steps_in_list= args.steps_in_list
    #prediction steps
    steps_out_list= args.steps_out_list
    #max_depth
    max_depth =5
    #reduced ranks
    reduced_ranks_list = args.reduced_ranks_list
    for steps_in in steps_in_list:
        for steps_out in steps_out_list:
            for reduced_ranks in reduced_ranks_list:
                print('running max_depth, steps_in, steps_out:', max_depth, steps_in, steps_out)
                #run model
                mae_intensity, mae_dx, mae_dy, cat_score, cat_score_base, compress_error= run_xgb(window_size = steps_in, predict_at = steps_out, max_depth = max_depth, reduced_ranks = reduced_ranks)
                print('mae_intensity, mae_dx, mae_dy, cat_score=:',mae_intensity, mae_dx, mae_dy, cat_score)
                #record accuracy
                accuracy = accuracy.append({'past_n_steps': str(steps_in),
                                                  'pred_n_steps': str(steps_out),
                                                  'max_depth': str(max_depth),
                                                  'reduced_ranks': reduced_ranks,
                                                  'mae_intensity': mae_intensity,
                                                  'mae_dx':mae_dx,
                                                  'mae_dy': mae_dy,
                                                  'intensity_cat_accu': cat_score,
                                                  'intensity_cat_accu_base': cat_score_base,
                                                  'compression_error': compress_error}, ignore_index=True) #'dx_xgb_mae':dx_xgb_mae, 'dy_xgb_mae':dy_xgb_mae

    #output results df
    accuracy.to_csv('cluster_results/xgb_tensor_accuracy.csv', index=False)
