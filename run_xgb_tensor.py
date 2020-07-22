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

parser.add_argument("--steps_in_list", type=int, nargs="+", default=[8, 12, 16],
                            help="number of in time steps")
parser.add_argument("--steps_out_list", type=int,  nargs="+", default=[8, 16],
                            help="list of prediction time steps")
parser.add_argument("--reduced_ranks_list", type=int, default= [[4,7,10,10],[3,5,5,5],[3,3,3,3]],
                            help="list of reduced ranks") #past_n_steps * maps * 25 * 25
parser.add_argument("--max_depth_list", type=int, default= [5,6],
                                help="list of max depth for xgb") #past_n_steps * maps * 25 * 25
parser.add_argument("--run_compression", type=int, default= 0,
                                help="boolean argument for running compression, 0 = False, 1=True") #past_n_steps * maps * 25 * 25


def run_xgb(window_size, predict_at, max_depth, reduced_ranks):
    #get training and test tensors
    x_viz_train = np.load('data/X_train_viz_reduce_1980_50_20_90_w'+str(window_size)+'_'+utils.ranks_to_str(reduced_ranks)+'.npy', allow_pickle=True)
    x_viz_test = np.load('data/X_test_viz_reduce_1980_50_20_90_w'+str(window_size)+'_'+utils.ranks_to_str(reduced_ranks)+'.npy', allow_pickle=True)

    x_stat_train = np.load('data/X_train_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True).reshape(-1, window_size, 22)[:,:,:7]
    x_stat_test = np.load('data/X_test_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True).reshape(-1, window_size, 22)[:,:,:7]

    tgt_intensity_train = np.load('data/y_train_intensity_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)
    tgt_intensity_test = np.load('data/y_test_intensity_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)

    tgt_intensity_cat_train = np.load('data/y_train_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)
    tgt_intensity_cat_test = np.load('data/y_test_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)
    tgt_intensity_cat_baseline_train = np.load('data/y_train_intensity_cat_baseline_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)
    tgt_intensity_cat_baseline_test = np.load('data/y_test_intensity_cat_baseline_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)

    tgt_displacement_train = np.load('data/y_train_displacement_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)
    tgt_displacement_test = np.load('data/y_test_displacement_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True)

    #standardize x and y
    x_viz_train, x_viz_test = utils.standardize_x_viz(x_viz_train, x_viz_test)
    x_stat_train, x_stat_test = utils.standardize_x_stat(x_stat_train, x_stat_test)
    tgt_intensity_train,tgt_intensity_test, mean_intensity, std_intensity  = utils.standardize_y(tgt_intensity_train,tgt_intensity_test)
    tgt_dx_train,tgt_dx_test, mean_dx, std_dx  = utils.standardize_y(tgt_displacement_train[:,0],tgt_displacement_test[:,0])
    tgt_dy_train,tgt_dy_test, mean_dy, std_dy  = utils.standardize_y(tgt_displacement_train[:,1],tgt_displacement_test[:,1])

    #concat viz with tabular
    # print(x_stat_train.shape, x_viz_train.shape)
    X_train=utils.concat_stat_viz(x_stat_train, x_viz_train)
    X_test=utils.concat_stat_viz(x_stat_test, x_viz_test)

    #run XGB on intensity
    print('training xgb...')
    xgb = XGBRegressor(max_depth=5, n_estimators=80)
    xgb.fit(X_train, tgt_intensity_train)
    yhat = xgb.predict(X_test)
    mae_intensity = mean_absolute_error(tgt_intensity_test*std_intensity + mean_intensity, yhat*std_intensity+mean_intensity)

    #stat only
    xgb = XGBRegressor(max_depth=5, n_estimators=80)
    #reshape and concat
    X_train_stat = x_stat_train.reshape(x_stat_train.shape[0], -1)
    X_test_stat = x_stat_test.reshape(x_stat_test.shape[0], -1)
    xgb.fit(X_train_stat, tgt_intensity_train)
    yhat = xgb.predict(X_test_stat)
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
    cat_score_xgb = accuracy_score(tgt_intensity_cat_test, yhat)
    cat_score_base = accuracy_score(tgt_intensity_cat_test, tgt_intensity_cat_baseline_test)

    return np.round(mae_intensity,3) , np.round(mae_intensity_stat,3), np.round(mae_dx,3), np.round(mae_dy,3),np.round(cat_score_xgb,3),np.round(cat_score_base,3)




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
    max_depth_list = args.max_depth_list
    #reduced ranks
    reduced_ranks_list = args.reduced_ranks_list


    # #-------- compress vision data: need to do this just once ------ #
    # if args.run_compression == 1:
    #     #get vision data: past 8 steps
    #     x_viz_train = np.load('data/X_train_vision_1980_50_20_90_w8.npy',  allow_pickle = True).reshape(-1, 8 , 9, 25, 25)
    #     x_viz_test = np.load('data/X_test_vision_1980_50_20_90_w8.npy', allow_pickle = True).reshape(-1, 8, 9, 25, 25)
    #
    #     #crop out some regions, keep past 3 steps
    #     x_viz_train = x_viz_train[:,-4:,:,3:20,3:20]
    #     x_viz_test = x_viz_test[:,-4:,:,3:20,3:20]
    #
    #     #standardize x viz
    #     x_viz_train, x_viz_test = utils.standardize_x_viz(x_viz_train,x_viz_test)
    #     for steps_in in steps_in_list:
    #         for reduced_ranks in reduced_ranks_list:
    #             print('compressing vision data for ranks ', reduced_ranks)
    #             #compress vision data
    #             x_viz_train_reduce, compress_error = utils.viz_reduce(x_viz_train, reduced_ranks)
    #             x_viz_test_reduce, _ = utils.viz_reduce(x_viz_test, reduced_ranks)
    #             print('ranks, compression error=',reduced_ranks, compress_error)
    #             #save data
    #             np.save('data/X_train_viz_reduce_1980_50_20_90_w'+str(steps_in)+'_'+utils.ranks_to_str(reduced_ranks)+'.npy', x_viz_train_reduce, allow_pickle=True)
    #             np.save('data/X_test_viz_reduce_1980_50_20_90_w'+str(steps_in)+'_'+utils.ranks_to_str(reduced_ranks)+'.npy', x_viz_test_reduce, allow_pickle=True)
    #-------- end  ------ #

    for steps_in in steps_in_list:
        for steps_out in steps_out_list:
            for reduced_ranks in reduced_ranks_list:
                for max_depth in max_depth_list:
                    print('running steps_in, steps_out, reduced_ranks, max_depth:', steps_in, steps_out,reduced_ranks, max_depth)
                    #run model
                    mae_intensity, mae_intensity_stat, mae_dx, mae_dy, cat_score_xgb, cat_score_base= run_xgb(window_size = steps_in, predict_at = steps_out, max_depth = max_depth, reduced_ranks = reduced_ranks)
                    print('mae_intensity,mae_intensity_stat, mae_dx, mae_dy',mae_intensity,mae_intensity_stat, mae_dx, mae_dy)
                    #record accuracy
                    accuracy = accuracy.append({'past_n_steps': str(steps_in),
                                                      'pred_n_steps': str(steps_out),
                                                      'max_depth': str(max_depth),
                                                      'reduced_ranks': reduced_ranks,
                                                      'mae_intensity_stat_viz': mae_intensity,
                                                      'mae_intensity_stat': mae_intensity_stat,
                                                      'mae_dx':mae_dx,
                                                      'mae_dy': mae_dy,
                                                      'intensity_cat_accu': cat_score_xgb,
                                                      'intensity_cat_accu_base': cat_score_base
                                                      }, ignore_index=True)


    #output results df
    accuracy.to_csv('cluster_results/xgb_tensor_accuracy.csv', index=False)
