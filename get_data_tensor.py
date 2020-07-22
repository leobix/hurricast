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
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps_in_list", type=int, nargs="+", default=[8, 12, 16],
                            help="number of in time steps")
parser.add_argument("--reduced_ranks_list", type=int, nargs="+", default=[[4,7,10,10],[3,5,5,5],[3,3,3,3]],
                            help="list of reduced ranks") #past_n_steps * maps * 25 * 25

if __name__ == "__main__":

    args = parser.parse_args()
    steps_in_list = args.steps_in_list
    reduced_ranks_list = args.reduced_ranks_list


    #set up empty dataframes
    compress_error = pd.DataFrame(columns={})
    # compress_error = pd.read_csv('cluster_results/viz_compress_error.csv')


    for steps_in in steps_in_list:
        for reduced_ranks in reduced_ranks_list:
            #get vision data:
            x_viz_train = np.load('data/X_train_vision_1980_50_20_90_w'+str(steps_in)+'.npy',  allow_pickle = True).reshape(-1, steps_in , 9, 25, 25)
            x_viz_test = np.load('data/X_test_vision_1980_50_20_90_w'+str(steps_in)+'.npy', allow_pickle = True).reshape(-1, steps_in, 9, 25, 25)
            #crop out some regions, keep past 8 steps
            x_viz_train = x_viz_train[:,-8:,:,3:20,3:20]
            x_viz_test = x_viz_test[:,-8:,:,3:20,3:20]
            #standardize x viz
            x_viz_train, x_viz_test = utils.standardize_x_viz(x_viz_train,x_viz_test)

            print('compressing vision data for ranks ', reduced_ranks)
            #compress vision data
            x_viz_train_reduce, compress_error_train = utils.viz_reduce(x_viz_train, reduced_ranks)
            x_viz_test_reduce, compress_error_test = utils.viz_reduce(x_viz_test, reduced_ranks)
            print('ranks, compression error=',reduced_ranks, compress_error)
            #save data
            np.save('data/X_train_viz_reduce_1980_50_20_90_w'+str(steps_in)+'_'+utils.ranks_to_str(reduced_ranks)+'.npy', x_viz_train_reduce, allow_pickle=True)
            np.save('data/X_test_viz_reduce_1980_50_20_90_w'+str(steps_in)+'_'+utils.ranks_to_str(reduced_ranks)+'.npy', x_viz_test_reduce, allow_pickle=True)

            compress_error = compress_error.append({'past_n_steps': steps_in,
                                              'reduced_ranks': reduced_ranks,
                                              'compress_error_train': compress_error_train,
                                              'compress_error_test': compress_error_test
                                              }, ignore_index=True)

            compress_error.to_csv('cluster_results/viz_compress_error.csv', index=False)

    #-------- end  ------ #
    #
    # #get training and test tensors
    # x_viz_train = np.load('data/X_train_viz_reduce_1980_50_20_90_w'+str(steps_in)+'_'+utils.ranks_to_str(reduced_ranks)+'.npy', allow_pickle=True)
    # x_viz_test = np.load('data/X_train_viz_reduce_1980_50_20_90_w'+str(steps_in)+'_'+utils.ranks_to_str(reduced_ranks)+'.npy', allow_pickle=True)
    #
    # x_stat_train = np.load('data/X_train_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True).reshape(-1, window_size, 22)[:,:,:7]
    # x_stat_test = np.load('data/X_test_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle=True).reshape(-1, window_size, 22)[:,:,:7]
    #
    # print(x_viz_train.shape)
