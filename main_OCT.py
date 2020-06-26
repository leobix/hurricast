#!/usr/bin/env python3


import argparse
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#from xgboost import XGBClassifier
#from xgboost import XGBRegressor
#from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import mean_absolute_error


##IAI
from julia import Julia
Julia(sysimage='../sys.so')
from interpretableai import iai


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps-in", type=int, default=8,
                            help="choose between 8, 12, 16")

parser.add_argument("--steps-out", type=int, default=8,
                            help="choose between 8")

parser.add_argument("--min-depth", type=int, default=5,
                            help="min depth")

parser.add_argument("--max-depth", type=int, default=9,
                            help="max depth (not included)")

parser.add_argument("--class-criterion", type=str, default='misclassification',
                            help="classification criterion, choose between misclassification, gini and entropy")

parser.add_argument("--filename", type=str, default='v1',
                            help="filename for Trees")

parser.add_argument("--target-intensity-cat", action='store_true',
                            help="intensity cat")

parser.add_argument("--target-displacement", action='store_true',
                            help="displacement")

def main(args):
    window_size = args.steps_in
    X_train = np.load('data/X_train_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle = True)
    X_test = np.load('data/X_test_stat_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle = True)

    #X_train_vision = np.load('data/X_train_vision_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)
    #X_test_vision = np.load('data/X_test_vision_1980_50_20_90_w' + str(window_size) + '.npy', allow_pickle = True)

    tgt_intensity_cat_train = np.load('data/y_train_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)
    tgt_intensity_cat_test = np.load('data/y_test_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)

    tgt_intensity_cat_baseline_train = np.load('data/y_train_intensity_cat_baseline_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)
    tgt_intensity_cat_baseline_test = np.load('data/y_test_intensity_cat_baseline_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)

    tgt_displacement_train = np.load('data/y_train_displacement_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)
    tgt_displacement_test = np.load('data/y_test_displacement_1980_50_20_90_w' + str(window_size) + '.npy',  allow_pickle = True)

    names = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND',
             'STORM_SPEED', 'STORM_DIR', 'storm_category', 'cat_basin_EP', 'cat_basin_NI',
             'cat_basin_SI', 'cat_basin_SP', 'cat_basin_WP', 'cat_nature_DS', 'cat_nature_ET',
             'cat_nature_MX', 'cat_nature_NR', 'cat_nature_SS', 'cat_nature_TS',
             'cat_UNKNOWN',
             'STORM_DISPLACEMENT_X', 'STORM_DISPLACEMENT_Y']
    names_all = names * window_size

    for i in range(len(names_all)):
        names_all[i] += '_' + str(i // 22)

    X_train2 = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    X_train2.columns = names_all
    X_test.columns = names_all

    cols = [c for c in X_train2.columns if c.lower()[-1] == '0' or c.lower()[:3] != 'cat']

    X_train2 = X_train2[cols]
    X_test = X_test[cols]


    grid_scenarios = iai.GridSearch(
            iai.OptimalTreeClassifier(
                random_seed=1,
                criterion = args.class_criterion
            ),
            max_depth=range(args.min_depth, args.max_depth),
        )

    if args.target_intensity_cat:
        y_train = tgt_intensity_cat_train
        y_test = tgt_intensity_cat_test

    elif args.target_displacement:
        y_train = tgt_displacement_train
        y_test = tgt_displacement_test


    grid_scenarios.fit(X_train2, y_train)
    lnr_scenarios = grid_scenarios.get_learner()
    lnr_scenarios.write_html(
        "Trees/" + args.filename + "_Classification_tree_scenarios_in" + str(args.steps_in) + "_out" + str(
            args.steps_out) + ".html")
    print(grid_scenarios.get_grid_results())
    print("Classification based scenarios, Accuracy: ",
          lnr_scenarios.score(X_test, y_test, criterion='misclassification'))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
