import sys
sys.path.append("../")
from src import prepro, metrics, run, setup
import src.models.factory as model_factory
import config
import torch

import numpy as np
from src.utils import models
import os.path as osp
import pandas as pd

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from src.utils.data_processing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = setup.create_setup()
args.mode = 'displacement'
args.full_encoder = True
args.encoder_config = 'full_encoder_config'
args.decoder_config = 'lstm_config'
args.window_size = 8
args.predict_at = 8
args.target_intensity_cat = False
#args.sub_window_size = 8
#args.sub_area = 1
args.output_dir = '../results/results8_16_20_44_28' #''../results/results8_16_18_57_7' best so far
#try: ../results/results8_16_20_44_28, 0.086 from training
#Reached 70.4 and the best results with combination. 7.47 GRU outputs 64. All rest normal.
#args.output_dir = './results/results7_20_17_12_19' #Be careful to change 2304 for intermediary layer
#args.output_dir = './results/results7_20_15_4_36' : best for acc 70.5 and nearly best for intensiy 7.50
#args. = ./results/results8_12_19_5_42 for best perf t+48h

#if args.sub_area > 0:
    #x_viz_train = x_viz_train[:, :, :, args.sub_area:-args.sub_area, args.sub_area:-args.sub_area]
    #x_viz_test = x_viz_test[:, :, :, args.sub_area:-args.sub_area, args.sub_area:-args.sub_area]

tgt_intensity_cat_train = torch.LongTensor(np.load('../data/y_train_intensity_cat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                      allow_pickle=True))
tgt_intensity_cat_test = torch.LongTensor(np.load('../data/y_test_intensity_cat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                     allow_pickle=True))

tgt_intensity_train = torch.Tensor(np.load('../data/y_train_intensity_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                  allow_pickle=True))
tgt_intensity_test = torch.Tensor(np.load('../data/y_test_intensity_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                 allow_pickle=True))

tgt_intensity_cat_baseline_train = torch.LongTensor(np.load('../data/y_train_intensity_cat_baseline_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',  allow_pickle = True))
tgt_intensity_cat_baseline_test = torch.LongTensor(np.load('../data/y_test_intensity_cat_baseline_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True))

tgt_displacement_train = torch.Tensor(np.load('../data/y_train_displacement_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                     allow_pickle=True))
tgt_displacement_test = torch.Tensor(np.load('../data/y_test_displacement_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                    allow_pickle=True))

tgt_displacement_train_unst = torch.Tensor(np.load('../data/y_train_displacement_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                     allow_pickle=True))
tgt_displacement_test_unst = torch.Tensor(np.load('../data/y_test_displacement_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                    allow_pickle=True))


mean_intensity = tgt_intensity_train.mean()
std_intensity = tgt_intensity_train.std()
tgt_intensity_train = (tgt_intensity_train - mean_intensity)/std_intensity
tgt_intensity_test = (tgt_intensity_test - mean_intensity)/std_intensity


###INTENSITY
mean_dx = tgt_displacement_train[:,0].mean()
std_dx = tgt_displacement_train[:,0].std()
tgt_displacement_train[:,0] = (tgt_displacement_train[:,0] - mean_dx)/std_dx
tgt_displacement_test[:,0] = (tgt_displacement_test[:,0] - mean_dx)/std_dx
std_dx = float(std_dx)
mean_dx = float(mean_dx)

mean_dy = tgt_displacement_train[:,1].mean()
std_dy = tgt_displacement_train[:,1].std()
tgt_displacement_train[:,1] = (tgt_displacement_train[:,1] - mean_dy)/std_dy
tgt_displacement_test[:,1] = (tgt_displacement_test[:,1] - mean_dy)/std_dy
std_dy = float(std_dy)
mean_dy = float(mean_dy)

def standardize(tgt_displacement_train, tgt_displacement_test):
    mean_dx = tgt_displacement_train[:, 0].mean()
    std_dx = tgt_displacement_train[:, 0].std()
    tgt_displacement_train[:, 0] = (tgt_displacement_train[:, 0] - mean_dx) / std_dx
    tgt_displacement_test[:, 0] = (tgt_displacement_test[:, 0] - mean_dx) / std_dx
    std_dx = float(std_dx)
    mean_dx = float(mean_dx)
    mean_dy = tgt_displacement_train[:, 1].mean()
    std_dy = tgt_displacement_train[:, 1].std()
    tgt_displacement_train[:, 1] = (tgt_displacement_train[:, 1] - mean_dy) / std_dy
    tgt_displacement_test[:, 1] = (tgt_displacement_test[:, 1] - mean_dy) / std_dy
    std_dy = float(std_dy)
    mean_dy = float(mean_dy)
    return tgt_displacement_train, tgt_displacement_test, std_dx, mean_dx, std_dy, mean_dy

def unstandardize(tgt_displacement_train, tgt_displacement_test, std_dx, mean_dx, std_dy, mean_dy):
    tgt_displacement_train[:, 0] = tgt_displacement_train[:, 0] *  std_dx + mean_dx
    tgt_displacement_test[:, 0] = tgt_displacement_test[:, 0] *  std_dx + mean_dx
    tgt_displacement_train[:, 1] = tgt_displacement_train[:, 1] * std_dy + mean_dy
    tgt_displacement_test[:, 1] = tgt_displacement_test[:, 1] * std_dy + mean_dy
    return tgt_displacement_train, tgt_displacement_test


###### LOADING MODELS AND GETTING EMBEDDINGS
############################################

x_stat_train = torch.Tensor(np.load('../data/X_train_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True).reshape(-1, args.window_size, 30)[:,:,:14]).to(device)
x_stat_test = torch.Tensor(np.load('../data/X_test_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True).reshape(-1, args.window_size, 30)[:,:,:14]).to(device)


x_viz_train = torch.Tensor(np.load('../data/X_train_vision_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',  allow_pickle = True).reshape(-1, args.window_size, 9, 25, 25)).to(device)
x_viz_test = torch.Tensor(np.load('../data/X_test_vision_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle = True).reshape(-1, args.window_size, 9, 25, 25)).to(device)

means = x_viz_train.mean(dim=(0, 1, 3, 4))
stds = x_viz_train.std(dim=(0, 1, 3, 4))


for i in range(len(means)):
            x_viz_train[:, :, i] = (x_viz_train[:, :, i] - means[i]) / stds[i]
            x_viz_test[:, :, i] = (x_viz_test[:, :, i] - means[i]) / stds[i]


means_stat = x_stat_train[:,:,:6].mean(dim=(0, 1))
stds_stat = x_stat_train[:,:,:6].std(dim=(0, 1))

for i in range(len(means_stat)):
            x_stat_train[:, :, i] = (x_stat_train[:, :, i] - means_stat[i]) / stds_stat[i]
            x_stat_test[:, :, i] = (x_stat_test[:, :, i] - means_stat[i]) / stds_stat[i]


modes = {#Modes and associated tasks
    'intensity': 'regression',
    'displacement': 'regression',
    'intensity_cat': 'classification',
    'baseline_intensity_cat': 'classification',
    'baseline_displacement': 'regression'
}


task = modes[args.mode]
print('MODE AND TASK: {} | {}'.format(args.mode, task))

encoder_conf = config.create_config(args.encoder_config)
decoder_conf = config.create_config(args.decoder_config)

model = model_factory.create_model(
        mode=args.mode, 
        encoder_config=encoder_conf,
        decoder_config=decoder_conf, 
        args=args)

model.load_state_dict(torch.load(osp.join(args.output_dir, 'model-best.pt'), map_location=torch.device('cpu')))
model.eval()


embeds_train1 = np.array(model.get_embeddings(x_stat_train[:30000], x_viz_train[:30000]).reshape(-1, 128).detach().numpy())
embeds_train2 = np.array(model.get_embeddings(x_stat_train[30000:55000], x_viz_train[30000:55000]).reshape(-1, 128).detach().numpy())
embeds_train3 = np.array(model.get_embeddings(x_stat_train[55000:80000], x_viz_train[55000:80000]).reshape(-1, 128).detach().numpy())
embeds_train4 = np.array(model.get_embeddings(x_stat_train[80000:], x_viz_train[80000:]).reshape(-1, 128).detach().numpy())
X_test_embed = np.array(model.get_embeddings(x_stat_test, x_viz_test).reshape(-1, 128).detach().numpy())
X_train_embed = np.concatenate((embeds_train1, embeds_train2, embeds_train3, embeds_train4))

##########
##########
########## PREPARING DATA FOR XGB

X_train = np.load('../data/X_train_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
            allow_pickle=True)
X_test = np.load('../data/X_test_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
            allow_pickle=True)


names = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND',
         'STORM_SPEED', 'cat_cos_day', 'cat_sign_day', 'COS_STORM_DIR', 'SIN_STORM_DIR',
         'COS_LAT', 'SIN_LAT', 'COS_LON', 'SIN_LON', 'cat_storm_category', 'cat_basin_AN',
         'cat_basin_EP', 'cat_basin_NI', 'cat_basin_SA',
         'cat_basin_SI', 'cat_basin_SP', 'cat_basin_WP', 'cat_nature_DS', 'cat_nature_ET',
         'cat_nature_MX', 'cat_nature_NR', 'cat_nature_SS', 'cat_nature_TS',
         'STORM_DISPLACEMENT_X', 'STORM_DISPLACEMENT_Y']

names_all = names * args.window_size

for i in range(len(names_all)):
    names_all[i] += '_' + str(i // 30)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train.columns = names_all
X_test.columns = names_all

cols = [c for c in X_train.columns if c.lower()[-2:] == '_0' or c.lower()[:3] != 'cat']

X_train = X_train[cols]
X_test = X_test[cols]


X_train_embed = np.load('../data/embeddings/X_train_embeds_1980_34_20_120_results8_16_20_44_28.npy', allow_pickle = True)
X_test_embed = np.load('../data/embeddings/X_test_embeds_1980_34_20_120_results8_16_20_44_28.npy', allow_pickle = True)
X_train_total = np.concatenate((X_train, X_train_embed), axis = 1)
X_test_total = np.concatenate((X_test, X_test_embed), axis = 1)

std_ = float(std_intensity)
mean_ = float(mean_intensity)

xgb2 = XGBRegressor(max_depth=6, n_estimators=140, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb2.fit(X_train, tgt_intensity_train)
print("MAE intensity: ", mean_absolute_error(np.array(tgt_intensity_test)*std_+mean_, np.array(xgb2.predict(X_test))*std_+mean_))

xgb = XGBRegressor(max_depth=8, n_estimators = 150, learning_rate = 0.07, subsample = 0.9)
xgb.fit(X_train_total, tgt_intensity_train)
print("MAE intensity: ", mean_absolute_error(np.array(tgt_intensity_test)*std_+mean_, np.array(xgb.predict(X_test_total))*std_+mean_))

xgb3 = XGBClassifier(max_depth=6, n_estimators=140, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb3.fit(X_train, tgt_intensity_cat_train)
print("Accuracy: ", accuracy_score(tgt_intensity_cat_test, xgb3.predict(X_test)))

xgb3 = XGBClassifier(max_depth=8, n_estimators=140, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb3.fit(X_train_total, tgt_intensity_cat_train)
print("Accuracy: ", accuracy_score(tgt_intensity_cat_test, xgb3.predict(X_test_total)))


####LOADING X_STAT WITH BASELINES

X_test_baseline = pd.DataFrame(np.load('../data/X_test_stat_1980_34_20_120_withforecast_2661_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True))

X_test_dates = pd.DataFrame(np.load('../data/X_test_stat_with_dates_columns_1980_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True)[:,:4])

X_test_dates.columns = ['YEAR', 'MONTH', 'DAY', 'HOUR']

names_baselines = [#'SID',
         'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND',
         'STORM_SPEED', 'cat_cos_day', 'cat_sign_day', 'COS_STORM_DIR', 'SIN_STORM_DIR',
         'COS_LAT', 'SIN_LAT', 'COS_LON', 'SIN_LON', 'cat_storm_category',
         'EMXI_24_lat', 'EMXI_24_lon', 'EMXI_24_vmax', 'EMXI_24_mslp',
         'SHIP_24_lat', 'SHIP_24_lon', 'SHIP_24_vmax', 'SHIP_24_mslp',
         'AP01_24_lat', 'AP01_24_lon', 'AP01_24_vmax', 'AP01_24_mslp',
         'CMC_24_lat', 'CMC_24_lon', 'CMC_24_vmax', 'CMC_24_mslp',
         'NAM_24_lat', 'NAM_24_lon', 'NAM_24_vmax', 'NAM_24_mslp',
         'HWRF_24_lat', 'HWRF_24_lon', 'HWRF_24_vmax', 'HWRF_24_mslp',
         'cat_basin_AN', 'cat_basin_EP', 'cat_basin_NI', 'cat_basin_SA',
         'cat_basin_SI', 'cat_basin_SP', 'cat_basin_WP',
         #'cat_nature_DS', 'cat_nature_ET',
         #'cat_nature_MX', 'cat_nature_NR', 'cat_nature_SS', 'cat_nature_TS',
         'STORM_DISPLACEMENT_X', 'STORM_DISPLACEMENT_Y']

names_all_baselines = names_baselines * 8#args.window_size

for i in range(len(names_all_baselines)):
    names_all_baselines[i] += '_' + str(i // 48)

X_test_baseline.columns = names_all_baselines

X_test_baseline = pd.concat([X_test_baseline, X_test_dates], axis = 1)


#### SPARSE LOADING

import pickle

X_train_total_ = pd.DataFrame(X_train_total)
X_test_total_ = pd.DataFrame(X_test_total)

col_names = ["col_" + str(i) for i in range(X_train_total.shape[1])]
X_train_total_.columns = col_names
X_test_total_.columns = col_names

numeric_weights_x = pickle.load(open("sparse_tot_x.pkl", "rb"))

X_train_total_sparse_x = X_train_total_[numeric_weights_x.keys()]
X_test_total_sparse_x = X_test_total_[numeric_weights_x.keys()]

numeric_weights_y = pickle.load(open("sparse_tot_y.pkl", "rb"))

X_train_total_sparse_y = X_train_total_[numeric_weights_y.keys()]
X_test_total_sparse_y = X_test_total_[numeric_weights_y.keys()]


####BASELINES GEOLOC



def train_xgb_track(last_storms = 1000, basin_only = False, sparse = False, max_depth = 8, n_estimators = 140, learning_rate = 0.15, subsample = 0.7, min_child_weight=5, basin = 'AN', forecast = 'HWRF'):
    train_x = X_train_total
    train_y = X_train_total
    test_x = X_test_total
    test_y = X_test_total
    tgt_train = tgt_displacement_train
    if sparse:
        train_x, train_y = X_train_total_sparse_x, X_train_total_sparse_y
        test_x, test_y = X_test_total_sparse_x, X_test_total_sparse_y
    if basin_only:
        train_x = X_train_total[X_train['cat_basin_'+basin+'_0'] == 1]
        train_y = train_x
        tgt_train = tgt_displacement_train[X_train['cat_basin_'+basin+'_0'] == 1]
    xgb_x = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, min_child_weight=min_child_weight)
    xgb_x.fit(train_x, tgt_train[:, 0])
    xgb_y = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, min_child_weight=min_child_weight)
    xgb_y.fit(train_y, tgt_train[:, 1])
    DLATS_PRED = np.array(xgb_x.predict(test_x)) * std_dx + mean_dx
    DLONS_PRED = np.array(xgb_y.predict(test_y)) * std_dy + mean_dy
    LATS_PRED_ = X_test['LAT_7'] + DLATS_PRED
    LONS_PRED_ = X_test['LON_7'] + DLONS_PRED
    compare_perf_track(basin=basin, forecast=forecast, mode='lat', LATS_PRED_=LATS_PRED_, LONS_PRED_=LONS_PRED_, last_storms = last_storms)



def train_xgb_intensity(last_storms = 1000, basin_only = False, sparse = False, max_depth = 8, n_estimators = 140, learning_rate = 0.15, subsample = 0.7, min_child_weight=5, basin = 'AN', forecast = 'HWRF'):
    train = X_train_total
    test = X_test_total
    tgt_train = tgt_intensity_train
    if sparse:
        train = X_train_total_sparse_x
        test = X_test_total_sparse_x
    if basin_only:
        train = X_train_total[X_train['cat_basin_'+basin+'_0'] == 1]
        tgt_train = tgt_intensity_train[X_train['cat_basin_'+basin+'_0'] == 1]
    xgb_total = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, min_child_weight=min_child_weight)
    xgb_total.fit(train, tgt_train)
    compare_perf_intensity(xgb_total = xgb_total, basin=basin, forecast=forecast, mode='vmax', last_storms = last_storms)

#train_xgb_track(n_estimators = 90, max_depth = 7, learning_rate = 0.12, subsample = 0.7, min_child_weight = 5, basin = 'AN', forecast = 'HWRF') 82.14 and 117.26
#train_xgb_track(n_estimators = 90, max_depth = 7, learning_rate = 0.1, subsample = 0.7, min_child_weight = 5, basin = 'EP', forecast = 'HWRF')

LATS_TEST = X_test['LAT_7'] + np.array(tgt_displacement_test[:,0])*std_dx+mean_dx
LONS_TEST = X_test['LON_7'] + np.array(tgt_displacement_test[:,1])*std_dy+mean_dy

def compare_perf_track(LATS_PRED_, LONS_PRED_, basin = 'AN', forecast = 'HWRF', mode = 'lat', last_storms = 1000):
    index = X_test_baseline.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1].index#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0].index
    baseline_ = X_test_baseline.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
    LATS_TEST_ = X_test['LAT_7'] + np.array(tgt_displacement_test[:, 0])*std_dx+mean_dx
    LONS_TEST_ = X_test['LON_7'] + np.array(tgt_displacement_test[:, 1])*std_dy+mean_dy
    baseline_1 = baseline_[forecast + '_24_'+mode+'_7']
    if mode == 'lat':
        baseline_2 = baseline_[forecast + '_24_lon_7']
        LATS_BASE = np.array(baseline_1)
        LONS_BASE = np.array(baseline_2)
        LATS_TEST_ = np.array(LATS_TEST_[index])
        LONS_TEST_ = np.array(LONS_TEST_[index])
        LATS_PRED_ = np.array(LATS_PRED_[index])
        LONS_PRED_ = np.array(LONS_PRED_[index])
        d_km_baseline = np.zeros(len(LATS_BASE))
        print(LATS_TEST_)
        print(LATS_PRED_)
        for i in range(len(LATS_BASE)):
            d_km_baseline[i] = get_distance_km(LONS_BASE[i], LATS_BASE[i], LONS_TEST_[i], LATS_TEST_[i])
        print("MAE Distance basin " + basin + " with " + forecast + ": ", d_km_baseline.mean(), "and std: ", d_km_baseline.std())
        print("MAE Distance basin " + basin + " with " + forecast + ": last timesteps ", last_storms, ": ", d_km_baseline[-last_storms:].mean(), "and std: ", d_km_baseline[-last_storms:].std())
        print("Number of busts > 200km", sum(d_km_baseline > 200))
        d_km_pred = np.zeros(len(LONS_PRED_))
        for i in range(len(LONS_PRED_)):
            d_km_pred[i] = get_distance_km(LONS_PRED_[i], LATS_PRED_[i], LONS_TEST_[i], LATS_TEST_[i])
        print("MAE Distance basin " + basin + " Hurricast: ", d_km_pred.mean(), "and std: ", d_km_pred.std())
        print("MAE Distance basin " + basin + " Hurricast last timesteps ", last_storms, ": ", d_km_pred[-last_storms:].mean(), "and std: ", d_km_pred[-last_storms:].std())
        print("Number of busts > 200km", sum(d_km_pred > 200))



def compare_perf_intensity(xgb_total, basin = 'AN', forecast = 'HWRF', last_storms = 1000, mode = 'vmax'):
    index = X_test_baseline.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1].index#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0].index
    #X_test_withBASELINE = X_test.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
    X_test_withBASELINE_total = X_test_total[index]
    baseline_ = X_test_baseline.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
    baseline_1 = baseline_[forecast + '_24_'+mode+'_7']
    if mode == 'vmax':
        tgt_ = tgt_intensity_test[index] * std_ + mean_
        preds = xgb_total.predict(X_test_withBASELINE_total) * std_ + mean_
        #print("MAE intensity basin " + basin + " X stat vs "+ forecast + " : ", mean_absolute_error(tgt_intensity_test_withBASELINE * std_ + mean_,
                                                     #xgb.predict(X_test_withBASELINE) * std_ + mean_))
        print("MAE intensity basin " + basin + " Hurricast : ", mean_absolute_error(tgt_, preds))
        print("MAE intensity basin " + basin + " Official Forecast "+ forecast + " : ",
              mean_absolute_error(tgt_, baseline_1))
        print("\n MAE intensity basin " + basin + " Hurricast last", last_storms, ": ", mean_absolute_error(tgt_[-last_storms:], preds[-last_storms:]))
        print("MAE intensity basin " + basin + " Official Forecast " + forecast + " : ",
              mean_absolute_error(tgt_[-last_storms:], baseline_1[-last_storms:]))


def compare_perf_intensity(xgb_total, basin = 'AN', forecast = 'HWRF', last_storms = 1000, mode = 'vmax'):
    index = X_test_baseline.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1].index#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0].index
    #X_test_withBASELINE = X_test.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
    X_test_withBASELINE_total = X_test_total[index]
    baseline_ = X_test_baseline.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
    baseline_1 = baseline_[forecast + '_24_'+mode+'_7']
    if mode == 'vmax':
        tgt_ = np.array(tgt_intensity_test[index] * std_ + mean_)
        preds = xgb_total.predict(X_test_withBASELINE_total) * std_ + mean_
        #print("MAE intensity basin " + basin + " X stat vs "+ forecast + " : ", mean_absolute_error(tgt_intensity_test_withBASELINE * std_ + mean_,
                                                     #xgb.predict(X_test_withBASELINE) * std_ + mean_))
        print("MAE intensity basin " + basin + " Hurricast : ", np.around(mean_absolute_error(tgt_, preds), decimals = 2), "with std ", np.around(np.std(tgt_ - preds), decimals=2))
        print("MAE intensity basin " + basin + " Official Forecast "+ forecast + " : ",
              np.around(mean_absolute_error(tgt_, baseline_1), decimals = 2), "with std ", np.around(np.std(tgt_ - baseline_1), decimals = 2))
        print("Percentage of missed intensification > 20kn Hurricast: ", np.around(sum(tgt_ - preds > 20)/len(preds) * 100, decimals = 2))
        print("Percentage of missed intensification > 20kn Official Forecast: ", np.around(sum(tgt_ - baseline_1 > 20) / len(baseline_1) * 100, decimals =2))
        print("\nMAE intensity basin " + basin + " Hurricast last", last_storms, ": ", np.around(mean_absolute_error(tgt_[-last_storms:], preds[-last_storms:]), decimals=2))
        print("MAE intensity basin " + basin + " Official Forecast " + forecast + " : ",
              np.around(mean_absolute_error(tgt_[-last_storms:], baseline_1[-last_storms:]), decimals=2))


train_xgb_intensity(forecast = 'SHIP', basin = 'EP', max_depth=8, n_estimators = 120, learning_rate = 0.07, subsample = 0.8, min_child_weight = 1)


def compare_perf_intensity_per_year(xgb_total, year, forecast2, basin = 'AN', forecast = 'HWRF', mode = 'vmax'):
    if forecast2 != None:
        index = X_test_baseline.loc[X_test_baseline['YEAR'] == year].loc[
            X_test_baseline[forecast2 + '_24_' + mode + '_7'] > -320].loc[
            X_test_baseline[forecast + '_24_' + mode + '_7'] > -320].loc[X_test_baseline['cat_basin_' + basin + '_0'] == 1].index  # .loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0].index
        # X_test_withBASELINE = X_test.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
        baseline_ = X_test_baseline.loc[X_test_baseline['YEAR'] == year].loc[
            X_test_baseline[forecast2 + '_24_' + mode + '_7'] > -320].loc[
            X_test_baseline[forecast + '_24_' + mode + '_7'] > -320].loc[
            X_test_baseline['cat_basin_' + basin + '_0'] == 1]  # .loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
        baseline_2 = baseline_[forecast2 + '_24_' + mode + '_7']
    else:
        index = X_test_baseline.loc[X_test_baseline['YEAR'] == year].loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1].index#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0].index
        #X_test_withBASELINE = X_test.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
        baseline_ = X_test_baseline.loc[X_test_baseline['YEAR'] == year].loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
    X_test_withBASELINE_total = X_test_total[index]
    baseline_1 = baseline_[forecast + '_24_'+mode+'_7']
    if mode == 'vmax':
        tgt_ = np.array(tgt_intensity_test[index] * std_ + mean_)
        print("Total number of steps for comparison: ", len(tgt_))
        preds = xgb_total.predict(X_test_withBASELINE_total) * std_ + mean_
        #print("MAE intensity basin " + basin + " X stat vs "+ forecast + " : ", mean_absolute_error(tgt_intensity_test_withBASELINE * std_ + mean_,
                                                     #xgb.predict(X_test_withBASELINE) * std_ + mean_))
        print("Year ", year, " MAE intensity basin " + basin + " Hurricast : ", np.around(mean_absolute_error(tgt_, preds), decimals = 2), "with std ", np.around(np.std(tgt_ - preds), decimals=2))
        print("Year ", year, " MAE intensity basin " + basin + " Official Forecast "+ forecast + " : ",
              np.around(mean_absolute_error(tgt_, baseline_1), decimals = 2), "with std ", np.around(np.std(tgt_ - baseline_1), decimals = 2))
        if forecast2 != None:
            print("Year ", year, " MAE intensity basin " + basin + " Official Forecast " + forecast2 + " : ",
                np.around(mean_absolute_error(tgt_, baseline_2), decimals=2), "with std ",
                np.around(np.std(tgt_ - baseline_2), decimals=2))

        #print("Year ", year, " Percentage of missed intensification > 20kn Hurricast: ", np.around(sum(tgt_ - preds > 20)/len(preds) * 100, decimals = 2))
        #print("Year ", year, " Percentage of missed intensification > 20kn Official Forecast: ", np.around(sum(tgt_ - baseline_1 > 20) / len(baseline_1) * 100, decimals =2))


def train_xgb_intensity_all_years(forecast2 = None, basin_only = False, sparse = False, max_depth = 8, n_estimators = 140, learning_rate = 0.15, subsample = 0.7, min_child_weight=5, basin = 'AN', forecast = 'HWRF'):
    train = X_train_total
    #test = X_test_total
    tgt_train = tgt_intensity_train
    if sparse:
        train = X_train_total_sparse_x
        #test = X_test_total_sparse_x
    if basin_only:
        train = X_train_total[X_train['cat_basin_'+basin+'_0'] == 1]
        tgt_train = tgt_intensity_train[X_train['cat_basin_'+basin+'_0'] == 1]
    xgb_total = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, min_child_weight=min_child_weight)
    xgb_total.fit(train, tgt_train)
    for year in range(2012, 2020):
        try:
            compare_perf_intensity_per_year(forecast2 = forecast2, xgb_total = xgb_total, basin=basin, forecast=forecast, mode='vmax', year = year)
            print("\n")
        except:
            print("\n No forecasts for year ", year)


################################################################################################################################
################################################################################################################################

####IAI####




grid = iai.GridSearch(
    iai.OptimalFeatureSelectionRegressor(
        random_seed=1,
    ),
    sparsity=range(60, 120, 5),
)

grid.fit(X_train, tgt_intensity_train)
y_hat_intensity = grid.predict(X_test)



#####

xgb_x = XGBRegressor(max_depth=8, n_estimators=120, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb_x.fit(X_train_embed, tgt_displacement_train[:,0])

xgb_y = XGBRegressor(max_depth=8, n_estimators=120, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb_y.fit(X_train_embed, tgt_displacement_train[:,1])
DLATS_PRED = np.array(xgb_x.predict(X_test_embed))*std_dx+mean_dx
DLONS_PRED = np.array(xgb_y.predict(X_test_embed))*std_dy+mean_dy
LATS_PRED = X_test['LAT_7'] + DLATS_PRED
LONS_PRED = X_test['LON_7'] + DLONS_PRED
LATS_TEST = X_test['LAT_7'] + np.array(tgt_displacement_test[:,0])*std_dx+mean_dx
LONS_TEST = X_test['LON_7'] + np.array(tgt_displacement_test[:,1])*std_dy+mean_dy

print("MAE DISPLACEMENT DX DEG: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx+mean_dx, DLATS_PRED))
print("MAE DISPLACEMENT DY DEG: ", mean_absolute_error(np.array(tgt_displacement_test[:,1])*std_dy+mean_dy, DLONS_PRED))

d_km = np.zeros(len(DLONS_PRED))
for i in range(len(DLONS_PRED)):
    d_km[i] = get_distance_km(LONS_PRED[i], LATS_PRED[i], LONS_TEST[i], LATS_TEST[i])

#
d_km.mean()

d_km[X_test_AN_34.index].mean()
d_km[X_test_EP_34.index].mean()
d_km[X_test_WP_34.index].mean()


#USING ONLY RELEVANT SUBBASSINS

X_train_total_AN = X_train_total[X_train['cat_basin_EP_0']+X_train['cat_basin_WP_0']+X_train['cat_basin_AN_0'] == 1]
tgt_displacement_train_AN = tgt_displacement_train[X_train['cat_basin_EP_0']+X_train['cat_basin_WP_0']+X_train['cat_basin_AN_0'] == 1]
xgb_x = XGBRegressor(max_depth=8, n_estimators=120, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb_x.fit(X_train_total_AN, tgt_displacement_train_AN[:,0])
#print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx + mean_dx, np.array(xgb_x.predict(X_test)*std_dx + mean_dx)))

xgb_y = XGBRegressor(max_depth=8, n_estimators=120, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb_y.fit(X_train_total_AN, tgt_displacement_train_AN[:,1])
DLATS_PRED = np.array(xgb_x.predict(X_test_total))*std_dx+mean_dx
DLONS_PRED = np.array(xgb_y.predict(X_test_total))*std_dy+mean_dy
LATS_PRED = X_test['LAT_7'] + DLATS_PRED
LONS_PRED = X_test['LON_7'] + DLONS_PRED
LATS_TEST = X_test['LAT_7'] + np.array(tgt_displacement_test[:,0])*std_dx+mean_dx
LONS_TEST = X_test['LON_7'] + np.array(tgt_displacement_test[:,1])*std_dy+mean_dy

print("MAE DISPLACEMENT: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx+mean_dx, DLATS_PRED))
print("MAE DISPLACEMENT: ", mean_absolute_error(np.array(tgt_displacement_test[:,1])*std_dy+mean_dy, DLONS_PRED))

d_km = np.zeros(len(DLONS_PRED))
for i in range(len(DLONS_PRED)):
    d_km[i] = get_distance_km(LONS_PRED[i], LATS_PRED[i], LONS_TEST[i], LATS_TEST[i])


#

d_km.mean()
d_km[X_test_AN_34.index].mean()
d_km[X_test_EP_34.index].mean()
d_km[X_test_WP_34.index].mean()

###### TABULAR INTENSITY REGRESSION ######
X_train_total = pd.DataFrame(X_train_total)
X_test_total = pd.DataFrame(X_test_total)

col_names = ["col_" + str(i) for i in range(X_train_total.shape[1])]
X_train_total.columns = col_names
X_test_total.columns = col_names

grid = iai.GridSearch(
    iai.OptimalFeatureSelectionRegressor(
        random_seed=1,
    ),
    sparsity=range(130, 220, 10),
)

grid.fit(X_train_total, tgt_intensity_train)
y_hat_intensity = grid.predict(X_test)

numeric_weights, categoric_weights = grid.get_prediction_weights()

X_train_total_sparse = X_train_total[numeric_weights.keys()]
X_test_total_sparse = X_test_total[numeric_weights.keys()]

xgb2 = XGBRegressor(max_depth=8, n_estimators=140, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb2.fit(X_train_total_sparse, tgt_intensity_train)
print("MAE intensity: ", mean_absolute_error(np.array(tgt_intensity_test)*std_+mean_, np.array(xgb2.predict(X_test_total_sparse))*std_+mean_))

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



#### DISPLACEMENT SPARSE TOTAL
from julia import Julia
Julia(sysimage='../../sys.so', compiled_modules = False)
from interpretableai import iai

X_train_total = pd.DataFrame(X_train_total)
X_test_total = pd.DataFrame(X_test_total)

col_names = ["col_" + str(i) for i in range(X_train_total.shape[1])]
X_train_total.columns = col_names
X_test_total.columns = col_names

grid = iai.GridSearch(
    iai.OptimalFeatureSelectionRegressor(
        random_seed=1,
    ),
    sparsity=range(180, 220, 10),
)

grid.fit(X_train_total, np.array(tgt_displacement_train[:,0]))

numeric_weights_x, categoric_weights_x = grid.get_prediction_weights()

X_train_total_sparse_x = X_train_total[numeric_weights_x.keys()]
X_test_total_sparse_x = X_test_total[numeric_weights_x.keys()]

grid.fit(X_train_total, np.array(tgt_displacement_train[:,1]))

numeric_weights_y, categoric_weights_y = grid.get_prediction_weights()

X_train_total_sparse_y = X_train_total[numeric_weights_y.keys()]
X_test_total_sparse_y = X_test_total[numeric_weights_y.keys()]

xgb_x = XGBRegressor(max_depth=8, n_estimators=100, learning_rate = 0.1, subsample = 0.8, min_child_weight = 5)
xgb_x.fit(X_train_total_sparse_x, tgt_displacement_train[:,0])

xgb_y = XGBRegressor(max_depth=8, n_estimators=100, learning_rate = 0.1, subsample = 0.8, min_child_weight = 5)
xgb_y.fit(X_train_total_sparse_y, tgt_displacement_train[:,1])
DLATS_PRED = np.array(xgb_x.predict(X_test_total_sparse_x))*std_dx+mean_dx
DLONS_PRED = np.array(xgb_y.predict(X_test_total_sparse_y))*std_dy+mean_dy
LATS_PRED = X_test['LAT_7'] + DLATS_PRED
LONS_PRED = X_test['LON_7'] + DLONS_PRED
LATS_TEST = X_test['LAT_7'] + np.array(tgt_displacement_test[:,0])*std_dx+mean_dx
LONS_TEST = X_test['LON_7'] + np.array(tgt_displacement_test[:,1])*std_dy+mean_dy

print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx+mean_dx, DLATS_PRED))
print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,1])*std_dy+mean_dy, DLONS_PRED))

d_km = np.zeros(len(DLONS_PRED))
for i in range(len(DLONS_PRED)):
    d_km[i] = get_distance_km(LONS_PRED[i], LATS_PRED[i], LONS_TEST[i], LATS_TEST[i])


#
d_km.mean()
d_km[X_test_AN_34.index].mean() #best obtained is 129.56
d_km[X_test_EP_34.index].mean() #best obtained is 81.48
d_km[X_test_WP_34.index].mean() #best obtained is 114.18 (113.



#####

xgb_x = XGBRegressor(max_depth=8, n_estimators=100, learning_rate = 0.07, subsample = 0.8, min_child_weight = 5)
xgb_x.fit(X_train_total_sparse_x, tgt_displacement_train[:,0])

xgb_y = XGBRegressor(max_depth=8, n_estimators=100, learning_rate = 0.07, subsample = 0.8, min_child_weight = 5)
xgb_y.fit(X_train_total_sparse_y, tgt_displacement_train[:,1])
DLATS_PRED = np.array(xgb_x.predict(X_test_total_sparse_x))*std_dx+mean_dx
DLONS_PRED = np.array(xgb_y.predict(X_test_total_sparse_y))*std_dy+mean_dy
LATS_PRED = X_test['LAT_7'] + DLATS_PRED
LONS_PRED = X_test['LON_7'] + DLONS_PRED
LATS_TEST = X_test['LAT_7'] + np.array(tgt_displacement_test[:,0])*std_dx+mean_dx
LONS_TEST = X_test['LON_7'] + np.array(tgt_displacement_test[:,1])*std_dy+mean_dy

print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx+mean_dx, DLATS_PRED))
print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,1])*std_dy+mean_dy, DLONS_PRED))

d_km = np.zeros(len(DLONS_PRED))
for i in range(len(DLONS_PRED)):
    d_km[i] = get_distance_km(LONS_PRED[i], LATS_PRED[i], LONS_TEST[i], LATS_TEST[i])


#
d_km.mean()
d_km[X_test_AN_34.index].mean() #best obtained is 129.14
d_km[X_test_EP_34.index].mean() #best obtained is 81.48
d_km[X_test_WP_34.index].mean() #best obtained is 114.18 (113.


xgb_x = XGBRegressor(max_depth=8, n_estimators=100, learning_rate = 0.07, subsample = 0.95, min_child_weight = 5)
xgb_x.fit(X_train_total_sparse_x, tgt_displacement_train[:,0])

xgb_y = XGBRegressor(max_depth=8, n_estimators=100, learning_rate = 0.07, subsample = 0.95, min_child_weight = 5)
xgb_y.fit(X_train_total_sparse_y, tgt_displacement_train[:,1])
DLATS_PRED = np.array(xgb_x.predict(X_test_total_sparse_x))*std_dx+mean_dx
DLONS_PRED = np.array(xgb_y.predict(X_test_total_sparse_y))*std_dy+mean_dy
LATS_PRED = X_test['LAT_7'] + DLATS_PRED
LONS_PRED = X_test['LON_7'] + DLONS_PRED
LATS_TEST = X_test['LAT_7'] + np.array(tgt_displacement_test[:,0])*std_dx+mean_dx
LONS_TEST = X_test['LON_7'] + np.array(tgt_displacement_test[:,1])*std_dy+mean_dy

print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx+mean_dx, DLATS_PRED))
print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,1])*std_dy+mean_dy, DLONS_PRED))

d_km = np.zeros(len(DLONS_PRED))
for i in range(len(DLONS_PRED)):
    d_km[i] = get_distance_km(LONS_PRED[i], LATS_PRED[i], LONS_TEST[i], LATS_TEST[i])


#
d_km.mean()
d_km[X_test_AN_34.index].mean() #best obtained is 129.14
d_km[X_test_EP_34.index].mean() #best obtained is 81.48
d_km[X_test_WP_34.index].mean() #best obtained is 114.18 (113.




#####

def train_xgb_track2(train_test_split = 0.8, last_storms = 1000, basin_only = False, sparse = False, max_depth = 8, n_estimators = 140, learning_rate = 0.15, subsample = 0.7, min_child_weight=5, basin = 'AN', forecast = 'HWRF'):
    train_x = X_train_total
    train_y = X_train_total
    test_x = X_test_total
    test_y = X_test_total
    tgt_train = tgt_displacement_train
    if sparse:
        train_x, train_y = X_train_total_sparse_x, X_train_total_sparse_y
        test_x, test_y = X_test_total_sparse_x, X_test_total_sparse_y
    if basin_only:
        train_x = X_train_total[X_train['cat_basin_'+basin+'_0'] == 1]
        train_y = train_x
        tgt_train = tgt_displacement_train[X_train['cat_basin_'+basin+'_0'] == 1]
    n = 0
    if train_test_split > 0.8:
        n = int(train_test_split*(len(X_train)+len(X_test)) - len(X_train))
        train_x = np.concatenate((X_train_total, X_test_total[:n]), axis = 0)
        train_y = train_x
        test_x = X_test_total[n:]
        test_y = test_x
        tgt_train = np.concatenate((tgt_displacement_train_unst, tgt_displacement_test_unst[:n]), axis = 0)
        tgt_test = tgt_displacement_test_unst[n:]
        tgt_train, _, std_dx2, mean_dx2, std_dy2, mean_dy2 = standardize(tgt_train, tgt_test)
    xgb_x = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, min_child_weight=min_child_weight)
    xgb_x.fit(train_x, tgt_train[:, 0])
    xgb_y = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, min_child_weight=min_child_weight)
    xgb_y.fit(train_y, tgt_train[:, 1])
    if train_test_split > 0.8:
        DLATS_PRED = np.array(xgb_x.predict(test_x)) * std_dx2 + mean_dx2
        DLONS_PRED = np.array(xgb_y.predict(test_y)) * std_dy2 + mean_dy2
        LATS_PRED_ = X_test[n:]['LAT_7'] + DLATS_PRED
        LONS_PRED_ = X_test[n:]['LON_7'] + DLONS_PRED
    else:
        DLATS_PRED = np.array(xgb_x.predict(test_x)) * std_dx + mean_dx
        DLONS_PRED = np.array(xgb_y.predict(test_y)) * std_dy + mean_dy
        LATS_PRED_ = X_test['LAT_7'] + DLATS_PRED
        LONS_PRED_ = X_test['LON_7'] + DLONS_PRED
    compare_perf_track(basin=basin, forecast=forecast, mode='lat', LATS_PRED_=LATS_PRED_, LONS_PRED_=LONS_PRED_, last_storms = last_storms, n = n)




def compare_perf_track2(LATS_PRED_, LONS_PRED_, basin = 'AN', forecast = 'HWRF', mode = 'lat', last_storms = 1000, n = 0):
    if n == 0:
        index = X_test_baseline.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1].index#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0].index
        baseline_ = X_test_baseline.loc[X_test_baseline[forecast + '_24_'+mode+'_7'] > -320].loc[X_test_baseline['cat_basin_'+basin+'_0'] == 1]#.loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0]
    LATS_TEST_ = X_test['LAT_7'] + np.array(tgt_displacement_test_unst[:, 0])
    LONS_TEST_ = X_test['LON_7'] + np.array(tgt_displacement_test_unst[:, 1])
    if n > 0:
        index = X_test_baseline[n:].loc[X_test_baseline[n:][forecast + '_24_' + mode + '_7'] > -320].loc[X_test_baseline[n:][
                                                                                                     'cat_basin_' + basin + '_0'] == 1].index  # .loc[#X_test_baseline['SHIP_24_'+mode+'_7'] > 0].index
        baseline_ = X_test_baseline[n:].loc[X_test_baseline[n:][forecast + '_24_' + mode + '_7'] > -320].loc[
            X_test_baseline[n:]['cat_basin_' + basin + '_0'] == 1]
        LATS_TEST_ = LATS_TEST_[n:]
        LONS_TEST_ = LONS_TEST_[n:]
    baseline_1 = baseline_[forecast + '_24_'+mode+'_7']
    if mode == 'lat':
        baseline_2 = baseline_[forecast + '_24_lon_7']
        LATS_BASE = np.array(baseline_1)
        LONS_BASE = np.array(baseline_2)
        LATS_TEST_ = np.array(LATS_TEST_[index])
        LONS_TEST_ = np.array(LONS_TEST_[index])
        LATS_PRED_ = np.array(LATS_PRED_[index])
        LONS_PRED_ = np.array(LONS_PRED_[index])
        d_km_baseline = np.zeros(len(LATS_BASE))
        print(LATS_TEST_)
        print(LATS_PRED_)
        for i in range(len(LATS_BASE)):
            d_km_baseline[i] = get_distance_km(LONS_BASE[i], LATS_BASE[i], LONS_TEST_[i], LATS_TEST_[i])
        print("MAE Distance basin " + basin + " with " + forecast + ": ", d_km_baseline.mean(), "and std: ", d_km_baseline.std())
        print("MAE Distance basin " + basin + " with " + forecast + ": last timesteps ", last_storms, ": ", d_km_baseline[-last_storms:].mean(), "and std: ", d_km_baseline[-last_storms:].std())
        print("Number of busts > 200km", sum(d_km_baseline > 200))
        d_km_pred = np.zeros(len(LONS_PRED_))
        for i in range(len(LONS_PRED_)):
            d_km_pred[i] = get_distance_km(LONS_PRED_[i], LATS_PRED_[i], LONS_TEST_[i], LATS_TEST_[i])
        print("MAE Distance basin " + basin + " Hurricast: ", d_km_pred.mean(), "and std: ", d_km_pred.std())
        print("MAE Distance basin " + basin + " Hurricast last timesteps ", last_storms, ": ", d_km_pred[-last_storms:].mean(), "and std: ", d_km_pred[-last_storms:].std())
        print("Number of busts > 200km", sum(d_km_pred > 200))



#### BASELINE INTENSITY

#SHIP

X_test_withBASELINE_SPEED_SHIP = X_test[X_test_baseline['SHIP_24_vmax_7'] > 0]
X_test_withBASELINE_SPEED_SHIP_total = X_test_total[X_test_baseline['SHIP_24_vmax_7'] > 0]
tgt_intensity_test_BASELINE_SPEED_SHIP = tgt_intensity_test[X_test_baseline['SHIP_24_vmax_7'] > 0]
baseline_intensity_SHIP = X_test_baseline[X_test_baseline['SHIP_24_vmax_7'] > 0]
baseline_intensity_SHIP = baseline_intensity_SHIP['SHIP_24_vmax_7']

print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_SHIP*std_+mean_, xgb2.predict(X_test_withBASELINE_SPEED_SHIP)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_SHIP*std_+mean_, xgb.predict(X_test_withBASELINE_SPEED_SHIP_total)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_SHIP*std_+mean_, baseline_intensity_SHIP))

#HWRF

X_test_withBASELINE_SPEED_HWRF = X_test[X_test_baseline[forecast + '_24_vmax_7'] > 0]
X_test_withBASELINE_SPEED_HWRF_total = X_test_total[X_test_baseline[forecast + '_24_vmax_7'] > 0]
tgt_intensity_test_BASELINE_SPEED_HWRF = tgt_intensity_test[X_test_baseline[forecast + '_24_vmax_7'] > 0]
baseline_intensity_HWRF = X_test_baseline[X_test_baseline[forecast + '_24_vmax_7'] > 0]
baseline_intensity_HWRF = baseline_intensity_HWRF[forecast + '_24_vmax_7']

print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_HWRF*std_+mean_, xgb2.predict(X_test_withBASELINE_SPEED_HWRF)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_HWRF*std_+mean_, xgb.predict(X_test_withBASELINE_SPEED_HWRF_total)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_HWRF*std_+mean_, baseline_intensity_HWRF))

#HWRF + SHIP
index = X_test_baseline.loc[X_test_baseline[forecast + '_24_vmax_7'] > 0].loc[X_test_baseline['SHIP_24_vmax_7'] > 0].index

X_test_withBASELINE_SPEED_ = X_test.loc[X_test_baseline[forecast + '_24_vmax_7'] > 0].loc[X_test_baseline['SHIP_24_vmax_7'] > 0]
X_test_withBASELINE_SPEED_total_ = X_test_total[index]
tgt_intensity_test_BASELINE_SPEED_ = tgt_intensity_test[index]
baseline_intensity_ = X_test_baseline.loc[X_test_baseline[forecast + '_24_vmax_7'] > 0].loc[X_test_baseline['SHIP_24_vmax_7'] > 0]
baseline_intensity_ = baseline_intensity_[forecast + '_24_vmax_7']

print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_*std_+mean_, xgb2.predict(X_test_withBASELINE_SPEED_)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_*std_+mean_, xgb.predict(X_test_withBASELINE_SPEED_total_)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_*std_+mean_, baseline_intensity_))

index2 = X_test_baseline.loc[X_test_baseline[forecast + '_24_vmax_7'] > 0].loc[X_test_baseline['SHIP_24_vmax_7'] > 0].loc[X_test_baseline['cat_basin_AN_0'] == 1].index

X_test_withBASELINE_SPEED_2 = X_test.loc[X_test_baseline[forecast + '_24_vmax_7'] > 0].loc[X_test_baseline['SHIP_24_vmax_7'] > 0].loc[X_test_baseline['cat_basin_AN_0'] == 1]
X_test_withBASELINE_SPEED_total_2 = X_test_total[index2]
tgt_intensity_test_BASELINE_SPEED_2 = tgt_intensity_test[index2]
baseline_intensity_2 = X_test_baseline.loc[X_test_baseline[forecast + '_24_vmax_7'] > 0].loc[X_test_baseline['SHIP_24_vmax_7'] > 0].loc[X_test_baseline['cat_basin_AN_0'] == 1]
baseline_intensity_2 = baseline_intensity_2[forecast + '_24_vmax_7']

print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_2*std_+mean_, xgb2.predict(X_test_withBASELINE_SPEED_2)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_2*std_+mean_, xgb.predict(X_test_withBASELINE_SPEED_total_2)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_2*std_+mean_, baseline_intensity_2))

#EMXI

X_test_withBASELINE_SPEED_EMXI = X_test[X_test_baseline['EMXI_24_vmax_7'] > 0]
X_test_withBASELINE_SPEED_EMXI_total = X_test_total[X_test_baseline['EMXI_24_vmax_7'] > 0]
tgt_intensity_test_BASELINE_SPEED_EMXI = tgt_intensity_test[X_test_baseline['EMXI_24_vmax_7'] > 0]
baseline_intensity_EMXI = X_test_baseline[X_test_baseline['EMXI_24_vmax_7'] > 0]
baseline_intensity_EMXI = baseline_intensity_EMXI['EMXI_24_vmax_7']

#Same in AN

X_test_withBASELINE_SPEED_EMXI_AN = X_test_withBASELINE_SPEED_EMXI.loc[X_test_withBASELINE_SPEED_EMXI['cat_basin_AN_0'] == 1]
X_test_withBASELINE_SPEED_EMXI_total_AN = X_test_total[X_test_withBASELINE_SPEED_EMXI_AN.index]
tgt_intensity_test_BASELINE_SPEED_EMXI_AN = tgt_intensity_test[X_test_withBASELINE_SPEED_EMXI_AN.index]
baseline_intensity_EMXI_AN = X_test_baseline.loc[X_test_baseline['EMXI_24_vmax_7'] > 0].loc[X_test_baseline['cat_basin_AN_0'] == 1]['EMXI_24_vmax_7']




##

X_test_withBASELINE_SPEED_EMXI_AN = X_test_withBASELINE_SPEED_EMXI.loc[X_test_withBASELINE_SPEED_EMXI['cat_basin_AN_0'] == 1]
X_test_withBASELINE_SPEED_EMXI_total_AN = X_test_total[X_test_withBASELINE_SPEED_EMXI_AN.index]
tgt_intensity_test_BASELINE_SPEED_EMXI_AN = tgt_intensity_test[X_test_withBASELINE_SPEED_EMXI_AN.index]
baseline_intensity_EMXI_AN = X_test_baseline.loc[X_test_baseline['EMXI_24_vmax_7'] > 0].loc[X_test_baseline['cat_basin_AN_0'] == 1]['EMXI_24_vmax_7']


print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_EMXI_AN*std_+mean_, xgb2.predict(X_test_withBASELINE_SPEED_EMXI_AN)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_EMXI_AN*std_+mean_, xgb.predict(X_test_withBASELINE_SPEED_EMXI_total_AN)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_EMXI_AN*std_+mean_, baseline_intensity_EMXI_AN))

#EP

X_test_withBASELINE_SPEED_EMXI_AN = X_test_withBASELINE_SPEED_EMXI.loc[X_test_withBASELINE_SPEED_EMXI['cat_basin_EP_0'] == 1]
X_test_withBASELINE_SPEED_EMXI_total_AN = X_test_total[X_test_withBASELINE_SPEED_EMXI_AN.index]
tgt_intensity_test_BASELINE_SPEED_EMXI_AN = tgt_intensity_test[X_test_withBASELINE_SPEED_EMXI_AN.index]
baseline_intensity_EMXI_AN = X_test_baseline.loc[X_test_baseline['EMXI_24_vmax_7'] > 0].loc[X_test_baseline['cat_basin_EP_0'] == 1]['EMXI_24_vmax_7']


print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_EMXI_AN*std_+mean_, xgb2.predict(X_test_withBASELINE_SPEED_EMXI_AN)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_EMXI_AN*std_+mean_, xgb.predict(X_test_withBASELINE_SPEED_EMXI_total_AN)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_BASELINE_SPEED_EMXI_AN*std_+mean_, baseline_intensity_EMXI_AN))



#### INTENSITY SUBBASINS TEST ####

X_test_WP = X_test_total[X_test['cat_basin_WP_0'] == 1]
X_test_EP = X_test_total[X_test['cat_basin_EP_0'] == 1]
X_test_AN = X_test_total[X_test['cat_basin_AN_0'] == 1]
tgt_intensity_test_EP = tgt_intensity_test[X_test['cat_basin_EP_0'] == 1]
tgt_intensity_test_WP = tgt_intensity_test[X_test['cat_basin_WP_0'] == 1]
tgt_intensity_test_AN = tgt_intensity_test[X_test['cat_basin_AN_0'] == 1]

print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_EP*std_+mean_, xgb.predict(X_test_EP)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_WP*std_+mean_, xgb.predict(X_test_WP)*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test_AN*std_+mean_, xgb.predict(X_test_AN)*std_+mean_))

#ERROR ON HIGHER CAT STORMS
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test[X_test['WMO_WIND_7'] > 34]*std_+mean_, xgb.predict(X_test_total[X_test['WMO_WIND_7'] > 34])*std_+mean_))
print("MAE intensity: ", mean_absolute_error(tgt_intensity_test[X_test['WMO_WIND_7'] > 34]*std_+mean_, xgb2.predict(X_test[X_test['WMO_WIND_7'] > 34])*std_+mean_))

X_test_AN_34 = X_test.loc[X_test['WMO_WIND_7'] > 34].loc[X_test['cat_basin_AN_0'] == 1]
X_test_EP_34 = X_test.loc[X_test['WMO_WIND_7'] > 34].loc[X_test['cat_basin_EP_0'] == 1]
X_test_WP_34 = X_test.loc[X_test['WMO_WIND_7'] > 34].loc[X_test['cat_basin_WP_0'] == 1]

X_test_AN_34_tot = X_test_total[X_test_AN_34.index]
X_test_EP_34_tot = X_test_total[X_test_EP_34.index]
X_test_WP_34_tot = X_test_total[X_test_WP_34.index]

tgt_intensity_test_AN_34 = tgt_intensity_test[X_test_AN_34.index]
tgt_intensity_test_EP_34 = tgt_intensity_test[X_test_EP_34.index]
tgt_intensity_test_WP_34 = tgt_intensity_test[X_test_WP_34.index]

print("MAE intensity AN 34 STAT: ", mean_absolute_error(tgt_intensity_test_AN_34*std_+mean_, xgb2.predict(X_test_AN_34)*std_+mean_))
print("MAE intensity EP 34 STAT: ", mean_absolute_error(tgt_intensity_test_EP_34*std_+mean_, xgb2.predict(X_test_EP_34)*std_+mean_))
print("MAE intensity WP 34 STAT: ", mean_absolute_error(tgt_intensity_test_WP_34*std_+mean_, xgb2.predict(X_test_WP_34)*std_+mean_))

print("MAE intensity AN 34 TOT: ", mean_absolute_error(tgt_intensity_test_AN_34*std_+mean_, xgb.predict(X_test_AN_34_tot)*std_+mean_))
print("MAE intensity EP 34 TOT: ", mean_absolute_error(tgt_intensity_test_EP_34*std_+mean_, xgb.predict(X_test_EP_34_tot)*std_+mean_))
print("MAE intensity WP 34 TOT: ", mean_absolute_error(tgt_intensity_test_WP_34*std_+mean_, xgb.predict(X_test_WP_34_tot)*std_+mean_))

##### DISPLACEMENT #####

## X TRAIN
xgb_x = XGBRegressor(max_depth=7, n_estimators=140, learning_rate = 0.15, subsample = 0.7, min_child_weight = 5)
xgb_x.fit(X_train, tgt_displacement_train[:,0])

xgb_y = XGBRegressor(max_depth=7, n_estimators=140, learning_rate = 0.15, subsample = 0.7, min_child_weight = 5)
xgb_y.fit(X_train, tgt_displacement_train[:,1])
DLATS_PRED = np.array(xgb_x.predict(X_test))*std_dx+mean_dx
DLONS_PRED = np.array(xgb_y.predict(X_test))*std_dy+mean_dy
LATS_PRED = X_test['LAT_7'] + DLATS_PRED
LONS_PRED = X_test['LON_7'] + DLONS_PRED
LATS_TEST = X_test['LAT_7'] + np.array(tgt_displacement_test[:,0])*std_dx+mean_dx
LONS_TEST = X_test['LON_7'] + np.array(tgt_displacement_test[:,1])*std_dy+mean_dy

print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx+mean_dx, DLATS_PRED))
print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,1])*std_dy+mean_dy, DLONS_PRED))

d_km = np.zeros(len(DLONS_PRED))
for i in range(len(DLONS_PRED)):
    d_km[i] = get_distance_km(LONS_PRED[i], LATS_PRED[i], LONS_TEST[i], LATS_TEST[i])


#

d_km.mean()
d_km[X_test_AN_34.index].mean()
d_km[X_test_EP_34.index].mean()
d_km[X_test_WP_34.index].mean()

## X TRAIN TOTAL

xgb_x = XGBRegressor(max_depth=9, n_estimators=120, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb_x.fit(X_train_total, tgt_displacement_train[:,0])

xgb_y = XGBRegressor(max_depth=9, n_estimators=120, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb_y.fit(X_train_total, tgt_displacement_train[:,1])
DLATS_PRED = np.array(xgb_x.predict(X_test_total))*std_dx+mean_dx
DLONS_PRED = np.array(xgb_y.predict(X_test_total))*std_dy+mean_dy
LATS_PRED = X_test['LAT_7'] + DLATS_PRED
LONS_PRED = X_test['LON_7'] + DLONS_PRED
LATS_TEST = X_test['LAT_7'] + np.array(tgt_displacement_test[:,0])*std_dx+mean_dx
LONS_TEST = X_test['LON_7'] + np.array(tgt_displacement_test[:,1])*std_dy+mean_dy

print("MAE DISPLACEMENT DX DEG: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx+mean_dx, DLATS_PRED))
print("MAE DISPLACEMENT DY DEG: ", mean_absolute_error(np.array(tgt_displacement_test[:,1])*std_dy+mean_dy, DLONS_PRED))

d_km = np.zeros(len(DLONS_PRED))
for i in range(len(DLONS_PRED)):
    d_km[i] = get_distance_km(LONS_PRED[i], LATS_PRED[i], LONS_TEST[i], LATS_TEST[i])

#
d_km.mean()

d_km[X_test_AN_34.index].mean()
d_km[X_test_EP_34.index].mean()
d_km[X_test_WP_34.index].mean()



