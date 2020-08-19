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

x_stat_train = torch.Tensor(np.load('../data/X_train_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True).reshape(-1, args.window_size, 30)[:,:,:14]).to(device)
x_stat_test = torch.Tensor(np.load('../data/X_test_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True).reshape(-1, args.window_size, 30)[:,:,:14]).to(device)

x_viz_train = torch.Tensor(np.load('../data/X_train_vision_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',  allow_pickle = True).reshape(-1, args.window_size, 9, 25, 25)).to(device)
x_viz_test = torch.Tensor(np.load('../data/X_test_vision_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle = True).reshape(-1, args.window_size, 9, 25, 25)).to(device)

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


means = x_viz_train.mean(dim=(0, 1, 3, 4))
stds = x_viz_train.std(dim=(0, 1, 3, 4))

means_stat = x_stat_train[:,:,:6].mean(dim=(0, 1))
stds_stat = x_stat_train[:,:,:6].std(dim=(0, 1))

for i in range(len(means)):
            x_viz_train[:, :, i] = (x_viz_train[:, :, i] - means[i]) / stds[i]
            x_viz_test[:, :, i] = (x_viz_test[:, :, i] - means[i]) / stds[i]

for i in range(len(means_stat)):
            x_stat_train[:, :, i] = (x_stat_train[:, :, i] - means_stat[i]) / stds_stat[i]
            x_stat_test[:, :, i] = (x_stat_test[:, :, i] - means_stat[i]) / stds_stat[i]

###### LOADING MODELS AND GETTING EMBEDDINGS
############################################
modes = {#Modes and associated tasks
    'intensity': 'regression',
    'displacement': 'regression',
    'intensity_cat': 'classification',
    'baseline_intensity_cat': 'classification',
    'baseline_displacement': 'regression'
}


task = modes[args.mode]
print('MODE AND TASK: {} | {}'.format(args.mode, task))


#========================
encoder_conf = config.create_config(args.encoder_config)
decoder_conf = config.create_config(args.decoder_config)
    
#=======================
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


baseline_intensity = x_stat_test[:, x_stat_test.shape[1]-1, 2]

std_ = float(std_intensity)
mean_ = float(mean_intensity)

xgb2 = XGBRegressor(max_depth=6, n_estimators=140, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb2.fit(X_train, tgt_intensity_train)
print("MAE intensity: ", mean_absolute_error(np.array(tgt_intensity_test)*std_+mean_, np.array(xgb2.predict(X_test))*std_+mean_))

xgb = XGBRegressor(max_depth=8, n_estimators = 120, learning_rate = 0.07, subsample = 0.7)
xgb.fit(X_train_total, tgt_intensity_train)
print("MAE intensity: ", mean_absolute_error(np.array(tgt_intensity_test)*std_+mean_, np.array(xgb.predict(X_test_total))*std_+mean_))

xgb3 = XGBClassifier(max_depth=6, n_estimators=140, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb3.fit(X_train, tgt_intensity_cat_train)
print("Accuracy: ", accuracy_score(tgt_intensity_cat_test, xgb3.predict(X_test)))

xgb3 = XGBClassifier(max_depth=8, n_estimators=140, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb3.fit(X_train_total, tgt_intensity_cat_train)
print("Accuracy: ", accuracy_score(tgt_intensity_cat_test, xgb3.predict(X_test_total)))


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
xgb_x = XGBRegressor(max_depth=7, n_estimators=140, learning_rate = 0.15, subsample = 0.7, min_child_weight = 5)
xgb_x.fit(X_train, tgt_displacement_train[:,0])
#print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx + mean_dx, np.array(xgb_x.predict(X_test)*std_dx + mean_dx)))

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

#print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,1])*std_dy + mean_dy, np.array(xgb_y.predict(X_test)*std_dy + mean_dy)))

xgb_x = XGBRegressor(max_depth=9, n_estimators=120, learning_rate = 0.07, subsample = 0.7, min_child_weight = 5)
xgb_x.fit(X_train_total, tgt_displacement_train[:,0])
#print("MAE intensity: ", mean_absolute_error(np.array(tgt_displacement_test[:,0])*std_dx + mean_dx, np.array(xgb_x.predict(X_test)*std_dx + mean_dx)))

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
