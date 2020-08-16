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
args.output_dir = '../results/results8_16_18_57_7' #Reached 70.4 and the best results with combination. 7.47 GRU outputs 64. All rest normal.
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


embeds_train1 = np.array(model.get_embeddings(x_stat_train[:30000], x_viz_train[:30000]).reshape(-1, 64*args.window_size).detach().numpy())
embeds_train2 = np.array(model.get_embeddings(x_stat_train[30000:55000], x_viz_train[30000:55000]).reshape(-1, 64*args.window_size).detach().numpy())
embeds_train3 = np.array(model.get_embeddings(x_stat_train[55000:80000], x_viz_train[55000:80000]).reshape(-1, 64*args.window_size).detach().numpy())
embeds_train4 = np.array(model.get_embeddings(x_stat_train[80000:], x_viz_train[80000:]).reshape(-1, 64*args.window_size).detach().numpy())
X_test_embed = np.array(model.get_embeddings(x_viz_test, x_stat_test).reshape(-1, 64*args.window_size).detach().numpy())


   

