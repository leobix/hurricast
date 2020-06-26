from run import Prepro
import numpy as np

###
train_test_split = 0.8 #how much train test data
predict_at = 8 #steps_out
window_size = 8 #how many timesteps from the past to take ie steps_in


####
vision_data = np.load('data/vision_data_50_20_90_1980_v3.npy', allow_pickle = True)
y = np.load('data/y_50_20_90_1980_v3.npy', allow_pickle = True) 

####
train_tensors, test_tensors = Prepro.process(vision_data, y, train_test_split, predict_at = predict_at, window_size =  window_size)
x_viz_train, x_stat_train, tgt_intensity_cat_train, tgt_intensity_cat_baseline_train, tgt_displacement_train, tgt_intensity_train = train_tensors
x_viz_test, x_stat_test, tgt_intensity_cat_test, tgt_intensity_cat_baseline_test, tgt_displacement_test, tgt_intensity_test = test_tensors

X_train = x_stat_train.reshape(x_stat_train.shape[0], -1)
X_test = x_stat_test.reshape(x_stat_test.shape[0], -1)

X_train_vision = x_viz_train.reshape(x_viz_train.shape[0], -1)
X_test_vision = x_viz_test.reshape(x_viz_test.shape[0], -1)

np.save('data/X_train_stat_1980_50_20_90_w' + str(window_size) + '.npy', X_train, allow_pickle = True)
np.save('data/X_test_stat_1980_50_20_90_w' + str(window_size) + '.npy', X_test, allow_pickle = True)

np.save('data/X_train_vision_1980_50_20_90_w' + str(window_size) + '.npy', X_train_vision, allow_pickle = True)
np.save('data/X_test_vision_1980_50_20_90_w' + str(window_size) + '.npy', X_test_vision, allow_pickle = True)

np.save('data/y_train_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy', tgt_intensity_cat_train, allow_pickle = True)
np.save('data/y_test_intensity_cat_1980_50_20_90_w' + str(window_size) + '.npy', tgt_intensity_cat_test, allow_pickle = True)

np.save('data/y_train_displacement_1980_50_20_90_w' + str(window_size) + '.npy', tgt_displacement_train, allow_pickle = True)
np.save('data/y_test_displacement_1980_50_20_90_w' + str(window_size) + '.npy', tgt_displacement_test, allow_pickle = True)

np.save('data/y_train_intensity_cat_baseline_1980_50_20_90_w' + str(window_size) + '.npy', tgt_intensity_cat_baseline_train, allow_pickle = True)
np.save('data/y_test_intensity_cat_baseline_1980_50_20_90_w' + str(window_size) + '.npy', tgt_intensity_cat_baseline_test, allow_pickle = True)
