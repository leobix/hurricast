from run import Prepro
import numpy as np



import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--window_size", type=int, default=8,
                            help="number of in time steps")

parser.add_argument("--steps_out", type=int, default=8,
                            help="steps out")

parser.add_argument("--vision", action='store_true',
                            help="save vision data")

def main(args):
    ###
    train_test_split = 0.8 #how much train test data
    steps_out = args.steps_out #steps_out
    window_size = args.window_size #how many timesteps from the past to take ie steps_in


    ####
    vision_data = np.load('data/vision_data_34_20_120_1980.npy', allow_pickle = True)
    y = np.load('data/y_34_20_120_1980.npy', allow_pickle = True)

    ####
    train_tensors, test_tensors = Prepro.process(vision_data, y, train_test_split, predict_at = steps_out, window_size =  window_size)
    x_viz_train, x_stat_train, tgt_intensity_cat_train, tgt_intensity_cat_baseline_train, tgt_displacement_train, tgt_intensity_train = train_tensors
    x_viz_test, x_stat_test, tgt_intensity_cat_test, tgt_intensity_cat_baseline_test, tgt_displacement_test, tgt_intensity_test = test_tensors

    X_train = x_stat_train.reshape(x_stat_train.shape[0], -1)
    X_test = x_stat_test.reshape(x_stat_test.shape[0], -1)

    X_train_vision = x_viz_train.reshape(x_viz_train.shape[0], -1)
    X_test_vision = x_viz_test.reshape(x_viz_test.shape[0], -1)

    np.save('data/X_train_stat_1980_34_20_120_w' + str(window_size) + '_at_' + str(steps_out) + '.npy', X_train, allow_pickle = True)
    np.save('data/X_test_stat_1980_34_20_120_w' + str(window_size)+ '_at_' + str(steps_out) + '.npy', X_test, allow_pickle = True)

    if args.vision:
        np.save('data/X_train_vision_1980_34_20_120_w' + str(window_size)+ '_at_' + str(steps_out) + '.npy', X_train_vision, allow_pickle = True)
        np.save('data/X_test_vision_1980_34_20_120_w' + str(window_size) + '_at_' + str(steps_out) + '.npy', X_test_vision, allow_pickle = True)

    np.save('data/y_train_intensity_cat_1980_34_20_120_w' + str(window_size)+ '_at_' + str(steps_out) + '.npy', tgt_intensity_cat_train, allow_pickle = True)
    np.save('data/y_test_intensity_cat_1980_34_20_120_w' + str(window_size)+ '_at_' + str(steps_out) + '.npy', tgt_intensity_cat_test, allow_pickle = True)

    np.save('data/y_train_displacement_1980_34_20_120_w' + str(window_size)+ '_at_' + str(steps_out) + '.npy', tgt_displacement_train, allow_pickle = True)
    np.save('data/y_test_displacement_1980_34_20_120_w' + str(window_size)+ '_at_' + str(steps_out) + '.npy', tgt_displacement_test, allow_pickle = True)

    np.save('data/y_train_intensity_cat_baseline_1980_34_20_120_w' + str(window_size) + '_at_' + str(steps_out) + '.npy', tgt_intensity_cat_baseline_train, allow_pickle = True)
    np.save('data/y_test_intensity_cat_baseline_1980_34_20_120_w' + str(window_size) + '_at_' + str(steps_out) + '.npy', tgt_intensity_cat_baseline_test, allow_pickle = True)

    np.save('data/y_train_intensity_1980_34_20_120_w' + str(window_size) + '_at_' + str(steps_out) + '.npy', tgt_intensity_train, allow_pickle = True)
    np.save('data/y_test_intensity_1980_34_20_120_w' + str(window_size) + '_at_' + str(steps_out) + '.npy', tgt_intensity_test, allow_pickle = True)


if __name__ == "__main__":
   args = parser.parse_args()
   print(args)
   main(args)
