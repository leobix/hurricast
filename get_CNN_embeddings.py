import sys

sys.path.append('../')
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import models
import matplotlib.pyplot as plt
import tqdm
import os
import os.path as osp
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error


def main(args):
    # Prepare device
    device = torch.device(
        f'cuda:{args.gpu_nb}' if torch.cuda.is_available() and args.gpu_nb != -1 else 'cpu')
    print(' Prepare the training using ', device)
    # Load files and reformat.

    x_stat_train = torch.Tensor(np.load('data/X_train_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True).reshape(-1, args.window_size, 25)[:,:,:9])
    x_stat_test = torch.Tensor(np.load('data/X_test_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True).reshape(-1, args.window_size, 25)[:,:,:9])

    x_viz_train = torch.Tensor(np.load('data/X_train_vision_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',  allow_pickle = True).reshape(-1, args.window_size, 9, 25, 25))
    x_viz_test = torch.Tensor(np.load('data/X_test_vision_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle = True).reshape(-1, args.window_size, 9, 25, 25))

    tgt_intensity_cat_train = torch.LongTensor(np.load('data/y_train_intensity_cat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                      allow_pickle=True))
    tgt_intensity_cat_test = torch.LongTensor(np.load('data/y_test_intensity_cat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                     allow_pickle=True))

    tgt_intensity_train = torch.Tensor(np.load('data/y_train_intensity_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                  allow_pickle=True))
    tgt_intensity_test = torch.Tensor(np.load('data/y_test_intensity_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                 allow_pickle=True))

    tgt_intensity_cat_baseline_train = torch.LongTensor(np.load('data/y_train_intensity_cat_baseline_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',  allow_pickle = True))
    tgt_intensity_cat_baseline_test = torch.LongTensor(np.load('data/y_test_intensity_cat_baseline_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True))

    tgt_displacement_train = torch.Tensor(np.load('data/y_train_displacement_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                     allow_pickle=True))
    tgt_displacement_test = torch.Tensor(np.load('data/y_test_displacement_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                    allow_pickle=True))

    #Standardize velocity target
    if args.normalize_intensity:
        max_intensity = tgt_intensity_train.max()
        min_intensity = tgt_intensity_train.min()
        tgt_intensity_train = (tgt_intensity_train - min_intensity) / (max_intensity - min_intensity)
        tgt_intensity_test = (tgt_intensity_test - min_intensity) / (max_intensity - min_intensity)
        #not actual values but used for denormalizing and making it simpler
        mean_intensity = min_intensity
        std_intensity = (max_intensity - min_intensity)

    else:
        mean_intensity = tgt_intensity_train.mean()
        std_intensity = tgt_intensity_train.std()
        tgt_intensity_train = (tgt_intensity_train - mean_intensity)/std_intensity
        tgt_intensity_test = (tgt_intensity_test - mean_intensity)/std_intensity

    #Standardize vision data
    if args.normalize:
        maxs = x_viz_train.permute(0,1,3,4,2).reshape(-1, 9).max(dim=0).values
        mins = x_viz_train.permute(0,1,3,4,2).reshape(-1, 9).min(dim=0).values

        maxs_stat = x_stat_train.reshape(-1, 7).max(dim=0).values
        mins_stat = x_stat_train.reshape(-1, 7).min(dim=0).values

        for i in range(len(maxs)):
            x_viz_train[:, :, i] = (x_viz_train[:, :, i] - mins[i]) / (maxs[i] - mins[i])
            x_viz_test[:, :, i] = (x_viz_test[:, :, i] - mins[i]) / (maxs[i] - mins[i])

        for i in range(len(maxs_stat)):
            x_stat_train[:, :, i] = (x_stat_train[:, :, i] - mins_stat[i]) / (maxs_stat[i] - mins_stat[i])
            x_stat_test[:, :, i] = (x_stat_test[:, :, i] - mins_stat[i]) / (maxs_stat[i] - mins_stat[i])
    else:
        means = x_viz_train.mean(dim=(0, 1, 3, 4))
        stds = x_viz_train.std(dim=(0, 1, 3, 4))

        means_stat = x_stat_train.mean(dim=(0, 1))
        stds_stat = x_stat_train.std(dim=(0, 1))

        for i in range(len(means)):
            x_viz_train[:, :, i] = (x_viz_train[:, :, i] - means[i]) / stds[i]
            x_viz_test[:, :, i] = (x_viz_test[:, :, i] - means[i]) / stds[i]

        for i in range(len(means_stat)):
            x_stat_train[:, :, i] = (x_stat_train[:, :, i] - means_stat[i]) / stds_stat[i]
            x_stat_test[:, :, i] = (x_stat_test[:, :, i] - means_stat[i]) / stds_stat[i]

    if args.test_nostat:
        n, t, _ = x_stat_train.shape
        x_stat_train = torch.zeros((n,t,1))
        x_stat_test = torch.zeros((x_stat_test.shape[0], t, 1))

    train_tensors = [x_viz_train, x_stat_train, tgt_intensity_cat_train, tgt_intensity_cat_baseline_train, tgt_displacement_train, tgt_intensity_train]
    test_tensors = [x_viz_test, x_stat_test, tgt_intensity_cat_test, tgt_intensity_cat_baseline_test, tgt_displacement_test, tgt_intensity_test]

    train_ds = TensorDataset(*train_tensors)
    test_ds = TensorDataset(*test_tensors)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Create model
    encoder_config = setup.encoder_config  # Load pre-defined config
    encoder = models.CNNEncoder(n_in=3 * 3,
                                n_out=128,
                                hidden_configuration=encoder_config)

    if args.encdec:
        decoder_config = setup.decoder_config
        if args.target_intensity_cat:
            # 7 classes of storms if categorical
            n_out_decoder = 7
        else:
            # if target intensity then 1 value to predict
            n_out_decoder = 2 - args.target_intensity
        # n_in decoder must be out encoder + 9 because we add side features!
        model = models.ENCDEC(n_in_decoder=128 + x_stat_train.shape[2],
                              n_out_decoder=n_out_decoder,
                              encoder=encoder,
                              hidden_configuration_decoder=decoder_config,
                              window_size=args.window_size)

    else:
        model = models.LINEARTransform(encoder, args.window_size, target_intensity=args.target_intensity, \
                                       target_intensity_cat=args.target_intensity_cat)
        decoder_config = None

    # Add Tensorboard
    model = model.to(device)

    print("Loading model", model)

    model.load_state_dict(torch.load(osp.join(args.output_dir, 'best_model.pt')))
    model.eval()


if __name__ == "__main__":
    import setup
    global best_accuracy
    best_accuracy = 0
    args = setup.create_setup()
    # print(vars(args))
    setup.create_seeds()
    main(args)









