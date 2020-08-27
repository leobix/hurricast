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

def create_loss_fn(mode='intensity'):
    """
    Wrappers that uses same signature for all loss_functions.
    Can easily add new losses function here.
    #TODO: Théo--> See if we can make an entropy based loss.

    """
    assert mode in ['displacement',
                    'intensity', 'intensity_cat']  # , 'sum']

    base_loss_fn = nn.MSELoss()
    base_classification_loss_fn = nn.CrossEntropyLoss()

    def displacement_loss(model_outputs,
                          target_displacement,
                          target_intensity,
                          target_intensity_cat):
        return base_loss_fn(model_outputs,
                            target_displacement)

    def intensity_loss(model_outputs,
                       target_displacement,
                       target_intensity,
                       target_intensity_cat):
        return base_loss_fn(model_outputs,
                            target_intensity)

    def intensity_cat_loss(model_outputs,
                           target_displacement,
                           target_intensity,
                           target_intensity_cat):
        return base_classification_loss_fn(model_outputs,
                                           target_intensity_cat)

    losses_fn = dict(displacement=displacement_loss,
                     intensity=intensity_loss,
                     intensity_cat=intensity_cat_loss)
    return losses_fn[mode]


def create_model():
    # TODO
    raise NotImplementedError


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def eval(model,
         loss_fn,
         test_loader,
         writer,
         args,
         epoch_number,
         mean_intensity,
         std_intensity,
         target_intensity=False,
         target_intensity_cat=False,
         device=torch.device('cpu')):
    """
    #TODO: Comment a bit
    """
    # set model in training mode
    model.eval()
    torch.manual_seed(0)
    # train model
    total_loss = 0.
    total_n_eval = 0.
    accuracy, accuracy_baseline = 0., 0.
    f1_micro, f1_micro_baseline = 0., 0.
    f1_macro, f1_macro_baseline = 0., 0.
    mae = 0
    loop = tqdm.tqdm(test_loader, desc='Evaluation')
    tgts = {'d': [], 'i': []}  # Get a dict of lists for tensorboard
    preds = {'i': []} if args.target_intensity else {'d': []}
    for data_batch in loop:
        # Put data on GPU
        data_batch = tuple(map(lambda x: x.to(device),
                               data_batch))
        x_viz, x_stat, tgt_intensity_cat, tgt_intensity_cat_baseline, tgt_displacement, tgt_intensity = data_batch
        with torch.no_grad():
            model_outputs = model(x_viz, x_stat)
            if target_intensity:
                target = tgt_intensity
            elif target_intensity_cat:
                target = tgt_intensity_cat
            else:
                target = tgt_displacement
            target = target.cpu()
            batch_loss = loss_fn(model_outputs, tgt_displacement, tgt_intensity, tgt_intensity_cat)
            assert_no_nan_no_inf(batch_loss)
            total_loss += batch_loss.item()  # Do we divide by the size of the data
            total_n_eval += tgt_intensity.size(0)

            if target_intensity_cat:
                class_pred = torch.softmax(model_outputs, dim=1).argmax(dim=1).detach().cpu().numpy()
                accuracy += accuracy_score(target, class_pred)
                f1_micro += f1_score(target, class_pred, average='micro')
                f1_macro += f1_score(target, class_pred, average='macro')
                accuracy_baseline += accuracy_score(target, tgt_intensity_cat_baseline.cpu().numpy())
                f1_micro_baseline += f1_score(target, tgt_intensity_cat_baseline.cpu().numpy(), average='micro')
                f1_macro_baseline += f1_score(target, tgt_intensity_cat_baseline.cpu().numpy(), average='macro')

            elif target_intensity:
                mae += mean_absolute_error(target*std_intensity + mean_intensity, model_outputs.cpu()*std_intensity + mean_intensity)

            # Keep track of the predictions/targets
            tgts['d'].append(tgt_displacement)
            tgts['i'].append(tgt_intensity)
            preds.get(tuple(preds.keys())[0]).append(model_outputs)

    tgts = {k: torch.cat(v) for k, v in tgts.items()}
    preds = {k: torch.cat(v) for k, v in preds.items()}
    # =====================================
    # Compute norms, duck type and add to board.
    tgts['d'] = torch.norm(tgts['d'], p=2, dim=1)
    writer.add_histogram("Distribution of targets (displacement)",
                         tgts['d'], global_step=epoch_number)
    writer.add_histogram("Distribution of targets (intensity)",
                         tgts['i'], global_step=epoch_number)
    try:
        preds['d'] = torch.norm(preds['d'], p=2, dim=1)
        log = "Distribution of predictions (displacement)"
        writer.add_histogram(log, preds['d'], global_step=epoch_number)
    except:
        log = "Distribution of predictions (intensity)"
        writer.add_histogram(log, preds['i'], global_step=epoch_number)

    writer.add_scalar('total_eval_loss',
                      total_loss,
                      epoch_number)
    writer.add_scalar('avg_eval_loss',
                      total_loss / float(total_n_eval),
                      epoch_number)
    if target_intensity:
        mae_eval = mae.item() / len(loop)
        writer.add_scalar('mae_eval',
                          mae_eval,
                          epoch_number)
        print("\n MAE Eval is: ", mae_eval)

    elif target_intensity_cat:
        eval_accuracy = accuracy.item() / len(loop)
        writer.add_scalar('eval_accuracy',
                          eval_accuracy,
                          epoch_number)
        writer.add_scalar('eval_f1_micro',
                          f1_micro.item() / len(loop),
                          epoch_number)
        writer.add_scalar('eval_f1_macro',
                          f1_macro.item() / len(loop),
                          epoch_number)
        writer.add_scalar('eval_accuracy_baseline',
                          accuracy_baseline.item() / len(loop),
                          epoch_number)
        writer.add_scalar('eval_f1_micro_baseline',
                          f1_micro_baseline.item() / len(loop),
                          epoch_number)
        writer.add_scalar('eval_f1_macro_baseline',
                          f1_macro_baseline.item() / len(loop),
                          epoch_number)
        print("\n Accuracy Eval is: ", eval_accuracy)
        print("\n Accuracy Eval Baseline is: ", accuracy_baseline.item() / len(loop))

    else:
        pass

    model.train()
    return model, total_loss, total_n_eval


def train(model,
          optimizer,
          loss_fn,
          n_epochs,
          train_loader,
          test_loader,
          args,
          writer,
          mean_intensity,                                      
          std_intensity,
          scheduler=None,
          l2_reg=0.,
          device=torch.device('cpu'),
          target_intensity=False,
          target_intensity_cat=False):
    """
    #TODO: Comment a bit
    """
    # set model in training mode
    model.train()

    torch.manual_seed(0)

    # train model
    training_loss = []
    previous_best = np.inf
    loop = tqdm.trange(n_epochs, desc='Epochs')

    for epoch in loop:
        inner_loop = tqdm.tqdm(train_loader, desc='Inner loop')
        for i, data_batch in enumerate(inner_loop):
            # Put data on GPU
            data_batch = tuple(map(lambda x: x.to(device),
                                   data_batch))
            x_viz, x_stat, \
            tgt_intensity_cat, tgt_intensity_cat_baseline, tgt_displacement, tgt_intensity = data_batch
            optimizer.zero_grad()

            model_outputs = model(x_viz, x_stat)
            if target_intensity:
                target = tgt_intensity
            elif target_intensity_cat:
                target = tgt_intensity_cat
            else:
                target = tgt_displacement
            target = target.cpu().numpy()
            batch_loss = loss_fn(
                model_outputs, tgt_displacement, tgt_intensity, tgt_intensity_cat)

            assert_no_nan_no_inf(batch_loss)
            if l2_reg > 0:
                L2 = 0.
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        L2 += (p ** 2).sum()
                batch_loss += 2. / x_viz.size(0) * l2_reg * L2
                assert_no_nan_no_inf(batch_loss)

            training_loss.append(batch_loss.item())
            writer.add_scalar('training loss',
                              batch_loss.item(),
                              epoch * len(train_loader) + i)
            if target_intensity_cat:
                class_pred = torch.softmax(model_outputs, dim=1).argmax(dim=1).detach().cpu().numpy()
                f1_micro = f1_score(target, class_pred, average='micro')
                f1_macro = f1_score(target, class_pred, average='macro')
                # f1_all =  f1_score(target, class_pred, average = None)
                accuracy = accuracy_score(target, class_pred)
                f1_micro_baseline = f1_score(target, tgt_intensity_cat_baseline.cpu().numpy(), average='micro')
                f1_macro_baseline = f1_score(target, tgt_intensity_cat_baseline.cpu().numpy(), average='macro')
                accuracy_baseline = accuracy_score(target, tgt_intensity_cat_baseline.cpu().numpy())
                writer.add_scalar('accuracy_train',
                                  accuracy.item(),
                                  epoch * len(train_loader) + i)
                writer.add_scalar('f1_micro_train',
                                  f1_micro.item(),
                                  epoch * len(train_loader) + i)
                writer.add_scalar('f1_macro_train',
                                  f1_macro.item(),
                                  epoch * len(train_loader) + i)
                writer.add_scalar('accuracy_train_baseline',
                                  accuracy_baseline.item(),
                                  epoch * len(train_loader) + i)
                writer.add_scalar('f1_micro_train_baseline',
                                  f1_micro_baseline.item(),
                                  epoch * len(train_loader) + i)
                writer.add_scalar('f1_macro_train_baseline',
                                  f1_macro_baseline.item(),
                                  epoch * len(train_loader) + i)
                # TODO Make the log clean, there is one value of f1 for each class
                # writer.add_histogram('f1_all_scores',
                # f1_all,
                # epoch * len(train_loader) + i)
            batch_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            inner_loop.set_description('Epoch {} | Loss {}'.format(epoch,
                                                                   batch_loss.item()))

        _, eval_loss_sample, _ = eval(model,
                                      loss_fn=loss_fn,
                                      test_loader=test_loader,
                                      args = args,
                                      writer=writer,
                                      epoch_number=epoch,
                                      mean_intensity=mean_intensity,
                                      std_intensity=std_intensity,
                                      target_intensity=target_intensity,
                                      target_intensity_cat=target_intensity_cat,
                                      device=device)

        if eval_loss_sample < previous_best:
            previous_best = eval_loss_sample
            torch.save(model.state_dict(),
                       osp.join(args.output_dir, 'best_model.pt'))
        model.train()
        loop.set_description('Epoch {} | Loss {}'.format(epoch,
                                                         eval_loss_sample)
                             )
    return model, optimizer, training_loss


def main(args):
    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' Prepare the training using ', device)
    # Load files and reformat.

    x_stat_train = torch.Tensor(np.load('data/X_train_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                        allow_pickle=True).reshape(-1, args.window_size, 30)[:,:,:14])[:,-args.sub_window_size:].to(device)
    x_stat_test = torch.Tensor(np.load('data/X_test_stat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                       allow_pickle=True).reshape(-1, args.window_size, 30)[:,:,:14])[:,-args.sub_window_size:].to(device)
    x_viz_train = torch.Tensor(np.load('data/X_train_vision_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                       allow_pickle = True).reshape(-1, args.window_size, 9, 25, 25))[:,-args.sub_window_size:].to(device)
    x_viz_test = torch.Tensor(np.load('data/X_test_vision_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                      allow_pickle = True).reshape(-1, args.window_size, 9, 25, 25))[:,-args.sub_window_size:].to(device)
    if args.sub_area > 0:
        x_viz_train = x_viz_train[:,:,:,args.sub_area:-args.sub_area,args.sub_area:-args.sub_area]
        x_viz_test = x_viz_test[: , :, :, args.sub_area:-args.sub_area, args.sub_area:-args.sub_area]

    tgt_intensity_cat_train = torch.LongTensor(np.load('data/y_train_intensity_cat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                      allow_pickle=True)).to(device)
    tgt_intensity_cat_test = torch.LongTensor(np.load('data/y_test_intensity_cat_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                     allow_pickle=True)).to(device)

    tgt_intensity_train = torch.Tensor(np.load('data/y_train_intensity_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                  allow_pickle=True)).to(device)
    tgt_intensity_test = torch.Tensor(np.load('data/y_test_intensity_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                 allow_pickle=True)).to(device)

    tgt_intensity_cat_baseline_train = torch.LongTensor(np.load('data/y_train_intensity_cat_baseline_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',  allow_pickle = True))
    tgt_intensity_cat_baseline_test = torch.LongTensor(np.load('data/y_test_intensity_cat_baseline_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy', allow_pickle=True))

    tgt_displacement_train = torch.Tensor(np.load('data/y_train_displacement_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                     allow_pickle=True)).to(device)
    tgt_displacement_test = torch.Tensor(np.load('data/y_test_displacement_1980_34_20_120_w' + str(args.window_size) + '_at_' + str(args.predict_at) + '.npy',
                                    allow_pickle=True)).to(device)


    args.window_size = args.sub_window_size
    print("\n Window of prediction is: ", args.window_size)
    print("\n X ", x_stat_train[0, 0])
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

        maxs_stat = x_stat_train[:,:,:7].reshape(-1, 7).max(dim=0).values
        mins_stat = x_stat_train[:,:,:7].reshape(-1, 7).min(dim=0).values

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
    writer = setup.create_board(args, model, configs=[encoder_config, decoder_config])
    model = model.to(device)

    print("Using model", model)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.target_intensity_cat:
        loss_mode = 'intensity_cat'
    else:
        loss_mode = 'intensity' if args.target_intensity else 'displacement'
    loss_fn = create_loss_fn(loss_mode)

    model, optimizer, loss = train(model,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   n_epochs=args.n_epochs,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   args=args,
                                   writer=writer,
                                   mean_intensity=mean_intensity,
                                   std_intensity=std_intensity,
                                   scheduler=None,
                                   l2_reg=args.l2_reg,
                                   device = device,
                                   target_intensity=args.target_intensity,
                                   target_intensity_cat=args.target_intensity_cat)
    plt.plot(loss)
    plt.title('Training loss')
    plt.show()

    if args.save:
        torch.save(model.state_dict(), osp.join(args.output_dir, 'final_model.pt'))


if __name__ == "__main__":
    import setup
    global best_accuracy
    best_accuracy = 0
    args = setup.create_setup()
    # print(vars(args))
    setup.create_seeds()
    main(args)









