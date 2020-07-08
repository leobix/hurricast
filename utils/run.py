import sys
sys.path.append('../')
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
from utils import utils_vision_data, data_processing, plot, models
import matplotlib.pyplot as plt
import tqdm
import time
import os 
import os.path as osp
from sklearn.metrics import f1_score, accuracy_score
#===============================
accepted_modes = (
    'intensity',
    'displacement',
    'intensity_cat',
    'baseline_intensity_cat',
    'baseline_displacement'
    )

accepted_tasks = (
    'regression', 
    'classification'
)

#task: regression
# train_loss_fn = create_loss_fn(task)
# test_loss_fn = train_loss_fn
# metrics_funcs = [regression_metrics]


# task: classification
# train_loss_fn = create_loss_fn(task)
# test_loss_fn = accuracy_score
# metrics_funcs = [classification_metrics]
#================================
# Utils
def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def compute_l2(model):
    L2 = 0.
    for name, p in model.named_parameters():
        if 'weight' in name:
            L2 += (p**2).sum()
    return L2


def move_to_device(D:dict, device: torch.device):
    """
    Move a dict/list of tensors to the correct device.
    """
    if isinstance(D, dict):
        return {k: (x.to(device) if isinstance(x, torch.Tensor)
                    else x) for k, x in D.items()}
    elif isinstance(D, list):
        return [(x.to(device) if isinstance(x, torch.Tensor)
                 else x) for x in D]
    else:
        raise TypeError('Need a dict or a list to use move_to_device')


def elapsed_time(start_time, end_time):
    elapse_time = end_time - start_time
    elapsed_mins = int(elapse_time / 60)
    elapsed_secs = int(elapse_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#=============
# Metrics, loss fn
def create_loss_fn(task='classification'):
    """
    Wrappers that uses same signature for all loss_functions.
    Can easily add new losses function here.
    #TODO: Théo--> See if we can make an entropy based loss.
    """
    assert task in ['classification',
                    'regression'], "The\
    prediction function needs either the classification or\
    regression flag."
    
    base_loss_fn = nn.MSELoss()
    base_classification_loss_fn = nn.CrossEntropyLoss()
    if task == 'classification':
        return base_loss_fn
    else:
        return base_classification_loss_fn


def create_eval_loss_fn(task='classification'):
    assert task in ['classification',
                    'regression']
    if task == 'classification':
        return lambda x,y : 1 - accuracy_score(x,y)
    else:
        return nn.MSELoss()


def create_eval_metrics_fn(task='classification'):
    assert task in ['classification',
                   'regression']
    if task == 'classification':
        return _classification_metrics
    else:
        return _regression_metrics


def _classification_metrics(model_outputs, target):
    """
    Compute accuracy metrics and add to writer.

    Parameters:
    ----------
    model_outputs: pre-softmax prediction
    target: correct class

    Out:
    ---------
    out: dict - contains f1_micro/macro/predicted_class and accuracy

    """
    def get_pred(x_out):
        if len(x_out.size()) > 2:
            return x_out.argmax(-1)
        else:
            return x_out

    class_pred = get_pred(model_outputs)
    f1_micro = f1_score(target, class_pred, average='micro')
    f1_macro = f1_score(target, class_pred, average='macro')
    accuracy = accuracy_score(target, class_pred)
    n_tokens = len(target)
    out = {
        'accuracy': accuracy,
        'class_pred': class_pred,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'n_tokens': n_tokens
    }
    return out


def _regression_metrics(model_outputs, target):
    """
    Compute accuracy metrics and add to writer.

    Parameters:
    ----------
    model_outputs: pre-softmax prediction
    target: correct class

    Out:
    ---------
    out: dict - contains avg L2 loss, total loss and number
                of tokens
    """
    n_tokens = len(target)
    loss_fn = nn.MSELoss()
    avg_loss = loss_fn(model_outputs, target)
    total_loss = n_tokens * avg_loss
    out = {
        'avg_loss': avg_loss,
        'total_loss': total_loss,
        'n_tokens': n_tokens
    }
    return out

#===========================
# Eval
#NEW
def get_predictions(model, iterator, task='classification', return_pt=True):
    def get_pred_classification(x_out):
        if len(x_out.size()) > 2:
            return x_out.argmax(-1)
        else:
            return x_out

    assert task in ['classification', 
                    'regression'], "The\
    prediction function needs either the classification or\
    regression flag."

    get_pred = get_pred_classification if task == 'classification' \
        else lambda z: z
    
    device = next(model.parameters()).device
    model.eval()
    preds, true_preds = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x_batch, y_batch = map(lambda x: move_to_device(x, device), batch)
            #x_batch = remove_trg(x_batch)
            out = model(**x_batch) 
            #(bs, N_dim)
            preds.extend(get_pred(out.cpu()).tolist())
            true_preds.extend(y_batch['trg_y'].cpu().tolist())
    
    if return_pt:
        preds = torch.stack(preds)
        true_preds = torch.stack(true_preds)
    model.train()
    return preds, true_preds


def evaluate(model: nn.Module, 
            iterator:torch.utils.data.DataLoader, 
            loss_fn: callable, 
            metrics_funcs: callable) -> (torch.Tensor, torch.Tensor, float, dict):
    #1. Get the predictions (moved to CPU)
    preds, true_preds = get_predictions(model, iterator, return_pt=True)
    #2. Get list of metrics that we want
    #out_metrics = []
    #for metric_func in metrics_funcs:
    #    out_metric = metric_func(preds, true_preds)
    #    out_metrics.append(out_metric)
    out_metrics = metric_func(preds=preds, target=true_preds)
    out_loss = loss_fn(x=preds, y=true_preds)
    return preds, true_preds, out_loss, out_metrics
    
#================================
#Train
#NEW
def train_epoch(model, train_iterator, optimizer, loss_fn, l2_reg, clip):
    device = next(model.parameters()).device
    model.train()
    epoch_loss = 0.
    train_losses = []
    for i, batch in enumerate(tqdm.tqdm(train_iterator, desc='Inner Training loop')):
        x_batch, y_batch = map(lambda x: move_to_device(x, device), batch)
        optimizer.zero_grad()
        out = model(**x_batch)
        loss = loss_fn(x=out, y=y_batch['trg_y'])
        assert_no_nan_no_inf(loss)
        #L2 REG
        if l2_reg > 0.:
            L2 = compute_l2(model)
            loss += 2./y_batch['trg_y'].size(0) * l2_reg * L2
            assert_no_nan_no_inf(loss)
        loss.backward()
        #Gradient clipping
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()  # need to do that ?
        train_losses.append(loss.item())

    return model, optimizer, epoch_loss / len(train_iterator), train_losses

#NEW
def train(model,
          optimizer,
          num_epochs,
          train_loss_fn,
          test_loss_fn,
          metrics_fn,
          train_iterator,
          val_iterator,
          test_iterator,
          mode,
          clip=None,
          l2_reg=0.,
          save=False,
          args={},
          none_idx=None,
          writer=None):

    #def _create_stats():
    #    return {
    #        'train_losses': [], 'valid_losses': [],
    #        'valid_accuracies': [], 'test_accuracies': [],
    #        'train_losses_gran': [], 'valid_losses_gran': []
    #        }

    #def _update_stats(stats, train_loss,
    #                  valid_loss, train_losses,
    #                  valid_losses, valid_acc):
    #    stats['train_losses'].append(train_loss)
    #    stats['valid_losses'].append(valid_loss)
    #    stats['train_losses_gran'].extend(train_losses)
    #    stats['valid_losses_gran'].extend(valid_losses)
    #    stats['valid_accuracies'].append(valid_acc)
    #    return stats

    #TODO: Double check if N_train or N_eval that we need.
    def _update_writer(writer, epoch: int, 
                    metrics_dict: dict,
                    train_losses: list, 
                    train_loss: list, valid_loss: list,
                    labels: list, preds: list,
                    N_train: int, N_eval: int, 
                    mode: str):
        
        write_list(writer, train_losses, epoch, 'Training loss')
        #write_list(writer, valid_losses, epoch, 'Eval loss')

        write_histograms(writer, preds, labels, epoch, mode)

        write_metrics(writer, metrics_dict, epoch, mode)

        writer.add_scalar('Train loss (per epoch)',
                          train_loss, epoch * N_train)
        writer.add_scalar('Eval loss (per epoch)',
                          valid_loss, epoch)
        
    #======================
    #Begin train
    #stats = _create_stats()
    best_valid_loss = float('inf')
    model.train()
    #Train
    loop = tqdm.trange(num_epochs, desc='Epochs')
    start_time = time.time()
    for epoch in loop:
        model, optimizer, train_loss,\
        train_losses = train_epoch(model=model,
                                    train_iterator=train_iterator,
                                    optimizer=optimizer,
                                    loss_fn=train_loss_fn,
                                    l2_reg=l2_reg,
                                    clip=clip)
        #Eval: 
        preds, true_preds, \
            valid_loss, eval_metrics = evaluate(model, val_iterator,
                                                loss_fn=test_loss_fn,
                                                metrics_funcs=metrics_fn)
    
        end_time = time.time()
        epoch_mins, epoch_secs = elapsed_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            if save:
                torch.save(model.state_dict(), args.output_dir +
                           '/model-best.pt')
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)

        if writer is not None:
            _update_writer(
                    writer=writer, epoch=epoch, 
                    metrics_dict=eval_metrics,
                    train_losses=train_losses,
                    train_loss=train_loss, 
                    valid_loss=valid_loss,
                    labels=true_preds, preds=preds,
                    N_train=len(train_iterator),
                    N_eval=len(val_iterator),
                    mode=mode)

        #Logging
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
        loop.set_description('Epoch {} | Loss {}'.format(epoch,
                                                         valid_loss))

    #=====================================
    # Test it
    #model.load_state_dict(torch.load('copy_task-best.pt'))
    #test_acc, test_correct, test_toks, \
    #    true_preds, preds = test(
    #        best_model, test_iterator, test_loss_fn)

    test_preds, test_labels, \
            test_loss, eval_metrics = evaluate(best_model, val_iterator,
                                                loss_fn=test_loss_fn,
                                                metrics_funcs=metrics_fn)
    print(
        f'\t Final test ACC {test_loss:.3f}')

    #stats['final_test_accuracy'] = test_acc
    #if writer is not None:
    #    utils.write_hparams(args.writer, args,
    #                        hparam_dict=None,
    #                        metric_dict={
    #                            **{k: v[0] for k, v in best_class_report.items()},
    #                            'Val_Loss': float(best_valid_loss),
    #                            'Valid_Acc': float(best_valid_acc)})
        #utils.write_model(args.writer, args, model, train_iterator)

    return best_model, optimizer, stats


#==============================
#TENSORBOARD UTILS
#NEW
def write_histograms(writer, preds, targets, epoch_number, mode):
    """
    preds: (N, 1) or (N,2)
    targets:  
    """
    assert mode in accepted_modes
    if mode in ('displacement',  'baseline_displacement'):
        targets = torch.norm(targets, p=2, dim=1)
        preds = torch.norm(preds, p=2, dim=1)

    writer.add_histogram("Distribution of targets ({})".format(mode),
                         targets, global_step=epoch_number)
    writer.add_histogram("Distribution of preds ({})".format(mode),
                         preds, global_step=epoch_number)

#NEW
def write_metrics(writer, metrics_dict, global_step, mode):
    for name, item in metrics_dict.items():
        writer.add_scalar(name,
                        item, 
                        global_step)


def write_list(writer, loss_list, global_step, name):
    #list -> = train_loss actually 
    #global_sterp
    #epoch --> needs 
    N = len(loss_list)
    for i, item in enumerate(loss_list):
        try:
            writer.add_scalar(name,
                        item, 
                        global_step * N + i)
        except:
            writer.add_scalar(name,
                              item.item(),
                              global_step * N + i)


def write_hparams(writer, args, metric_dict, hparam_dict=None):    
    hparam_list = [
        "lr", "l2_reg",
        "batch_size", 
        "l2_reg", 
        "lr", "model", "num_epochs"]

    if hparam_dict is None:
        hparam_dict = {}
        for param in hparam_list:
            try:
                hparam_dict[param] = getattr(args, param)
                #print(param, type(hparam_dict[param]), hparam_dict[param])
            except Exception as e:
                print('Problem', e)
                hparam_dict[param] = 'error saving'
    #for k, v in metric_dict.items():
    #    print(k,v, type(v))
    #print('HPARAM is', hparam_dict, metric_dict)
    writer.add_hparams(hparam_dict, metric_dict)

#================================
#Old func
#TODO: Clean Up and Remove
def train_old(model,
              optimizer,
        loss_fn, 
        n_epochs,
        train_loader,
        test_loader,
        args,
        writer, 
        scheduler=None,
        l2_reg=0., 
        device=torch.device('cpu'),
        target_intensity=False,
        target_intensity_cat=False):
    """
    Train function. 
    Expect a loss based on one of the following targest
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
            #Put data on GPU
            data_batch = tuple(map(lambda x: x.to(device), 
                                    data_batch)) #careful because 
            x_viz, x_stat,\
            tgt_intensity_cat,\
            tgt_intensity_cat_baseline,\
            tgt_displacement_baseline,\
            tgt_displacement, tgt_intensity = data_batch

            optimizer.zero_grad()

            model_outputs = model(x_viz, x_stat)
            if target_intensity:
                target = tgt_intensity
            elif target_intensity_cat:
                target = tgt_intensity_cat
            else:
                target = tgt_displacement

            batch_loss = loss_fn(
                model_outputs, tgt_displacement, tgt_intensity, tgt_intensity_cat)

            assert_no_nan_no_inf(batch_loss)
            if l2_reg > 0:
                L2 = 0.
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        L2 += (p**2).sum()
                batch_loss += 2./x_viz.size(0) * l2_reg * L2
                assert_no_nan_no_inf(batch_loss)

            training_loss.append(batch_loss.item())
            writer.add_scalar('training loss',
                              batch_loss.item(),
                              epoch * len(train_loader) + i)
            if target_intensity_cat:
                class_pred = torch.softmax(model_outputs, dim=1).argmax(dim=1)
                f1_micro = f1_score(target, class_pred, average='micro')
                f1_macro = f1_score(target, class_pred, average='macro')
                #f1_all =  f1_score(target, class_pred, average = None)
                accuracy = accuracy_score(target, class_pred)
                f1_micro_baseline = f1_score(target, tgt_intensity_cat_baseline, average='micro')
                f1_macro_baseline = f1_score(target, tgt_intensity_cat_baseline, average='macro')
                accuracy_baseline = accuracy_score(target, tgt_intensity_cat_baseline)
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
                #TODO Make the log clean, there is one value of f1 for each class
                #writer.add_histogram('f1_all_scores',
                                  #f1_all,
                                  #epoch * len(train_loader) + i)
            batch_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            inner_loop.set_description('Epoch {} | Loss {}'.format(epoch,
                                                         batch_loss.item()))
        
        _, eval_loss_sample, _ = eval(model,
                                loss_fn=loss_fn,
                                test_loader=test_loader,
                                writer=writer,
                                epoch_number=epoch,
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

#TODO: Clean Up and Remove
def main_old(args):
    #Prepare device
    device = torch.device(
        f'cuda:{args.gpu_nb}' if torch.cuda.is_available() and args.gpu_nb !=-1 else 'cpu')
    print(' Prepare the training using ', device)
    #Load files and reformat.
    vision_data = np.load(osp.join(args.data_dir, args.vision_name), allow_pickle = True) #NUMPY ARRAY
    y = np.load(osp.join(args.data_dir, args.y_name), allow_pickle = True)
   
    train_tensors, test_tensors = Prepro.process(vision_data, 
                                y, 
                                args.train_test_split,
                                predict_at=args.predict_at, 
                                window_size=args.window_size)
    train_ds = TensorDataset(*train_tensors)
    test_ds = TensorDataset(*test_tensors)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Create model
    encoder_config = setup.encoder_config #Load pre-defined config
    encoder = models.CNNEncoder(n_in=3*3,
                                n_out=128,
                                hidden_configuration=encoder_config)

    if args.encdec:
        decoder_config = setup.decoder_config
        if args.target_intensity_cat:
            #7 classes of storms if categorical
            n_out_decoder = 7
        else:
            # if target intensity then 1 value to predict
            n_out_decoder = 2 - args.target_intensity
        # n_in decoder must be out encoder + 9 because we add side features!
        model = models.ENCDEC(n_in_decoder=128+10,
                                n_out_decoder=n_out_decoder,
                                encoder=encoder,
                                hidden_configuration_decoder=decoder_config,
                                window_size=args.window_size)
    #TODO:Uncomment
    elif args.transformer:
        #ecoder_config = setup.decoder_config
        decoder_config = setup.transformer_config
        if args.target_intensity_cat:
            #7 classes of storms if categorical
            n_out_decoder = 7
        else:
            # if target intensity then 1 value to predict
            n_out_decoder = 2 - args.target_intensity
        # n_in decoder must be out encoder + 9 because we add side features!
        model = models.TRANSFORMER(encoder,
                                   n_in_decoder=128+10,
                                   n_out_transformer=128,
                                   n_out_decoder=n_out_decoder,
                                   hidden_configuration_decoder=decoder_config,
                                   window_size=8)
    

    else:
        model = models.LINEARTransform(encoder, args.window_size, target_intensity=args.target_intensity, \
                                       target_intensity_cat=args.target_intensity_cat)
        decoder_config = None

    #Add Tensorboard    
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
                                scheduler=None,
                                l2_reg=args.l2_reg,
                                target_intensity = args.target_intensity,
                                target_intensity_cat = args.target_intensity_cat)
    plt.plot(loss)
    plt.title('Training loss')
    plt.show()

    if args.save:
        torch.save(model.state_dict(), osp.join(args.output_dir, 'final_model.pt'))

#TODO: Clean Up and Remove
def eval_old(model,
             loss_fn,
             test_loader,
             writer,
             epoch_number,
             target_intensity=False,
             target_intensity_cat=False,
             device=torch.device('cpu')):
    """
    #TODO: Comment a bit    
    #TODO: Check the eval loss working with intensity. 
    """
    # set model in training mode
    model.eval()
    torch.manual_seed(0)
    # train model
    total_loss = 0.
    total_n_eval = 0.
    baseline_loss_eval = 0.
    accuracy, accuracy_baseline = 0., 0.
    f1_micro, f1_micro_baseline = 0., 0.
    f1_macro, f1_macro_baseline = 0., 0.
    baseline_disp_eval = []
    loop = tqdm.tqdm(test_loader, desc='Evaluation')
    # Get a dict of lists for tensorboard
    tgts = {'d': [], 'i': [], 'i_cat': []}
    #preds = {'i': [] } if target_intensity or target_intensity_cat else {'d': [] }
    if target_intensity:
        preds = {'i': []}
    elif target_intensity_cat:
        preds = {'i_cat': []}
    else:
        preds = {'d': []}
    for data_batch in loop:
        #Put data on GPU
        data_batch = tuple(map(lambda x: x.to(device),
                               data_batch))

        x_viz, x_stat, \
            tgt_intensity_cat, tgt_intensity_cat_baseline,\
            tgt_displacement_baseline, \
            tgt_displacement, tgt_intensity = data_batch

        with torch.no_grad():
            model_outputs = model(x_viz, x_stat)
            if target_intensity:
                target = tgt_intensity
            elif target_intensity_cat:
                target = tgt_intensity_cat
            else:
                target = tgt_displacement

            batch_loss = loss_fn(
                model_outputs, tgt_displacement, tgt_intensity, tgt_intensity_cat)
            assert_no_nan_no_inf(batch_loss)
            # Do we divide by the size of the data
            total_loss += batch_loss.item() * tgt_intensity.size(0)
            total_n_eval += tgt_intensity.size(0)

            if target_intensity_cat:
                # Theo: Same as torch argmax directly.
                class_pred = torch.softmax(model_outputs, dim=1).argmax(dim=1)
                accuracy += accuracy_score(target, class_pred)
                f1_micro += f1_score(target, class_pred, average='micro')
                f1_macro += f1_score(target, class_pred, average='macro')
                accuracy_baseline += accuracy_score(
                    target, tgt_intensity_cat_baseline)
                f1_micro_baseline += f1_score(target,
                                              tgt_intensity_cat_baseline, average='micro')
                f1_macro_baseline += f1_score(target,
                                              tgt_intensity_cat_baseline, average='macro')
            else:
                if not target_intensity:
                    baseline_loss_eval += (tgt_displacement_baseline.size(0)*loss_fn(
                        tgt_displacement_baseline, tgt_displacement, tgt_intensity, tgt_intensity_cat)
                    )

            #Keep track of the predictions/targets
            tgts['d'].append(tgt_displacement)
            tgts['i'].append(tgt_intensity)
            tgts['i_cat'].append(tgt_intensity_cat.float())
            if not target_intensity_cat:
                preds.get(tuple(preds.keys())[0]).append(model_outputs)
            else:
                print('cat', class_pred)
                preds.get(tuple(preds.keys())[0]).append(class_pred.float())

    #Re-defined the losses
    tgts = {k: torch.cat(v) for k, v in tgts.items()}
    preds = {k: torch.cat(v) for k, v in preds.items()}
    #=====================================
    #Compute norms, duck type and add to board.
    tgts['d'] = torch.norm(tgts['d'], p=2, dim=1)
    writer.add_histogram("Distribution of targets (displacement)",
                         tgts['d'], global_step=epoch_number)
    writer.add_histogram("Distribution of targets (intensity)",
                         tgts['i'], global_step=epoch_number)
    writer.add_histogram("Distribution of targets (intensity cat)",
                         tgts['i_cat'], global_step=epoch_number)
    try:
        preds['d'] = torch.norm(preds['d'], p=2, dim=1)
        log = "Distribution of predictions (displacement)"
        writer.add_histogram(log, preds['d'], global_step=epoch_number)

    except:
        if target_intensity:
            log = "Distribution of predictions (intensity)"
            writer.add_histogram(log, preds['i'], global_step=epoch_number)
        else:
            log = "Distribution of predictions (intensity cat)"
            writer.add_histogram(log, preds['i_cat'], global_step=epoch_number)

    writer.add_scalar('total_eval_loss',
                      total_loss,
                      epoch_number)
    writer.add_scalar('avg_eval_loss',
                      total_loss/float(total_n_eval),
                      epoch_number)
    if target_intensity_cat:
        writer.add_scalar('accuracy_eval',
                          accuracy.item()/len(loop),
                          epoch_number)
        writer.add_scalar('f1_micro_eval',
                          f1_micro.item()/len(loop),
                          epoch_number)
        writer.add_scalar('f1_macro_eval',
                          f1_macro.item()/len(loop),
                          epoch_number)
        writer.add_scalar('accuracy_eval_baseline',
                          accuracy_baseline.item() / len(loop),
                          epoch_number)
        writer.add_scalar('f1_micro_eval_baseline',
                          f1_micro_baseline.item() / len(loop),
                          epoch_number)
        writer.add_scalar('f1_macro_eval_baseline',
                          f1_macro_baseline.item() / len(loop),
                          epoch_number)
    else:
        if not target_intensity:
            writer.add_scalar('baseline_disp_eval_loss',
                              baseline_loss_eval/float(total_n_eval),
                              epoch_number)

    model.train()
    return model, total_loss, total_n_eval


if __name__ == "__main__":
    import setup
    args = setup.create_setup()
    setup.create_seeds()
    main_old(args)

    




    


