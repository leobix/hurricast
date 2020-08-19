import torch
import math

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
#==============================
#GENERAL UTILS
def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def compute_l2(model):
    L2 = 0.
    for name, p in model.named_parameters():
        if 'weight' in name:
            L2 += (p**2).sum()
    return L2


def move_to_device(D: dict, device: torch.device):
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

#==============================
#TENSORBOARD UTILS
def _write_histograms(writer, preds, targets, epoch_number, mode):
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


def _write_metrics(writer, metrics_dict, global_step, mode):
    for name, item in metrics_dict.items():
        try:
            writer.add_scalar(name,
                          item,
                          global_step)
        except Exception as e:
            #print('Could not write {} due to {}'.format(name, e))
            pass


def _write_list(writer, loss_list, global_step, name):
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
        "lr", 
        "l2_reg",
        "batch_size",
        "lr", 
        "mode", 
        "n_epochs"]

    if hparam_dict is None:
        hparam_dict = {}
        for param in hparam_list:
            try:
                hparam_dict[param] = getattr(args, param)
                #print(param, type(hparam_dict[param]), hparam_dict[param])
            except Exception as e:
                print('Problem', e)
                hparam_dict[param] = 'error saving'
    writer.add_hparams(hparam_dict, metric_dict)


def update_writer(writer,
                  epoch: int,
                  metrics_dict: dict,
                  train_losses: list,
                  train_loss: list, valid_loss: list,
                  labels: list, preds: list,
                  N_train: int, N_eval: int,
                  mode: str):

        _write_list(writer, train_losses, epoch, 'Training loss')
        #write_list(writer, valid_losses, epoch, 'Eval loss')

        _write_histograms(writer, preds, labels, epoch, mode)

        _write_metrics(writer, metrics_dict, epoch, mode)

        writer.add_scalar('Train loss (per epoch)',
                          train_loss, epoch * N_train)
        writer.add_scalar('Eval loss (per epoch)',
                          valid_loss, epoch)


def update_post_train_writer(writer,
                             args, 
                             test_metrics, 
                             hparam_metric_dict, 
                             hparam_dict=None):
    write_hparams(
        writer, args,
        hparam_dict=hparam_dict,
        metric_dict={k: v for k, v in test_metrics.items()
                     if isinstance(v, (float, int))}
    )
    writer.add_text('classification_report', test_metrics['classification_report'])  #,
                                                   #'Val_Loss': float(best_valid_loss),
                       #                            'Valid_Acc': float(best_valid_acc)})
    
#========================
#MODEL UTILS
def get_pred_fn(task='classification'):
    def _get_pred_classification(x_out):
        if len(x_out.size()) > 1:
            return x_out.argmax(-1)
        else:
            return x_out
    assert task in accepted_tasks
    get_pred = _get_pred_classification if task == 'classification' \
        else lambda z: z
    return get_pred

#===========================
# Training Stats and other equivalents
def update_all_stats(
    all_stats, epoch_stats):
    """
    Update/create a dictionary.
    """
    #Loop through the new dict
    for k, v in epoch_stats.items():
        if k not in all_stats.keys():
            all_stats[k] = []
        if isinstance(v, list):
            all_stats[k].extend(v)
        elif isinstance(v, (float, int)):#, np.float32, np.float64)):
            all_stats[k].append(v)
        else: #If we have str for instanc 
            pass
    return all_stats


def logging_message(epoch, start_time, end_time, train_loss, valid_loss, **kwargs):
            
    epoch_mins, epoch_secs = elapsed_time(start_time, end_time)
            
    print(
                f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    
    print(
                f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.3f}')
    
    print(
                f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    if 'f1_micro' in kwargs and 'f1_macro' in kwargs:
        try:
            val_acc = kwargs['accuracy']
            val_f1micro = kwargs['f1_micro']
            val_f1macro = kwargs['f1_macro']
            print(
                    f'\t Val. F1 Macro: {val_f1macro:.3f} |  Val. F1 Micro: {val_f1micro:.3f}')
        except:
            pass
            