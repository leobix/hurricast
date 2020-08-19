import torch 
import tqdm
#import math
import time
import os 
from .utils import run as urun
import copy

#===========================
# Eval
def get_predictions(model, iterator, task='classification', return_pt=True):
    
    assert task in ['classification', 
                    'regression'], "The\
    prediction function needs either the classification or\
    regression flag."
    get_pred = urun.get_pred_fn(task)
    device = next(model.parameters()).device
    model.eval()
    preds, true_preds = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x_batch, y_batch = map(lambda x: urun.move_to_device(x, device), batch)
            #x_batch = remove_trg(x_batch)
            out = model(**x_batch) 
            #(bs, N_dim)
            preds.extend(get_pred(out.cpu()).tolist())
            true_preds.extend(y_batch['trg_y'].cpu().tolist())
    
    if return_pt:
        preds = torch.Tensor(preds)
        true_preds = torch.Tensor(true_preds)
    model.train()
    return preds, true_preds


def evaluate(model,
            iterator,
            task,
            loss_fn,
            metrics_func) -> (torch.Tensor, torch.Tensor, float, dict):
    #1. Get the predictions (moved to CPU)
    model.eval()
    preds, true_preds = get_predictions(model, iterator, task=task, return_pt=True)
    #2.
    out_metrics = metrics_func(preds=preds, target=true_preds)
    out_loss = loss_fn(preds, true_preds)
    return preds, true_preds, out_loss, out_metrics    

#TODO:Write it down
def evaluate_naive_baseline(iterator,
                            task,
                            loss_fn,
                            metrics_func):
    return None
    
#================================
#Train

def train_epoch(model, 
                train_iterator, 
                optimizer, 
                loss_fn, 
                l2_reg, 
                global_step, 
                task=None,
                return_pt=False,
                clip=None, 
                scheduler=None):
    """
    Loop through the data for one epoch. If task is not None, 
    we will get the predictions on the training data.
    """
    if task is not None: get_pred = urun.get_pred_fn(task)
    device = next(model.parameters()).device
    print('ON DEVICE', device)
    model.train()
    
    epoch_loss = 0.
    train_losses = []
    preds, true_preds = [], []

    inner_loop = tqdm.tqdm(train_iterator, desc='Inner Training loop')
    for i, batch in enumerate(inner_loop):
        x_batch, y_batch = map(lambda x: urun.move_to_device(x, device), batch)
        optimizer.zero_grad()
        out = model(**x_batch)
        loss = loss_fn(out, y_batch['trg_y'])
        urun.assert_no_nan_no_inf(loss)
        #L2 REG
        if l2_reg > 0.:
            L2 = urun.compute_l2(model)
            loss += 2./y_batch['trg_y'].size(0) * l2_reg * L2
            urun.assert_no_nan_no_inf(loss)
        loss.backward()
        #Gradient clipping
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()  # need to do that ?
        train_losses.append(loss.item())
        
        if task is not None:
            preds.extend(get_pred(out.detach().cpu()).tolist())
            true_preds.extend(y_batch['trg_y'].detach().cpu().tolist())

    if scheduler is not None:
        scheduler.step()
    
    inner_loop.set_description('Epoch {} | Loss {}'.format(global_step,
                                                         epoch_loss))
    if return_pt:
        preds = torch.Tensor(preds)
        true_preds = torch.Tensor(true_preds)

    return model, optimizer, epoch_loss / len(train_iterator), train_losses, preds, true_preds


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
          task,
          get_training_stats=False,
          clip=None,
          scheduler=None,
          l2_reg=0.,
          save=False,
          args={},
          output_dir=None,
          none_idx=None,
          writer=None):
    #======================
    #Begin train
    #Create the dictionaries that we will need to store the metrics.
    all_train_stats = {} 
    all_eval_stats = {} 
    best_valid_loss = -1. if task == 'classification' \
        else float('inf')
    if get_training_stats: get_training_stats = task
    
    #Start looping
    model.train()
    loop = tqdm.trange(num_epochs, desc='Epochs')
    start_time = time.time()
    for epoch in loop:
        
        # Train
        model, optimizer, train_loss,\
        train_losses, preds, true_preds=train_epoch(
                                    model=model,
                                    train_iterator=train_iterator,
                                    optimizer=optimizer,
                                    loss_fn=train_loss_fn,
                                    global_step=epoch,
                                    task=get_training_stats,
                                    return_pt=True,
                                    l2_reg=l2_reg,
                                    clip=clip,
                                    scheduler=scheduler)
        # Get metrics and update the main dict.
        train_metrics = metrics_fn(preds, true_preds)
        all_train_stats = urun.update_all_stats(
                    all_train_stats, 
                    {'train_loss': train_loss,
                    'train_losses': train_losses, 
                    **train_metrics})
        
        # Eval (Metrics already computed)
        preds, true_preds, \
        valid_loss, eval_metrics = evaluate(
                                    model=model, 
                                    iterator=val_iterator,
                                    loss_fn=test_loss_fn,
                                    metrics_func=metrics_fn,
                                    task=task)
        # Update the main dict.
        all_eval_stats = urun.update_all_stats(
                   all_eval_stats , {'valid_loss': valid_loss, **eval_metrics})

        end_time = time.time()
        
        
        if ((task=='classification') and (valid_loss > best_valid_loss)) or (
            (task=='regression') and (valid_loss < best_valid_loss)): #Best metric achieved
            print('New Best model found: Epoch {} - Loss {}'.format(epoch, valid_loss))
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
        
        #Logging
        urun.logging_message(epoch, start_time, end_time, train_loss, valid_loss, **eval_metrics)

        # WRITE ON BOARD WHAT WE WANT
        if writer is not None:
            urun.update_writer(
                    writer=writer, 
                    epoch=epoch, 
                    metrics_dict=eval_metrics,
                    train_losses=train_losses,
                    train_loss=train_loss, 
                    valid_loss=valid_loss,
                    labels=true_preds, preds=preds,
                    N_train=len(train_iterator),
                    N_eval=len(val_iterator),
                    mode=mode)          
        
        loop.set_description('Epoch {} | Loss {}'.format(epoch,
                                                         valid_loss))

    #=====================================
    #EVAL
    test_preds, test_labels, \
    test_loss, test_metrics = evaluate(
                model=best_model, 
                iterator=test_iterator,
                loss_fn=test_loss_fn,
                task=task,
                metrics_func=metrics_fn)
    print(
        f'\t Final test ACC {test_loss:.3f}')

    if save:
        torch.save(best_model.state_dict(), output_dir +
                '/model-best.pt')

    #Add Final metrics to Board --> Hparams
    if writer is not None:
        urun.update_post_train_writer(
            writer, args, test_metrics=test_metrics, 
            hparam_metric_dict=test_metrics, hparam_dict=None)
        
    training_stats = {
        'train_metrics': all_train_stats, 
        'eval_metrics': all_eval_stats, 
        'test_metrics': test_metrics,
        'test_preds': test_preds,
        'test_labels': test_labels
        }
    
    return best_model, optimizer, training_stats

   


    


