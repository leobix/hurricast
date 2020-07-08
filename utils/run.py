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
import os 
import os.path as osp
from sklearn.metrics import f1_score, accuracy_score
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

#=============
#TODO: Put in metrics? 
def create_loss_fn(mode='intensity'):
    """
    Wrappers that uses same signature for all loss_functions.
    Can easily add new losses function here.
    #TODO: Théo--> See if we can make an entropy based loss.
    """
    assert mode in ['displacement', 
                    'intensity', 
                    'intensity_cat']#, 'sum']
    
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
        return  base_loss_fn(model_outputs, 
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

def collate_fn(batch):
    """
    Function to turn a batch of data into 
    a dict.
    #TODO: Cleaner way: Make a Batch class
    """

    

def create_model():
    #TODO
    raise NotImplementedError

#TODO: Create an eval metric function that computes all the F1 SCORE 
# and so on
def eval(model,
        loss_fn, 
        test_loader,
        writer, 
        epoch_number,
        target_intensity = False,
        target_intensity_cat = False,
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
    tgts = {'d': [], 'i': [], 'i_cat': [] } #Get a dict of lists for tensorboard
    #preds = {'i': [] } if target_intensity or target_intensity_cat else {'d': [] }
    if target_intensity:
        preds = {'i':[]}
    elif target_intensity_cat:
        preds = {'i_cat': []}
    else:
        preds = {'d':[]}
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

            batch_loss = loss_fn(model_outputs, tgt_displacement, tgt_intensity, tgt_intensity_cat)
            assert_no_nan_no_inf(batch_loss)
            total_loss += batch_loss.item() * tgt_intensity.size(0) #Do we divide by the size of the data
            total_n_eval += tgt_intensity.size(0)

            if target_intensity_cat:
                class_pred = torch.softmax(model_outputs, dim=1).argmax(dim=1) #Theo: Same as torch argmax directly.
                accuracy += accuracy_score(target, class_pred)
                f1_micro += f1_score(target, class_pred, average='micro')
                f1_macro += f1_score(target, class_pred, average='macro')
                accuracy_baseline += accuracy_score(target, tgt_intensity_cat_baseline)
                f1_micro_baseline += f1_score(target, tgt_intensity_cat_baseline, average='micro')
                f1_macro_baseline += f1_score(target, tgt_intensity_cat_baseline, average='macro')
            else:
                if not target_intensity:
                    baseline_loss_eval += ( tgt_displacement_baseline.size(0)*loss_fn(
                        tgt_displacement_baseline, tgt_displacement, tgt_intensity, tgt_intensity_cat)
                    )
                    
            #Keep track of the predictions/targets
            tgts['d'].append(tgt_displacement)
            tgts['i'].append(tgt_intensity)
            tgts['i_cat'].append(tgt_intensity_cat.float())
            if not target_intensity_cat:
                preds.get(tuple(preds.keys())[0]).append(model_outputs)
            else:
                print('cat',class_pred )
                preds.get(tuple(preds.keys())[0]).append(class_pred.float())
                
    

    #Re-defined the losses
    tgts = { k : torch.cat(v) for k, v in tgts.items() }
    preds = { k: torch.cat(v) for k, v in preds.items()} 
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


def train(model,
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


def main(args):
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



if __name__ == "__main__":
    import setup
    args = setup.create_setup()
    setup.create_seeds()
    main(args)

    




    


