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

class Prepro:
    """
    Tensor Dataset to leverage torch utils.
    IN:
        param: vision_data
        param: y
        predict_at:
    Example:
    >>> train_data = run.HurricaneDS(vision_data, y, predict_at=8)
    >>> loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    >>> data_ = next(iter(loader)) --> Returns a dictionary of tensors

    """
    def __init__(self, 
                vision_data, 
                y, 
                train_split,
                predict_at, 
                window_size):
        self.vision_data = torch.tensor(vision_data)
        self.original_timestep = y.shape[1]
        self.y = y[:, :, 1:]  # Remove the index
        self.y = torch.tensor(self.y.astype(np.float32))
        self.split = train_split

        #self.create_targets(predict_at, window_size)

        #self.timestep = self.train_data['X_vision'].size(1)
    
    #TODO:Remove we don't need it actually ?
    def clean_timesteps(self, y, convert_type=np.float32):
        """
        Change the 'timestep' of the statistical 
        data in order to have something that is readable
        by torch.
        Convert the string type into a predefined type.
        """
        y[:, :, 0] = np.repeat(
            np.expand_dims(np.arange(self.original_timestep), 0),
            self.__len__(), axis=0)
        y = y.astype(convert_type)
        return y

    def create_targets(self, predict_at: int, window_size: int):
        """
        Reformat the series into sub-components. 
        Use class attributes ```vision_data``` and ```y```
        (tensors of images and tabular data).
        IN:
            param predict_at: timestep to predict at in the future
            param window_size: number of components of a sub time-series
        OUT:
            train data: dict
                target_displacement: (N, predict_at, 2): an intermediary version where 
                                    zeros have not yet been remove
                target_intensity: N, 
                X_vision
                X_stat:
        """
        #Unfold
        X_vision = self.vision_data[:, :-predict_at].unfold(1, window_size, 1)
        X_stat = self.y[:, :-predict_at].unfold(1, window_size, 1)
        #Targets

        target_displacement = self.y[:, window_size:].unfold(1, predict_at, 1)#.sum(axis=-1) 
        target_intensity = self.y[:, (predict_at + window_size)-1:]
        #here as a baseline, we take the last category to be the category in predict_at
        target_intensity_cat_baseline = self.y[:, (window_size-1):(-predict_at)]
        #Permute: Put the last dimension on axis=2: (..., ..., 8, ...,...
        X_vision = X_vision.permute(0, 1, 6, 2, 3, 4, 5) 
        X_stat = X_stat.permute(0, 1, 3, 2)  
        target_displacement = target_displacement.permute(0, 1, 3, 2)
        #N_ouragans, N_unrolled, ... --> N_ouragans x N_unrolled
        X_vision, X_stat, \
            target_displacement, target_intensity, target_intensity_cat_baseline = map(lambda x: x.flatten(
                end_dim=1), (X_vision, X_stat, target_displacement, target_intensity, target_intensity_cat_baseline))  # Flatten everything

        #Resize X_vision
        X_vision = X_vision.flatten(start_dim=2, end_dim=3)
    
        tgt_displacement = torch.index_select(target_displacement,
                                                 dim=-1,
                                                 index=torch.tensor([target_displacement.size(-1)-2,
                                                                     target_displacement.size(-1)-1]))
        tgt_intensity = torch.select(target_intensity,
                                       dim=-1,
                                       index=2)
        tgt_intensity_cat = torch.select(target_intensity,
                                     dim=-1,
                                     index=-3).type(torch.LongTensor)
        tgt_intensity_cat_baseline = torch.select(target_intensity_cat_baseline,
                                         dim=-1,
                                         index=-3).type(torch.LongTensor)
        
        train_data = dict(X_vision=X_vision.float(),
                               X_stat=X_stat.float(),
                               target_displacement=tgt_displacement,
                               target_intensity=tgt_intensity,
                               target_intensity_cat = tgt_intensity_cat,
                               target_intensity_cat_baseline = tgt_intensity_cat_baseline)
        print("New dataset and corresponding sizes (null elements included):")

        for k, v in train_data.items():
            print(k, v.size())
        return train_data

    def remove_zeros(self, x_viz, x_stat, tgt_displacement, tgt_velocity, tgt_intensity_cat, tgt_intensity_cat_baseline):
        """
        - Remove zeros based on the values on tgt_displacement.
        - Reshape tgt_displacement as a sum over the window.
        """
        #OLD: good_indices = torch.sum(tgt_displacement == 0, axis=1) != 2
        
        # Get the indices for which the displacement in both directions = 0
        good_indices = (tgt_displacement == 0).sum(axis=-1) == 2 #(N, predict_at)
        
        # Count the number of zeros in a sequence of predict_at (=8) elements
        good_indices = good_indices.sum(axis=-1)  #(N,)
        
        # Keep only the samples for which we have no identically nul 
        # tensors over the sequence.
        good_indices = (good_indices == 0) #tensor of bool (N,)
        
        print('Keeping {} samples out of the initial {}.'.format(
            torch.sum(good_indices).item(), len(good_indices)))

        x_viz = x_viz[good_indices]
        x_stat = x_stat[good_indices]
        tgt_displacement = tgt_displacement[good_indices]
        tgt_velocity = tgt_velocity[good_indices]
        tgt_intensity_cat = tgt_intensity_cat[good_indices]
        tgt_intensity_cat_baseline = tgt_intensity_cat_baseline[good_indices]
        print('Reshaping the displacement target...')
        tgt_displacement = tgt_displacement.sum(axis=1)  # (N', 2)

        return x_viz, x_stat, tgt_intensity_cat, tgt_intensity_cat_baseline, tgt_displacement, tgt_velocity

    def get_mean_std(self, X_vision):
        m = torch.mean(
            X_vision.flatten(end_dim=1), 
                        axis=(0,-2,-1)).view(1, 1, 9, 1, 1)
        s = torch.std(
            X_vision.flatten(end_dim=1), 
            axis=(0, -2, -1)).view(1, 1, 9, 1, 1)
        
        return m,s
                        

    def __len__(self):
        return self.vision_data.size(0)

    @staticmethod
    def process(*args, **kwargs):
        #TODO: Discuss whether we want to
        #  divide the set into train/test differently.
        obj = Prepro(*args, **kwargs)
        data = obj.create_targets(**kwargs)
        #Unfold the time series
        data_tensors = obj.remove_zeros(*data.values())
        #Split the tensors into train/test
        len_ = data_tensors[0].size(0)
        #Get randon test indices
        #test_idx = np.random.choice(range(len_), int((1 - obj.split) * len_), replace=False)
        train_idx = np.arange(int(len_*obj.split))
        #Corresponding train indices
        test_idx = np.delete(np.arange(len_), train_idx)
        #train_idx = np.delete(np.arange(len_), test_idx )
        #Select tensors
        train_tensors = list(map(
            lambda x: x.index_select(
                dim=0, index=torch.Tensor(train_idx).long()),
            data_tensors)
        )
        test_tensors = list(map(
            lambda x: x.index_select(
                dim=0, index=torch.Tensor(test_idx).long()),
            data_tensors)
        )
        #Compute mean/std on the training set 
        m, s = obj.get_mean_std(train_tensors[0]) #Only the X_vision for now.
        train_tensors[0] = (train_tensors[0] - m)/s
        test_tensors[0] = (test_tensors[0] - m)/s

        #TODO Normalize x_stat
        #Normalize velocity target
        m_velocity = train_tensors[-1].mean()
        s_velocity = train_tensors[-1].std()
        train_tensors[-1] = (train_tensors[-1] - m_velocity)/s_velocity
        test_tensors[-1] = (test_tensors[-1] - m_velocity)/s_velocity
        #Normalize displacement
        m_dis =  train_tensors[-2].mean(axis=0)
        s_dis = train_tensors[-2].std(axis=0)
        train_tensors[-2] = (train_tensors[-2] - m_dis)/s_dis
        test_tensors[-2] = (test_tensors[-2] - m_dis)/s_dis
        return train_tensors, test_tensors    

    
def create_loss_fn(mode='intensity'):
    """
    Wrappers that uses same signature for all loss_functions.
    Can easily add new losses function here.
    #TODO: Théo--> See if we can make an entropy based loss.
    
    """
    assert mode in ['displacement', 
                    'intensity', 'intensity_cat']#, 'sum']
    
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
                     intensity_cat = intensity_cat_loss)
    return losses_fn[mode]
    
def create_model():
    #TODO
    raise NotImplementedError

def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


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
    loop = tqdm.tqdm(test_loader, desc='Evaluation')
    tgts = {'d': [], 'i': [] } #Get a dict of lists for tensorboard
    preds = {'i': [] } if args.target_intensity else {'d': [] }
    for data_batch in loop:
        #Put data on GPU
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

            batch_loss = loss_fn(model_outputs, tgt_displacement, tgt_intensity, tgt_intensity_cat)
            assert_no_nan_no_inf(batch_loss)
            total_loss += batch_loss.item() #Do we divide by the size of the data
            total_n_eval += tgt_intensity.size(0)

            if target_intensity_cat:
                class_pred = torch.softmax(model_outputs, dim=1).argmax(dim=1)
                accuracy += accuracy_score(target, class_pred)
                f1_micro += f1_score(target, class_pred, average='micro')
                f1_macro += f1_score(target, class_pred, average='macro')
                accuracy_baseline += accuracy_score(target, tgt_intensity_cat_baseline)
                f1_micro_baseline += f1_score(target, tgt_intensity_cat_baseline, average='micro')
                f1_macro_baseline += f1_score(target, tgt_intensity_cat_baseline, average='macro')

            #Keep track of the predictions/targets
            tgts['d'].append(tgt_displacement)
            tgts['i'].append(tgt_intensity)
            preds.get(tuple(preds.keys())[0]).append(model_outputs)
    
    tgts = { k : torch.cat(v) for k, v in tgts.items() }
    preds = { k: torch.cat(v) for k, v in preds.items()} 
    #=====================================
    #Compute norms, duck type and add to board.
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
        scheduler=None,
        l2_reg=0., 
        device=torch.device('cpu'),
        target_intensity = False,
        target_intensity_cat = False):
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
            #Put data on GPU
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
                                target_intensity_cat = target_intensity_cat,
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last = True)
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
    #print(vars(args))
    setup.create_seeds()
    main(args)

    




    


