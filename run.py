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
        self.y = torch.tensor(self.clean_timesteps(y))
        self.y = self.y[:,:,1:] #Remove the index
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

    def create_targets(self, predict_at, window_size):
        #Unfold
        X_vision = self.vision_data[:, :-predict_at].unfold(1, window_size, 1)
        X_stat = self.y[:, :-predict_at].unfold(1, window_size, 1)
        targets = self.y[:, (predict_at + window_size)-1:]
        #Permute
        X_vision = X_vision.permute(0, 1, 6, 2, 3, 4, 5) #TODO:Make more automatic
        X_stat = X_stat.permute(0, 1, 3, 2)  # TODO:Make more automatic

        #print('vis', X_vision.size())
        X_vision, X_stat, targets = map(lambda x: x.flatten(
            end_dim=1), (X_vision, X_stat, targets)) #Flatten everything

        #REsize X_vision
        X_vision = X_vision.flatten(start_dim=2, end_dim=3)

        print("New dataset and corresponding sizes: ", X_vision.size(), X_stat.size(), targets.size())
        target_displacement = torch.index_select(targets,
                                                 dim=-1,
                                                 index=torch.tensor([targets.size(-1)-2,
                                                                     targets.size(-1)-1]))
        target_velocity = torch.select(targets,
                                       dim=-1,
                                       index=4)
        
        train_data = dict(X_vision=X_vision.float(),
                               X_stat=X_stat.float(),
                               target_displacement=target_displacement,
                               target_velocity=target_velocity)
        return train_data

    def remove_zeros(self, x_viz, x_stat, tgt_displacement, tgt_velocity):
        good_indices = torch.sum(tgt_displacement == 0, axis=1) != 2
        x_viz = x_viz[good_indices]
        x_stat = x_stat[good_indices]
        tgt_displacement = tgt_displacement[good_indices]
        tgt_velocity = tgt_velocity[good_indices]
        return x_viz, x_stat, tgt_displacement, tgt_velocity
                

    def get_mean_std(self, X_vision):
        m = torch.mean(
            X_vision.flatten(end_dim=1), 
                        axis=(0,-2,-1)).view(1, 1, 9, 1, 1)
        s = torch.mean(
            X_vision.flatten(end_dim=1), 
            axis=(0, -2, -1)).view(1, 1, 9, 1, 1)
        return m,s
                        

    def __len__(self):
        return self.vision_data.size(0)

    @staticmethod
    def process(*args, **kwargs):
        obj = Prepro(*args, **kwargs)
        data = obj.create_targets(**kwargs)
        #Unfold the time series
        data_tensors = obj.remove_zeros(*data.values())
        #Split the tensors into train/test
        len_ = data_tensors[0].size(0)
        #Get randon test indices
        test_idx = np.random.choice(range(len_), int((1 - obj.split) * len_), replace=False) 
        #Corresponding train indices
        train_idx = np.delete(np.arange(len_), test_idx )
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
        return train_tensors, test_tensors
        
        

    #@staticmethod
    #def process(*args, **kwargs):       
    #    obj = Prepro(*args, **kwargs) 
    #    return obj.create_targets(**kwargs)
    

    '''
    #def create_targets(self, predict_at):
        """
        Create samples/targets. Will reshape the
        """
        samples_idx = torch.arange(self.timestep-predict_at)
        targets_idx = samples_idx + predict_at

        X_vision = torch.index_select(self.vision_data,
                                      index=samples_idx, dim=1)
        X_stat = torch.index_select(self.y,
                                    index=samples_idx, dim=1)

        target_vision = torch.index_select(self.vision_data,
                                           index=targets_idx, dim=1)
        target_stat = torch.index_select(self.y,
                                         index=targets_idx, dim=1)

        #Get the two last elements
        target_displacement = torch.index_select(target_stat,
                                                 dim=-1,
                                                 index=torch.tensor([target_stat.size(-1)-2,
                                                                     target_stat.size(-1)-1]))
        target_velocity = torch.select(target_stat,
                                       dim=-1,
                                       index=4)

        self.train_data = dict(X_vision=X_vision,
                               X_stat=X_stat,
                               target_displacement=target_displacement,
                               target_velocity=target_velocity)
    


    def __str__(self):
        str_ = f"Number of elements: {self.__len__()}. Timesteps: {self.timestep}.\
        Respective shapes: {self.y.shape} and {self.vision_data.shape}"
        return str_

    def __getitem__(self, i):
        out = {k: v[i] for k, v in self.train_data.items()}
        return out
    '''


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()

def eval(model,
        loss_fn, 
        test_loader,
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
    loop = tqdm.tqdm(test_loader, desc='Evaluation')
    for data_batch in loop:
        #Put data on GPU
        data_batch = tuple(map(lambda x: x.to(device), 
                                    data_batch))
        x_viz, x_stat, target, _ = data_batch
        with torch.no_grad():
           
            model_outputs = model(x_viz, x_stat)
            batch_loss = loss_fn(model_outputs, target)
            assert_no_nan_no_inf(batch_loss)
            total_loss += batch_loss.item() #Do we divide by the size of the data
            total_n_eval += target.size(0)
    print(total_loss/float(total_n_eval))
    return model, total_loss, total_n_eval

def train(model,
        optimizer,
        loss_fn, 
        n_epochs,
        train_loader,
        test_loader,
        args,
        scheduler=None,
        l2_reg=0., 
        device=torch.device('cpu')):
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
        for data_batch in inner_loop:
            #Put data on GPU
            data_batch = tuple(map(lambda x: x.to(device), 
                                    data_batch))
            x_viz, x_stat, target, _ = data_batch
            optimizer.zero_grad()

            model_outputs = model(x_viz, x_stat)
            batch_loss = loss_fn(model_outputs, target)
            assert_no_nan_no_inf(batch_loss)
            if l2_reg > 0:
                L2 = 0.
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        L2 += (p**2).sum()
                batch_loss += 2./x_viz.size(0) * l2_reg * L2
                assert_no_nan_no_inf(batch_loss)

            training_loss.append(batch_loss.item())


            batch_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            inner_loop.set_description('Epoch {} | Loss {}'.format(epoch,
                                                         batch_loss.item()))
        eval_loss_sample ,_ , _ = eval(model,
                                loss_fn=loss_fn,
                                test_loader=test_loader,
                                device=device)

        if eval_loss_sample < previous_best:
            previous_best = eval_loss_sample
            torch.save(model, osp.join(args.output_dir, 'best_model.pt'))
    
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
    vision_data = torch.load(osp.join(args.data_dir, 'vision_data.pt')) #NUMPY ARRAY
    y = torch.load(osp.join(args.data_dir, 'y.pt'))
    
    #y, _ = data_processing.prepare_tabular_data_vision(
    #        path=osp.join(args.data_dir, "ibtracs.last3years.list.v04r00.csv"), 
    #        min_wind=args.min_wind, 
    #        min_steps=args.min_steps,
    #        max_steps=args.max_steps, 
    #        get_displacement=True)
   
    train_tensors, test_tensors = Prepro.process(vision_data, 
                                y, 
                                args.train_test_split,
                                predict_at=args.predict_at, 
                                window_size=args.window_size)
    train_ds = TensorDataset(*train_tensors)
    test_ds = TensorDataset(*test_tensors)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Create model
    encoder_config = setup.encoder_config #Load pre-defined config
    encoder = models.CNNEncoder(n_in=3*3,
                                n_out=128,
                                hidden_configuration=encoder_config)
    #n_in decoder must be out encoder + 7!
    #decoder_config = setup.decoder_config
    #model = models.ENCDEC(n_in_decoder=128+7, 
    #                           n_out_decoder=2, 
    #                           encoder=encoder, 
    #                           hidden_configuration_decoder=decoder_config, 
    #                            window_size=args.window_size)
    model = models.LINEARTransform(encoder)
    model = model.to(device)
    
    print("Using model", model)
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr)
    
    model, optimizer, loss = train(model,
                                optimizer=optimizer,
                                loss_fn=nn.MSELoss(),
                                n_epochs=args.n_epochs,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                args={},
                                scheduler=None,
                                l2_reg=args.l2_reg)
    plt.plot(loss)
    plt.title('Training loss')
    plt.show()
    #Sve results
    #with open(path_to_results, 'w') as writer:
    torch.save(model, osp.join(args.output_dir, 'final_model.pt'))

if __name__ == "__main__":
    import setup
    args = setup.create_setup()
    #main(args)

    




    


