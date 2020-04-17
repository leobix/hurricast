import sys 
sys.path.append('../')
import torch 
from torch.utils import data
import numpy as np
import argparse
from utils import utils_vision_data, data_processing, plot
import matplotlib.pyplot as plt
import tqdm

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
                predict_at=8, 
                window_size=8):
        self.vision_data = torch.tensor(vision_data)
        self.original_timestep = y.shape[1]
        self.y = torch.tensor(self.clean_timesteps(y))

        #self.create_targets(predict_at, window_size)

        #self.timestep = self.train_data['X_vision'].size(1)

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

    def __len__(self):
        return self.vision_data.size(0)

    @staticmethod
    def process(*args, **kwargs):       
        obj = Prepro(*args, **kwargs) 
        return obj.create_targets(**kwargs)
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


def train(model,
        optimizer,
        loss_fn, 
        n_epochs,
        train_loader,
        args,
        scheduler=None,
        l2_reg=0.):
    """
    #TODO: Comment a bit    
    """
    # set model in training mode
    model.train()

    torch.manual_seed(0)

    # train model
    training_loss = []
    loop = tqdm.trange(n_epochs, desc='Epochs')
    for epoch in loop:
        for data_batch in tqdm.tqdm(train_loader):
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
        loop.set_description('Epoch {} | Loss {}'.format(epoch,
                                                         batch_loss.item())
                             )
    return model, optimizer, training_loss


def run_pipeline():
    min_wind = 50
    min_steps = 20
    max_steps = 60
    #Load data
    data = utils_vision_data.get_storms(min_wind=min_wind, min_steps=min_steps, max_steps=max_steps,
                                        extraction=True, path='./data/ibtracs.last3years.list.v04r00.csv')

    #y represents the actual target data we are going to use
    y, _ = data_processing.prepare_tabular_data_vision(
        min_wind=min_wind, min_steps=min_steps, max_steps=max_steps)
    #Then we get their corresponding vision maps
    vision_data = utils_vision_data.extract_vision(data, epsilon=0.05)
    animation = plot.animate_ouragan(vision_data, n=0)
    plt.show()
    
    train_data = HurricaneDS(vision_data, y)
    loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    data_ = next(iter(loader))
    
    for key, value in data_.items():
        print(key, value.size())
    

if __name__ == "__main__":
    run_pipeline()




    


