import sys 
sys.path.append('../')
import torch 
from torch.utils import data
import numpy as np
import argparse
from utils import utils_vision_data, data_processing, plot
import matplotlib.pyplot as plt

class HurricaneDS(torch.utils.data.Dataset):
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
    def __init__(self, vision_data, y, predict_at=8):
        self.vision_data = torch.tensor(vision_data)
        self.timestep = y.shape[1]
        self.y = torch.tensor(self.clean_timesteps(y))
        print(self)
        self.create_targets(predict_at)

    def clean_timesteps(self, y, convert_type=np.float32):
        """
        Change the 'timestep' of the statistical 
        data in order to have something that is readable
        by torch.
        Convert the string type into a predefined type.
        """
        y[:, :, 0] = np.repeat(
            np.expand_dims(np.arange(self.timestep), 0),
            self.__len__(), axis=0)
        y = y.astype(convert_type)
        return y

    def create_targets(self, predict_at):
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

    def __len__(self):
        return self.vision_data.size(0)

    def __str__(self):
        str_ = f"Number of elements: {self.__len__()}. Timesteps: {self.timestep}.\
        Respective shapes: {self.y.shape} and {self.vision_data.shape}"
        return str_

    def __getitem__(self, i):
        out = {k: v[i] for k, v in self.train_data.items()}
        return out


def train(model, 
        optimizer,
        loader, 
        args):
    #TODO: Implement it once we've decided the
    #architectures we want. 
    return None


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




    


