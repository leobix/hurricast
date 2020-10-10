import torch
import numpy as np
import os
import os.path as osp
import copy
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List

"""
#IMPORTANT TODO.
1.. Add WeightedSampler
2. Remove the baseline from the named tensors and add a special option.
3. Write documentation !
4. Add argument to save the processed tensors beforehand.


#Secondary TODO:
1. Add typing (safer).
2. Add memory buffer + pinning for the loaders (faster computation)
"""
#TODO: Add mixture intensity + displacement
accepted_modes = {  # Modes and associated target vars.
    'intensity': 'tgt_intensity',
    'displacement': 'tgt_displacement',
    'intensity_cat': 'tgt_intensity_cat',
    'baseline_intensity_cat': 'tgt_intensity_cat_baseline',
    'baseline_displacement': 'tgt_displacement_baseline'
}

accepted_modes = {k: (v, 'x_viz', 'x_stat'
                      ) for k, v in accepted_modes.items()}


def CheckMode(func):
    def decorator(*fargs, **fkwargs):
        if 'mode' not in fkwargs:
            raise KeyError("Please provide the mode as a named argument.")

        mode = fkwargs.get('mode')
        assert mode in accepted_modes.keys(), "\
        Try to use the wrong mode argument.\
        {} is not supported.\
        Please choose among {}".format(mode,
                                       list(accepted_modes.keys()))
        out = func(*fargs, **fkwargs)
        return out
    return decorator


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
                y,
                vision_data,
                train_split,
                predict_at,
                window_size):
        self.vision_data = torch.tensor(vision_data)
        self.original_timestep = y.shape[1]
        self.y = y[:, :, 1:]  # Remove the index
        self.y = torch.tensor(self.y.astype(np.float32))
        self.split = train_split
        self.predict_at = predict_at


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

    def create_targets(self, 
                        predict_at: int, 
                        window_size: int):
        """
        Reformat the series into sub-components. 
        Use class attributes ```vision_data``` and ```y```
        (tensors of images and tabular data).
        IN:
            param predict_at: timestep to predict at in the future
            param window_size: number of components of a sub time-serie
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

        target_displacement = self.y[:, window_size:].unfold(
            1, predict_at, 1)  # .sum(axis=-1)
        target_intensity = self.y[:, (predict_at + window_size)-1:]
        #here as a baseline, we take the last category to be the category in predict_at
        target_intensity_cat_baseline = self.y[:,
                                               (window_size-1):(-predict_at)]
        #Permute: Put the last dimension on axis=2: (..., ..., 8, ...,...
        X_vision = X_vision.permute(0, 1, 6, 2, 3, 4, 5)
        X_stat = X_stat.permute(0, 1, 3, 2)
        target_displacement = target_displacement.permute(0, 1, 3, 2)
        #N_ouragans, N_unrolled, ... --> N_ouragans x N_unrolled
        X_vision, X_stat, \
            target_displacement, \
            target_intensity, \
            target_intensity_cat_baseline = map(lambda x: x.flatten(end_dim=1),
                                                (X_vision, X_stat,
                                                 target_displacement, target_intensity,
                                                 target_intensity_cat_baseline))  # Flatten everything

        #Resize X_vision
        #TODO Be careful, everytime the format of y changes, change index
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
                                         index=14).type(torch.LongTensor)
        tgt_intensity_cat_baseline = torch.select(target_intensity_cat_baseline,
                                                  dim=-1,
                                                  index=14).type(torch.LongTensor)

        train_data = dict(X_vision=X_vision.float(),
                          X_stat=X_stat.float(),
                          target_displacement=tgt_displacement,
                          target_intensity=tgt_intensity,
                          target_intensity_cat=tgt_intensity_cat,
                          target_intensity_cat_baseline=tgt_intensity_cat_baseline)
        print("New dataset and corresponding sizes (null elements included):")

        for k, v in train_data.items():
            print(k, v.size())
        return train_data

    def remove_zeros(self, 
                     x_viz: torch.Tensor,
                     x_stat: torch.Tensor,
                     tgt_displacement: torch.Tensor,
                     tgt_velocity: torch.Tensor,
                     tgt_intensity_cat: torch.Tensor,
                     tgt_intensity_cat_baseline: torch.Tensor):
        """
        - Remove zeros based on the values on tgt_displacement.
        - Reshape tgt_displacement as a sum over the window.
        """
        #OLD: good_indices = torch.sum(tgt_displacement == 0, axis=1) != 2

        # Get the indices for which the displacement in both directions = 0
        good_indices = (tgt_displacement == 0).sum(
            axis=-1) == 2  # (N, predict_at)

        # Count the number of zeros in a sequence of predict_at (=8) elements
        good_indices = good_indices.sum(axis=-1)  # (N,)

        # Keep only the samples for which we have no identically nul
        # tensors over the sequence.
        good_indices = (good_indices == 0)  # tensor of bool (N,)

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

    def get_mean_std(self, 
                    X_vision: torch.Tensor):
        """
        Compute the mean and std of the vision data. 
        """
        m = torch.mean(
            X_vision.flatten(end_dim=1),
            axis=(0, -2, -1)).view(1, 1, 9, 1, 1)
        s = torch.std(
            X_vision.flatten(end_dim=1),
            axis=(0, -2, -1)).view(1, 1, 9, 1, 1)

        return m, s

    def __len__(self):
        return self.vision_data.size(0)

    @staticmethod
    def process(y,
                vision_data,
                train_split,
                predict_at,
                window_size) -> (dict, dict):
        """
        Parameters:
        ------------
        args, kwargs: Prepro's init args

        Out:
        ------------
        named_train_tensors: Dict[torch.Tensor]                             
        named_test_tensors: Dict[torch.Tensor]

        Note:
        ----------
        Keys to be found in the output dictionary:
        'x_viz', 'x_stat',
        'tgt_intensity_cat',
        'tgt_intensity_cat_baseline',
        'tgt_displacement_baseline',
        'tgt_displacement', 
        'tgt_intensity'
        """
        #TODO: Discuss whether we want to
        #  divide the set into train/test differently.
        obj = Prepro(y, vision_data,
                     train_split,
                     predict_at,
                     window_size)
        data = obj.create_targets(predict_at,
                                  window_size)
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

        #THEO:
        # sum over the timesteps
        tgt_displacement_baseline = train_tensors[1][:,
                                                     :-(obj.predict_at-1),
                                                     torch.tensor([4, 5])].sum(axis=1)
        m_tgb = torch.mean(tgt_displacement_baseline, axis=0)
        s_tgb = torch.std(tgt_displacement_baseline, axis=0)
        tgt_displacement_baseline = (tgt_displacement_baseline - m_tgb) / s_tgb

        tgt_test_displacement_baseline = test_tensors[1][:,
                                                         :-(obj.predict_at-1),
                                                         torch.tensor([4, 5])].sum(axis=1)
        tgt_test_displacement_baseline = (
            tgt_test_displacement_baseline - m_tgb) / s_tgb

        #Compute mean/std on the training set
        m, s = obj.get_mean_std(train_tensors[0])  # Only the X_vision for now.
        train_tensors[0] = (train_tensors[0] - m)/s
        test_tensors[0] = (test_tensors[0] - m)/s

        #Standardize x_stat
        train_tensors[1] = train_tensors[1][:,:,:14]
        test_tensors[1] = test_tensors[1][:,:,:14]
        m_xstat = train_tensors[1].mean(axis=(0,1))
        #m_xstat[14] = 0 #Dont normalize the cat
        s_xstat = train_tensors[1].std(axis=(0, 1))
        #s_xstat[14] = 1 #Dont standardize the cat
        train_tensors[1] = (train_tensors[1] - m_xstat)/s_xstat
        test_tensors[1] = (test_tensors[1] - m_xstat)/s_xstat

        #Standardize velocity target
        m_velocity = train_tensors[-1].mean()
        s_velocity = train_tensors[-1].std()
        train_tensors[-1] = (train_tensors[-1] - m_velocity)/s_velocity
        test_tensors[-1] = (test_tensors[-1] - m_velocity)/s_velocity
        #Standardize displacement
        m_dis = train_tensors[-2].mean(axis=0)
        s_dis = train_tensors[-2].std(axis=0)
        train_tensors[-2] = (train_tensors[-2] - m_dis)/s_dis
        test_tensors[-2] = (test_tensors[-2] - m_dis)/s_dis
        #THEO: Add the displacement baseline
        train_tensors.insert(4, tgt_displacement_baseline)
        test_tensors.insert(4, tgt_test_displacement_baseline)
        #Théo: Add named args
        names = (
            'x_viz', 'x_stat',
            'tgt_intensity_cat',
            'tgt_intensity_cat_baseline',
            'tgt_displacement_baseline',
            'tgt_displacement', 'tgt_intensity')
        
        named_train_tensors = {
            name: tensor for name, tensor in zip(
                names, train_tensors)}
        
        named_test_tensors = {
            name: tensor for name, tensor in zip(
                names, test_tensors)}
                
        
        return named_train_tensors, named_test_tensors


@CheckMode
def filter_keys(train_tensors: Dict[str, torch.Tensor], 
                test_tensors: Dict[str, torch.Tensor], 
                mode: str) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
    """
    Given a "data mode", will filter the training tensors to keep only the 
    relevant elements.

    Parameters:
    -----------
    train_tensors: Dict[torch.Tensor] - named trained tensors 
    test_tensors: Dict[torch.Tensor] - named test tensors 
    mode: str   - One of the accepted modes to create the data

    Out:
    ------------
    train_tensors: Dict[torch.Tensor] - filtered dict with trained tensors 
    test_tensors: Dict[torch.Tensor] - filtered dict with test tensors 
    """
    def _filter_keys(input_dict, filtered_keys):
        return { k: v for k, v in input_dict.items()
                if k in filtered_keys}
    
    assert mode in accepted_modes, "Wrong mode chosen.\
    {} not in {}".format(mode, accepted_modes.keys())

    
    filtered_keys = accepted_modes[mode]
    train_tensors = _filter_keys(train_tensors, filtered_keys)
    test_tensors = _filter_keys(test_tensors, filtered_keys)
    
    return train_tensors, test_tensors

def create_collate_fn(dataset_keys: List[str],
                    keys_model: list=['x_viz', 'x_stat'], 
                    keys_loss: list=['trg_y']):
    """
    Create a collate fn to feed dict of tensors to the models.
    
    Given a batch (list of tensors), create two dictionaries:
        1: Dict with keys=keys_models, namely x_viz, x_stat (args to model.fwd)
        2: Dict with keys=keys_loss, namely x_viz, x_stat (args to loss func)
    """
    indices_mod = {i: k for i, k in enumerate(dataset_keys) if k in keys_model}
    #idx = {i: k for i, k in enumerate(dataset_keys) if k not in keys_model}
    idx_loss = (set(range(3)) - set(indices_mod.keys())).pop()
    indices_loss = {idx_loss: keys_loss[0]}

    #0: x_viz, 1: 'tgt_intensity_cat', ...
    #idx_model = {}
    #idx_loss = {
    #    i: k for i, k in idx_map.items() if k not keys_model}

    def _collate_fn(batch, indices_model, indices_loss):
        tupled_batch = list(zip(*batch))
        in_model = {k: torch.stack(tupled_batch[i]) for i,k in indices_model.items()}
        in_loss = {k: torch.stack(tupled_batch[i]) for i,k in indices_loss.items()}
        #in_loss = {keys_loss[0]: torch.stack(tupled_batch[idx_loss[0]])}
        #print(in_model, in_loss)
        return in_model, in_loss
    
    return lambda batch:  _collate_fn(batch, indices_mod, indices_loss)


def save_tensors(tensors: dict, 
                data_dir: str, 
                prefix: str='train'):
    """
    Save the different tensors in tensors in the data_dir. 
    The prefix will be prepended to the tensor name.
    Save under numpy format.

    tensors: Dict[str, torch.Tensor] - Tensors to be saved
    data_dir: str - Path of the data
    prefix: str - The tensors will be saved under the format 
        data_dir/{prefix}_{dict_key}.npy
    """
    for k, v in tensors.items():
        print('Saving - ', k)
        file = osp.join(data_dir, prefix + '_' + k +'.npy')
        assert isinstance(v, torch.Tensor)
        arr = v.numpy()
        with open(file, 'wb') as f:
            np.save(f, arr)
        

def load_tensors(data_dir: str, 
                prefix: str='train'):
    """
    Load the different tensors in the data_dir (whose names start with prefix)
    The prefix will be prepended to the tensor name.
    Save under numpy format.

    tensors: Dict[str, torch.Tensor] - Tensors to be saved
    data_dir: str - Path of the data
    prefix: str - The tensors will be saved under the format 
        data_dir/{prefix}_{dict_key}.npy
    """
    def clean_up_name(name:str, prefix:str, suffix:str):
        return name.split(prefix+"_")[-1].split(suffix)[0]
    
    tensors = {}
    
    files = tuple(
        filter(lambda x : x.startswith(prefix), os.listdir(data_dir)))
    
    for file in files:
        arr = np.load(
            osp.join(data_dir, file), allow_pickle=True)
        key = clean_up_name(file, prefix=prefix, suffix=".npy")
        tensors[key] = torch.from_numpy(arr)
    return tensors
    

def get_baseline(eval_tensors, mode):
    MAP = {
    'intensity': None, 
    'intensity_cat': 'tgt_intensity_cat',
    'displacement':'tgt_displacement'}
    modes = {#Modes and associated tasks
    'intensity': 'regression',
    'displacement': 'regression',
    'intensity_cat': 'classification'}
    
    task = modes.get(mode)
    MODE = MAP.get(mode) 
        
    tgt = copy.deepcopy(eval_tensors.get(MODE)) if MODE is not None else None
    baseline = eval_tensors.get(MODE + "_baseline") if MODE is not None else None
        
    return {"model_outputs": baseline, "target": tgt, "task": task}
    

@CheckMode        
def create_loaders(mode: str,
                    data_dir: str, 
                    vision_name: str, 
                    y_name: str, 
                    batch_size: int,
                    train_test_split: float, 
                    predict_at: int, 
                    window_size: int, 
                    debug:bool=False, 
                    do_save_tensors: bool=False, 
                    do_load_tensors: bool=True,
                    weights=[]):
    """
    #TODO: Write small doc
    """
    if not do_load_tensors:
    #Load numpyy arrays form disk
        vision_data = np.load(osp.join(data_dir, vision_name),
                              allow_pickle=True)
        y = np.load(osp.join(data_dir, y_name),
                    allow_pickle=True)
    
        #Create named tensors
        train_tensors, test_tensors = Prepro.process(
            y=y, 
            vision_data=vision_data,
            train_split=train_test_split,
            predict_at=predict_at,
            window_size=window_size)
        print('Done Preprocessing - preparing to save the tensors')
        if do_save_tensors:
            save_tensors(train_tensors,
                         data_dir,
                         prefix='tens_train')
            save_tensors(test_tensors,
                         data_dir,
                         prefix='tens_test')

    else: #Load train_tensors
        train_tensors = load_tensors(data_dir, prefix='tens_train')
        test_tensors = load_tensors(data_dir, prefix='tens_test')

    test_baseline = get_baseline(test_tensors, mode)
    #Filter the relevant keys
    train_tensors, test_tensors = filter_keys(
        train_tensors, test_tensors, mode=mode)

    print(train_tensors.keys())
    #Unroll in tensordataset
    train_ds = TensorDataset(*train_tensors.values())
    test_ds = TensorDataset(*test_tensors.values())
    if debug:
        N_DEBUG = 200
        train_ds = train_ds[:N_DEBUG]
        test_ds = test_ds[:N_DEBUG]
    #Create collate_fn 
    train_collate_fn = create_collate_fn(train_tensors.keys())
    test_collate_fn = create_collate_fn(test_tensors.keys())
        
    #if len(weights)==0:

    train_loader = DataLoader(train_ds, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            drop_last=True,
                            collate_fn=train_collate_fn)

    test_loader = DataLoader(test_ds, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=test_collate_fn)
    #TODO: Add Weighted Sampler
    if len(weights) > 0: 
        pass
    return train_loader, test_loader, test_baseline




if __name__ == "__main__":
    def test():
        from dataclasses import dataclass
        @dataclass
        class Args:
            data_dir: str
            y_name: str
            vision_name: str
            predict_at: int
            window_size: int
            train_test_split: float
            mode: str

        args = Args(data_dir="data/",
                    y_name="y.npy",
                    vision_name="vision_data.npy",
                    predict_at=8,
                    window_size=8,
                    train_test_split=0.8,
                    batch_size=10,
                    mode='intensity')

        create_loaders(**vars(args))



