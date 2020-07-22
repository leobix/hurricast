import numpy as np
import pandas as pd
import tensorly as tl #tensorly package
from tensorly.decomposition import tucker #tucker decomp package

#function to calculate dimensional norms for 4d tensor
def get_norms_4d(tensor):

    tensor_shape = tensor.shape
    dim = len(tensor.shape)

    core, _ = tucker(tensor, ranks=tensor_shape)

    norm0=[]
    for i in range(core.shape[0]):
        norm0.append(tl.norm(core[i,:,:,:]))

    norm1=[]
    for i in range(core.shape[1]):
        norm1.append(tl.norm(core[:,i,:,:]))

    norm2=[]
    for i in range(core.shape[2]):
        norm2.append(tl.norm(core[:,:,i,:]))

    norm3 = []
    for i in range(core.shape[2]):
        norm3.append(tl.norm(core[:,:,:,i]))

    norms = pd.concat([pd.Series(norm0),pd.Series(norm1),pd.Series(norm2),pd.Series(norm3)], axis=1)
    norms_cumsum = norms.cumsum()/norms.sum()#'cumsum of norms'
    return norms_cumsum

#function reconstruct tensor given core and factors
def tucker_to_tensor(core, factors):
    tensor = core.copy()
    dim = len(core.shape)
    for i in range(dim):
        tensor = tl.tenalg.mode_dot(tensor, factors[i], mode=i)
    return tensor


#vision data to array data using tensor decomp
def viz_to_arr(tensor, reduced_ranks):
    #low rank decomp
    core, factors = tucker(tensor, reduced_ranks)
    out_arr = core.flatten()

    #calculate approximation error
    approx = tucker_to_tensor(core, factors)
    approx_error = tl.norm(approx-tensor)/tl.norm(tensor)*100  #euclidean norm

    return out_arr,  approx_error

#multiple tensor data reduce dims
def viz_reduce(tensors, core_ranks): #input: tensors, sample_size = first dimension
    # tensors = tensors.numpy()
    #low rank decomp
    core_out = np.zeros((tensors.shape[0], core_ranks[0], core_ranks[1], core_ranks[2], core_ranks[3]))
    approx_error = 0
    for i in range(tensors.shape[0]):
        tensor = tensors[i]
        core, factors = tucker(tensor, core_ranks)
        core_out[i,:,:,:,:] = core
        #calculate approximation error
        approx = tucker_to_tensor(core, factors)
        approx_error +=  tl.norm(approx-tensor)/tl.norm(tensor)*100  #euclidean norm

    approx_error = approx_error/tensors.shape[0]
    # core_out = torch.tensor(core_out).float()
#     print('approximation error=', approx_error)
    return core_out,  approx_error


#function intaking x_stat, x_viz and concat them according to ranks
def concat_stat_viz(x_stat, x_viz_reduce): #third arg is vision tensor's dimensions
    #reshape tabular data
    x_stat = x_stat.reshape(x_stat.shape[0], -1)
    #reshape vision data
    viz = x_viz_reduce.reshape(x_viz_reduce.shape[0],-1)
    x_out = np.concatenate((x_stat, viz), axis=1)
    return x_out


def ranks_to_str(ranks):
    # Converting integer list to string list
    s = [str(i) for i in ranks]
    # Join list items using join()
    out = "".join(s)
    return out


#standardize x vision:
def standardize_x_viz(x_viz_train, x_viz_test):
    means = x_viz_train.mean(axis=(0, 1, 3, 4))
    stds = x_viz_train.std(axis=(0, 1, 3, 4))

    for i in range(len(means)):
        x_viz_train[:, :, i] = (x_viz_train[:, :, i] - means[i]) / stds[i]
        x_viz_test[:, :, i] = (x_viz_test[:, :, i] - means[i]) / stds[i]

    return x_viz_train, x_viz_test

#standardize x stat:
def standardize_x_stat(x_stat_train, x_stat_test):
    means_stat = x_stat_train.mean(axis=(0, 1))
    stds_stat = x_stat_train.std(axis=(0, 1))

    for i in range(len(means_stat)):
        x_stat_train[:, :, i] = (x_stat_train[:, :, i] - means_stat[i]) / stds_stat[i]
        x_stat_test[:, :, i] = (x_stat_test[:, :, i] - means_stat[i]) / stds_stat[i]

    return x_stat_train, x_stat_test

#standardize y:
def standardize_y(y_train, y_test):
    y_train = y_train
    y_test = y_test
    mean = y_train.mean()
    std = y_train.std()
    y_train = (y_train-mean)/std
    y_test = (y_test-mean)/std
    return y_train, y_test, mean , std
