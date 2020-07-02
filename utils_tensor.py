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
