import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl #tensorly package
from tensorly.decomposition import tucker #tucker decomp package



class tucker_forward(torch.nn.Module):
    # m,n,t is the size of input x, k is the low rank
    def __init__(self, x, ranks, I, hosvd_init=True):

        super(tucker_forward, self).__init__()

        if hosvd_init==True:
            #do hosvd
            s, factors = tucker(x.numpy(), ranks)

            # register parameters
            self.S = torch.nn.Parameter(torch.tensor(np.array(s), requires_grad=True))
            self.U0= torch.nn.Parameter(torch.tensor(np.array(factors[0]), requires_grad=True))
            self.U1= torch.nn.Parameter(torch.tensor(np.array(factors[1]), requires_grad=True))
            self.U2= torch.nn.Parameter(torch.tensor(np.array(factors[2]), requires_grad=True))

        else:
            self.S = torch.nn.Parameter(torch.rand(ranks))
            self.U0= torch.nn.Parameter(torch.randn(x.shape[0],ranks[0]))
            self.U1= torch.nn.Parameter(torch.randn(x.shape[1],ranks[1]))
            self.U2= torch.nn.Parameter(torch.randn(x.shape[2],ranks[2]))


    def forward(self, x):
        t1=torch.einsum('ijk,li->ljk', self.S, self.U0) #mode 1 product: i to l
        t2=torch.einsum('ijk,lj->ilk', t1, self.U1) #mode 2 product: j to l
        x_hat=torch.einsum('ijk,lk->ijl', t2, self.U2)#mode 3 product k to l
        return x_hat


def loss_fn(x_hat, x, I):
    loss =  torch.mul((x_hat - x),I).pow(2).sum()
    return loss


#define function
def run_tucker(x, ranks, I,max_itr=500, lr=1e-6, hosvd_init=True, plot_loss=False): #x: input torch.Tensor

    m,n,t = x.shape
    model = tucker_forward(x, ranks, I, hosvd_init)

    loss_arr =[] #record loss trajectory
    ugrad_arr=[]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for t in range(max_itr):
        optimizer.zero_grad()
        x_hat= model(x)
        loss = loss_fn(x_hat, x, I)
        loss.backward()
        optimizer.step()

        #record loss and gradients
        loss_arr.append(loss.detach().item())


    x_hat = model(x)
    final_itr = t

    if plot_loss:
        plt.plot(loss_arr)

    return loss_arr, x_hat


#function reconstruct tensor given core and factors
def tucker_to_tensor(core, factors):
    tensor = core.copy()
    dim = len(core.shape)
    for i in range(dim):
        tensor = tl.tenalg.mode_dot(tensor, factors[i], mode=i)
    return tensor


#function to calculate dimensional norms  
def get_norms(tensor):   
  
    tensor_shape = tensor.shape
    core, _ = tucker(tensor, ranks=tensor_shape)
    
    norm0=[]
    for i in range(core.shape[0]):
        norm0.append(np.linalg.norm(core[i,:,:], ord='fro'))  
    
    norm1=[]
    for i in range(core.shape[1]):
        norm1.append(np.linalg.norm(core[:,i,:], ord='fro'))  
    
    norm2=[]
    for i in range(core.shape[2]):
        norm2.append(np.linalg.norm(core[:,:,i], ord='fro'))  
    
    norms = pd.concat([pd.Series(norm0),pd.Series(norm1),pd.Series(norm2)], axis=1) 
    return norms 


def error(T_true, T_hat):
    mse = round((T_hat-T_true).pow(2).sum().item())
    pct = round(((T_hat-T_true).pow(2).detach().sum()/T_true.pow(2).sum()).item()*100, 2)
    print('mse, pct_mse =', mse, pct)
    return mse, pct