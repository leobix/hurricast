import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Slice Learning
def slice_learning(tensor, r, max_iteration=50, convergence=1e-5, plot_loss=False):
    m,n,t = tensor.shape
    old_tensor = np.array(tensor).copy()
    mae_all=[]

    mae = convergence + 1
    #Iterative slice learning
    for i in range(max_iteration):
        if mae > convergence:
            new_tensor = slice_iteration(old_tensor,r)
            mae = get_mae(new_tensor,old_tensor)
            mae_all.append(mae)
            old_tensor = new_tensor

    #plot MAE difference between iterations
    max_i = len(mae_all)

    print('Code stopped at iteration = %s, and mae difference = %s'%(max_i,mae_all[-1]))
    if plot_loss:
        plt.plot(np.linspace(1,max_i,max_i),mae_all)
        plt.yscale('log')
        plt.show()
    return new_tensor

#2 dimensional SVD
def mat_svd(tensor,r):
    m,n,t= tensor.shape
    #Mode-1 folding
    mode1= tensor.reshape(m,n*t)
    U,S,V= np.linalg.svd(mode1, compute_uv=True)
    U1=U[:,:r]
    #Mode-2 folding
    mode2= tensor.reshape(m*t,n)
    U,S,V= np.linalg.svd(mode2,compute_uv=True)
    V1=V[:,:r]
    return U1,V1

#calculate MAE:
def get_mae(tensor1,tensor2):
    error = np.abs(tensor1-tensor2).sum()
    return error


#Slice Iteration: for one iteration
def slice_iteration(tensor,r):
    m,n,t = tensor.shape
    new_tensor =np.zeros((m,n,t))

    #get U,V
    U,V = mat_svd(tensor, r)

    #update tensor
    UU = U @ U.T
    VV = V @ V.T
    for i in range(t):
        new_tensor[:,:,i] = UU @ tensor[:,:,i] @ VV
    return new_tensor
