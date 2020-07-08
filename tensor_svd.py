from __future__ import print_function
import torch
import numpy as np

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create a randon tensor
# tensor size
m, n, t = 6, 6, 6;
# create sparse tensor X with ones at idx 
idx = torch.LongTensor([[0, 1, 1],
                        [2, 0, 2],
                        [1, 1, 2]]);  # idx for
value = torch.ones(3);
X = torch.sparse.FloatTensor(idx, value, torch.Size([m, n, t])).to_dense();


# SVD
def svd(X):
    r = 3
    # Mode-1 folding
    X1 = X.reshape(m, n * t)
    U, S, V = torch.svd(X1, some=True, compute_uv=True)
    U1 = U[:, :r]
    # Mode-2 folding
    X2 = X.reshape(m * t, n)
    U, S, V = torch.svd(X2, some=True, compute_uv=True)
    V1 = V[:, :r]
    return U1, V1


U1, V1 = svd(X)
print(U1, V1)
