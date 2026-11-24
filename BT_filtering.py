import numpy as np
import torch

def filter(A, X, Xv,device):

    if type(A) is np.ndarray:
        A = torch.from_numpy(A).to(device)
    if type(X) is np.ndarray:
        X = torch.from_numpy(X).to(device)
    X = X.float().to(device)
    A = A.float().to(device)
    I_n = torch.eye(A.shape[0]).to(device)
    I_d = torch.eye(X.shape[1]).to(device)
    if A.layout == torch.strided:
        A = A.to_sparse()
    # A @ ones repeats each row sum across its row.
    A = I_n + torch.sparse.mm(A, A.new_ones((A.shape[1], 1), layout=torch.strided))
    D = torch.sum(A, 1)
    D = torch.pow(D, -0.5)
    D[torch.isinf(D)] = 0
    A = D[:, None] * A * D[None, :]
    Ls = I_n - A
    G = I_n - 0.5 * Ls
    f = 2
    k = 5 # [1,2,3,4,5,7,10]
    a = 50 # [0.1,1,10,100,1000]

    # Apply the prior G^f to Xv and propagate the consensus by G^k.
    A_Xv = Xv
    for _ in range(f):
        A_Xv = G.matmul(A_Xv)
    X_bar = X
    for _ in range(k):
        X_bar = G.matmul(X_bar)

    # Woodbury applied to Xv: only the feature-dimensional system is solved.
    XXt_bar = X_bar.t().matmul(X_bar)
    tmp = torch.linalg.solve(a * I_d + XXt_bar, X_bar.t().matmul(Xv - A_Xv))
    X_ = A_Xv + X_bar.matmul(tmp)

    return X_
