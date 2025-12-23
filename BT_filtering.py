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
    A = torch.sparse.addmm(I_n, A, torch.ones(A.shape[0], A.shape[1]).to(device))
    D = torch.sum(A, 1)
    D = torch.pow(D, -0.5)
    D[torch.isinf(D)] = 0
    D = torch.diag(D)
    A = D.matmul(A).matmul(D)
    Ls = I_n - A
    G = I_n - 0.5 * Ls
    A_ = I_n
    f = 2

    for _ in range(f):
        A_ = G.matmul(A_)
    G_ = G
    kk = 1
    k = 5 # [1,2,3,4,5,7,10]
    a = 50# [0.1,1,10,100,1000]

    while (kk <= k):
        X_bar = G_.matmul(X)
        XtX_bar = X_bar.matmul(X_bar.T)
        XXt_bar = X_bar.t().matmul(X_bar)
        tmp = torch.inverse(I_d + XXt_bar / a)
        tmp = X_bar.matmul(tmp).matmul((X_bar.t()))
        tmp = I_n / a - tmp / (a * a)
        S = tmp.matmul(a * A_ + XtX_bar)
        kk += 1
        G_ = G_.matmul(G)
    X_ = S.matmul(Xv)

    return X_


