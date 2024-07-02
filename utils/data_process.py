import numpy as np
import torch


class MinMaxScaler():
    def __init__(self):
        self.min = 0
        self.max = 0

    def fit_transform(self, data):
        self.min = data.min()
        self.max = data.max()
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class LogScaler():
    def __init__(self):
        self.log = 0

    def fit_transform(self, data):
        return np.log(data + 1.0)

    def inverse_transform(self, data):
        return np.exp(data) - 1.0


class MulScaler():
    def __init__(self, mul):
        self.mul = mul

    def fit_transform(self, data):
        return data * self.mul

    def inverse_transform(self, data):
        return data / self.mul


class AddScaler():
    def __init__(self, add):
        self.add = add

    def fit_transform(self, data):
        return data + self.add

    def inverse_transform(self, data):
        return data - self.add


def sym_norm_laplacian(A, deg_axis=0, add_loop=False):
    """
    input: A[T,V,V]
    deg_axis: 0--origin 1--destination
    output: symmetric normalized L[T,V,V] of A
    """
    T = A.shape[0]
    L = np.ones(A.shape)
    I = np.eye(A.shape[1])
    for t in range(T):
        adj = A[t]
        # add self-loop
        # 如果原始数据包括self-loop就不用
        if add_loop:
            adj = adj + I
        # degree matrix
        deg = adj.sum(axis=deg_axis)
        deg_sq_i = np.power(deg, -0.5).flatten()
        deg_sq_i[np.isinf(deg_sq_i)] = 0
        deg_sq_i = np.diag(deg_sq_i)
        L[t] = np.matmul(np.matmul(deg_sq_i, adj), deg_sq_i)
    return L


def diag_zero(OD):
    od = OD
    for i in range(od.shape[0]):
        for j in range(od.shape[1]):
            a = od[i][j]
            diag = torch.diag(a)
            diag = torch.diag_embed(diag)
            od[i][j] = a - diag
    return od


def y_dim_reduce(OD):
    # OD[N,T,V,C] -->[N,T,V,C]
    OD = torch.exp(OD) - 1.0
    ioFlow = OD.sum(axis=2)
    ioFlow = torch.log(ioFlow + 1.0)
    return ioFlow

