import torch
from torch.utils.data import Dataset
import json
import numpy as np
from utils.data_process import *


class HumanMobilityDataset(Dataset):
    def __init__(self, data, X_path, A_path, timeF_path,
                 T_i=5,  # input window size
                 T_o=1,  # output window size
                 type='train',
                 train_split=0.8,
                 normalize=0,
                 norm_param=0
                 ):
        # 1. 读取数据
        # dataset_X:[T,V,C]
        dataset_X = np.load(X_path)
        # dataset_A:[T,V,V]
        dataset_A_raw = np.load(A_path)
        # dataset_timeF:[T,C_TF=5]
        dataset_timeF = np.load(timeF_path)
        # T: 数据集总时间步
        T = dataset_X.shape[0]
        # V: num of nodes 区域的数量
        V = dataset_X.shape[1]
        # C: feature size
        C = dataset_X.shape[2]
        self.V = V
        self.C = C
        # 总共可以划分出多少个时间长为T_i + T_o的输入序列
        num_seq = T - (T_i + T_o) + 1
        # train:val:test 划分
        val_split = (1 - train_split) / 2
        num_train = int(num_seq * train_split)
        num_val = int(num_seq * val_split)
        num_test = int(num_seq * val_split)

        # 2:处理数据
        self.scaler_X = None
        if normalize == 1:
            self.scaler_X = MinMaxScaler()
        elif normalize == 2:
            self.scaler_X = LogScaler()
        elif normalize == 3:
            self.scaler_X = MulScaler(norm_param)
        elif normalize == 4:
            self.scaler_X = AddScaler(norm_param)
        # X归一化
        if normalize != 0:
            dataset_X = self.scaler_X.fit_transform(dataset_X)
        # 对 adj matrix进行处理
        dataset_A = np.zeros((T, 2, V, V), dtype=np.float32)
        # self-loop
        add_loop = False
        dataset_need_add_loop = ['NYC','NYC15m','NYC60m', 'WX', 'EZ']
        if data in dataset_need_add_loop:
            add_loop = True
        # Laplacian(origin)
        dataset_A[:, 0, :, :] = sym_norm_laplacian(dataset_A_raw, deg_axis=0, add_loop=add_loop)
        # Laplacian(dest)
        dataset_A[:, 1, :, :] = sym_norm_laplacian(dataset_A_raw, deg_axis=1, add_loop=add_loop)

        # 3. 数据集划分
        # train_set:[N=1762,C,T,V]
        if type == 'train':
            x = np.zeros((num_train, T_i, V, C), dtype=np.float32)
            y = np.zeros((num_train, T_o, V, C), dtype=np.float32)
            A = np.zeros((num_train, T_i, 2, V, V), dtype=np.float32)
            x_timeF = np.zeros((num_train, T_i, 5), dtype=np.int32)
            y_timeF = np.zeros((num_train, T_o, 5), dtype=np.int32)

            for t in range(num_train):
                x[t, :, :, :] = dataset_X[t:(t + T_i), :, :]
                y[t, :, :, :] = dataset_X[(t + T_i):(t + T_i + T_o), :, :]
                A[t, :, :, :, :] = dataset_A[t:(t + T_i), :, :, :]
                x_timeF[t, :, :] = dataset_timeF[t:(t + T_i), :]
                y_timeF[t, :, :] = dataset_timeF[(t + T_i):(t + T_i + T_o), :]

        # val_set: [N=220,C,T,V]
        elif type == 'val':
            x = np.zeros((num_val, T_i, V, C), dtype=np.float32)
            y = np.zeros((num_val, T_o, V, C), dtype=np.float32)
            A = np.zeros((num_val, T_i, 2, V, V), dtype=np.float32)
            x_timeF = np.zeros((num_val, T_i, 5), dtype=np.int32)
            y_timeF = np.zeros((num_val, T_o, 5), dtype=np.int32)

            for t in range(num_train, num_train + num_val):
                x[t - num_train, :, :, :] = dataset_X[t:(t + T_i), :, :]
                y[t - num_train, :, :, :] = dataset_X[(t + T_i):(t + T_i + T_o), :, :]
                A[t - num_train, :, :, :, :] = dataset_A[t:(t + T_i), :, :, :]
                x_timeF[t - num_train, :, :] = dataset_timeF[t:(t + T_i), :]
                y_timeF[t - num_train, :, :] = dataset_timeF[(t + T_i):(t + T_i + T_o), :]

        # test_set:[N=220,C,T,V]
        elif type == 'test':
            x = np.zeros((num_test, T_i, V, C), dtype=np.float32)
            y = np.zeros((num_test, T_o, V, C), dtype=np.float32)
            A = np.zeros((num_test, T_i, 2, V, V), dtype=np.float32)
            x_timeF = np.zeros((num_test, T_i, 5), dtype=np.int32)
            y_timeF = np.zeros((num_test, T_o, 5), dtype=np.int32)

            for t in range(num_train + num_val, num_train + num_val + num_test):
                x[t - num_train - num_val, :, :, :] = dataset_X[t:(t + T_i), :, :]
                y[t - num_train - num_val, :, :, :] = dataset_X[(t + T_i):(t + T_i + T_o), :, :]
                A[t - num_train - num_val, :, :, :, :] = dataset_A[t:(t + T_i), :, :, :]
                x_timeF[t - num_train - num_val, :, :] = dataset_timeF[t:(t + T_i), :]
                y_timeF[t - num_train - num_val, :, :] = dataset_timeF[(t + T_i):(t + T_i + T_o), :]

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.A = torch.from_numpy(A)
        self.x_timeF = torch.from_numpy(x_timeF)
        self.y_timeF = torch.from_numpy(y_timeF)
        self.len = x.shape[0]
        print(type, 'X shape')
        print(x.shape)
        print(type, 'Y shape')
        print(y.shape)

    def __getitem__(self, index):
        return self.x[index], self.A[index], self.y[index], self.x_timeF[index], self.y_timeF[index]

    def __len__(self):
        return self.len

    def numRegions(self):
        return self.V

    def numChannels(self):
        return self.C


if __name__ == '__main__':
    path='D:\Projects\PycharmProjects\HumanFlow\HMPred-Graph\data\GZZone\OD_GZZone.npy'
    od=np.load(path)
    od=torch.from_numpy(od)
    for i in range(od.shape[0]):
        a=od[i]
        diag = torch.diag(a)
        diag = torch.diag_embed(diag)
        od[i]=a-diag
    print(od[1])