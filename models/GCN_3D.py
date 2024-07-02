import torch
import torch.nn as nn
from utils.data_process import sym_norm_laplacian
import os
import numpy as np


class GCN_3D(nn.Module):
    def __init__(self, C, V, device,
                 patch_size=512,
                 d_model=512,
                 d_ff=2048,
                 T_i=24,
                 T_o=24,
                 T_label_i=12,
                 pad_idx=0,
                 data='GZZone',
                 fusion='concat'):
        super(GCN_3D, self).__init__()
        # channel
        self.C = C
        # 输入序列data的时间维度
        self.T_i = T_i
        self.T_o = T_o
        # 卷积核时间维度大小
        self.kernel_time = 1
        # res param for spatial dependency
        self.alpha = 0.5
        self.device=device

        self.gcn_OD = nn.ModuleList([
            DGCN(C_in=1, C_out=16, kernel_size=2 * self.kernel_time + 1),
            DGCN(C_in=16, C_out=32, kernel_size=2 * self.kernel_time + 1),
            DGCN(C_in=32, C_out=16, kernel_size=2 * self.kernel_time + 1),
            DGCN(C_in=16, C_out=8, kernel_size=2 * self.kernel_time + 1),
        ])

        self.gcn_dist = nn.ModuleList([
            DGCN(C_in=1, C_out=16, kernel_size=2 * self.kernel_time + 1, is_A_OD=False),
            DGCN(C_in=16, C_out=32, kernel_size=2 * self.kernel_time + 1, is_A_OD=False),
            DGCN(C_in=32, C_out=16, kernel_size=2 * self.kernel_time + 1, is_A_OD=False),
            DGCN(C_in=16, C_out=8, kernel_size=2 * self.kernel_time + 1, is_A_OD=False),
        ])
        # 获取物理距离adj matrix
        A_path = os.path.join('data', data, 'neighbor_' + data + '.npy')
        # [V,V] -->[T,V,V]
        A_dist = sym_norm_laplacian(np.expand_dims(np.load(A_path), axis=0))
        A_dist = torch.from_numpy(A_dist).repeat(T_i, 1, 1)
        self.A_dist = A_dist.float().to(device)

        self.conv1 = nn.Conv3d(16, 1, kernel_size=(1, 1, 1))
        self.proj1 = nn.Linear(T_i, T_o, bias=True)

    def forward(self, x, A, x_timeF, y_timeF):
        # N=Batch size
        # T=Time windows in
        # V=Num of Region
        # C=Channel
        N, T, V, C = x.size()
        # [N,T,V,C] -->[N,1,T,V,C]
        x = x.unsqueeze(1)
        x_m = x

        # [N,1,T,V,C] -->[N,8,T,V,C]
        for gcn in self.gcn_OD:
            x = gcn(x, A)

        # [T,V,V] -->[N,T,V,V]
        A_dist = self.A_dist.unsqueeze(0).repeat(N, 1, 1, 1)
        # [N,1,T,V,C] -->[N,8,T,V,C]
        for gcn in self.gcn_dist:
            x_m = gcn(x_m, A_dist)

        # residual connection
        # x = self.alpha * x + (1 - self.alpha) * x_m
        # [N,8,T,V,C],[N,8,T,V,C] -->[N,16,T,V,C]
        x = torch.cat([x, x_m], dim=1)
        # [N,16,T,V,C] --> [N,1,T,V,C]
        x = self.conv1(x)
        # [N,1,T,V,C] -->[N,T,V,C]
        x = x.squeeze(1)

        # [N,T,V,C] -->[N,V,C,T]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [N,V,C,TI] -->[N,V,C,TO]
        x = self.proj1(x)
        # [N,V,C,TO] -->[N,TO,V,C]
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class DGCN(nn.Module):
    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size,
                 is_A_OD=True):
        super(DGCN, self).__init__()
        self.C_out = C_out
        # 使用的adj matrix是[N,T,2,V,V]的基于OD的矩阵
        # 如果为false，就是[N,T,V,V]的基于物理距离的矩阵
        self.is_A_OD = is_A_OD
        self.conv = nn.Sequential(nn.BatchNorm3d(C_in),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(C_in,
                                            C_out,
                                            kernel_size=(kernel_size, 1, 1),
                                            padding=(1, 0, 0),
                                            stride=(1, 1, 1),
                                            bias=True),
                                  nn.BatchNorm3d(C_out),
                                  nn.Dropout3d(p=0.2, inplace=True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # X[N,C_IN,T,V,C]
        # A[N,T,2,V,V] OR [N,T,V,V]
        N, C_in, T, V, C = x.size()

        # [N,C_IN,T,V,C] --> [N*C_IN*T,V,C]
        x = x.view(N * C_in * T, V, C)
        # [N,T,V,V] --> [N*C_IN*T,V,V]
        if self.is_A_OD:
            A_origin = A[:, :, 0, :, :].unsqueeze(1).repeat(1, C_in, 1, 1, 1).view(N * C_in * T, V, V)
            A_dest = A[:, :, 1, :, :].unsqueeze(1).repeat(1, C_in, 1, 1, 1).view(N * C_in * T, V, V)
        else:
            A_origin = A.unsqueeze(1).repeat(1, C_in, 1, 1, 1).view(N * C_in * T, V, V)
            A_dest = None

        # Channel: OD flow
        if self.is_A_OD and C == V:
            # ([M,V,V]*[M,V,C])*[M,V,V] --> [M,V,C]
            x = torch.bmm(torch.bmm(A_origin, x), A_dest)
        else:
            # [M,V,V]*[M,V,C] --> [M,V,C]
            x = torch.bmm(A_origin, x)
        # [M=N*C_IN*T,V,C] --> [N,C_IN,T,V,C]
        x = x.view(N, C_in, T, V, C)
        # [N,C_IN,T,V,C] --> [N,C_OUT,T,V,C]
        x = self.conv(x)
        x = self.relu(x)

        return x