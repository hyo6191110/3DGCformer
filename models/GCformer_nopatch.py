import torch
import torch.nn as nn
import os
import numpy as np
from models.Transformer.transformer import Transformer
from models.Transformer.utils import subsequent_mask
from utils.data_process import sym_norm_laplacian


class GCformer_nopatch(nn.Module):
    def __init__(self, C, V, device,
                 patch_size=512,
                 d_model=512,
                 d_ff=2048,
                 T_i=24,
                 T_o=24,
                 T_label_i=12,
                 pad_idx=0,
                 data='GZZone',
                 fusion='concat',
                 patch='embed',
                 dropout=0.2):
        super(GCformer_nopatch, self).__init__()
        # channel, region
        self.C = C
        self.V = V
        # 输入序列data的时间维度
        self.T_i = T_i
        self.T_o = T_o
        self.T_label_i = T_label_i
        # 卷积核时间维度大小
        self.kernel_time = 1
        self.pad_idx = pad_idx
        # res param for spatial dependency
        self.alpha = 0.5
        self.device = device

        # (1)-1 spatial module
        self.gcn_OD = nn.ModuleList([
            DGCN(C_in=1, C_out=16, kernel_size=2 * self.kernel_time + 1),
            DGCN(C_in=16, C_out=32, kernel_size=2 * self.kernel_time + 1),
            DGCN(C_in=32, C_out=16, kernel_size=2 * self.kernel_time + 1),
            DGCN(C_in=16, C_out=8, kernel_size=2 * self.kernel_time + 1),
        ])
        # (1)-2 spatial module
        self.gcn_neighbor = nn.ModuleList([
            DGCN(C_in=1, C_out=16, kernel_size=2 * self.kernel_time + 1, is_A_OD=False),
            DGCN(C_in=16, C_out=32, kernel_size=2 * self.kernel_time + 1, is_A_OD=False),
            DGCN(C_in=32, C_out=16, kernel_size=2 * self.kernel_time + 1, is_A_OD=False),
            DGCN(C_in=16, C_out=8, kernel_size=2 * self.kernel_time + 1, is_A_OD=False),
        ])
        # 获取地理邻接关系adj matrix
        A_path = os.path.join('data', data, 'neighbor_' + data + '.npy')
        # [V,V] -->[T,V,V]
        A_neighbor = sym_norm_laplacian(np.expand_dims(np.load(A_path), axis=0))
        A_neighbor = torch.from_numpy(A_neighbor).repeat(T_i, 1, 1)
        self.A_neighbor = A_neighbor.float().to(device)
        # (2) temporal module
        self.transformer = Transformer(c_enc=V*C, c_dec=V*C,
                                       N=6, d_model=d_model, d_ff=d_ff, h=8, dropout=0.1)
        # (F) fusion method
        self.fusion = fusion
        if fusion == 'concat':
            fuse_channel = 16
        else:
            fuse_channel = 8
        self.conv1 = nn.Conv3d(fuse_channel, 1, kernel_size=(1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm3d(num_features=1)
        # (E) output head
        self.proj1 = nn.Linear(d_model, V * C, bias=True)

    def makeTransformerInput(self, x, x_timeF, y_timeF):
        _, _, P = x.size()
        N, _, F = x_timeF.size()
        # 1. enc和dec的输入
        # [N,TI,P]
        enc_in = x
        # [N,TL,P],[N,TO,P] -->[M,TL+TO,P]
        dec_in = torch.zeros([N, self.T_o, P]).float().to(self.device)
        dec_in = torch.cat([enc_in[:, self.T_i - self.T_label_i:, :], dec_in], dim=1).float().to(self.device)
        # 2. enc和dec输入的时间特征
        enc_timeF = x_timeF
        dec_timeF = y_timeF
        # [N,T0,F] -->[N,TL+TO,F]
        dec_timeF = torch.cat([enc_timeF[:, self.T_i - self.T_label_i:, :], dec_timeF], dim=1)
        # 3. transformer输入的掩码
        dec_mask = subsequent_mask(N, self.T_label_i + self.T_o).to(self.device)
        return enc_in, dec_in, enc_timeF, dec_timeF, dec_mask

    def forward(self, x, A, x_timeF, y_timeF):
        # N=Batch size
        # T=Time windows in
        # V=Num of Region
        # C=Channel
        N, T, V, C = x.size()

        # 3DGCNs
        # [N,T,V,C] -->[N,1,T,V,C]
        x = x.unsqueeze(1)
        x_m = x
        # [N,1,T,V,C] -->[N,8,T,V,C]
        for gcn in self.gcn_OD:
            x = gcn(x, A)
        # [T,V,V] -->[N,T,V,V]
        A_neighbor = self.A_neighbor.unsqueeze(0).repeat(N, 1, 1, 1)
        # [N,1,T,V,C] -->[N,8,T,V,C]
        for gcn in self.gcn_neighbor:
            x_m = gcn(x_m, A_neighbor)

        # fuse 3DGCNs
        # [N,8,T,V,C],[N,8,T,V,C] -->[N,FC,T,V,C]
        if self.fusion == 'concat':
            x = torch.cat([x, x_m], dim=1)
        elif self.fusion == 'add':
            x = x + x_m
        elif self.fusion == 'mul':
            x = x * x_m

        x = self.dropout(x)
        # [N,FC,T,V,C] --> [N,1,T,V,C]
        x = self.relu(self.conv1(x))
        x = self.batchnorm(x)
        # [N,1,T,V,C] -->[N,T,V,C]
        x = x.squeeze(1)
        # [N,T=TI,V,C] -->[N,TI,V*C]
        x = x.view(N, T, V * C)

        # make transformer input
        enc_in, dec_in, enc_timeF, dec_timeF, dec_mask = self.makeTransformerInput(x, x_timeF, y_timeF)

        # transformer
        # [N,TI,V*C],[N,TI+TL,V*C] -->[N,TL+TO,d_model]
        dec_out = self.transformer(enc_in, dec_in, enc_timeF, dec_timeF, dec_mask=dec_mask)
        # [N,TL+TO,d_model] -->[N,TO,d_model]
        dec_out = dec_out[:, -self.T_o:, :]

        # output
        # [N,TO,d_model] -->[N,TO,V*C]
        y = self.proj1(dec_out)

        # [N,TO,V*C] -->[N,TO,V,C]
        y = y.view(N, self.T_o, V, C)
        return y


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