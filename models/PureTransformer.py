import torch
import torch.nn as nn
import os
import numpy as np
from models.Transformer.transformer import Transformer
from models.Transformer.utils import subsequent_mask
from utils.data_process import sym_norm_laplacian
from models.Transformer.embeddings import PatchEmbedding


class PureTransformer(nn.Module):
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
        super(PureTransformer, self).__init__()
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
        # patch divide
        self.patch_size = patch_size
        self.num_patch = (V * C) // self.patch_size + 1
        self.patch_embed = PatchEmbedding(V * C, self.patch_size, self.num_patch)
        self.is_embed = False
        if patch == 'embed':
            self.is_embed = True
        elif patch == 'divide':
            self.is_embed = False

        # convert patch in decoding
        self.convert_dmodel_patchsize=True
        if d_model == patch_size:
            self.convert_dmodel_patchsize = False
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
        self.transformer = Transformer(c_enc=self.patch_size, c_dec=self.patch_size,
                                       N=6, d_model=d_model, d_ff=d_ff, h=8, dropout=dropout)
        # (F) fusion method
        self.fusion = fusion
        fuse_channel = 8
        self.conv1 = nn.Conv3d(fuse_channel, 1, kernel_size=(1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm3d(num_features=1)
        # (E) output head
        self.proj1 = nn.Linear(V * C, self.num_patch * self.patch_size, bias=True)
        self.proj2 = nn.Linear(d_model, self.patch_size, bias=True)
        self.proj3 = nn.Linear(self.num_patch * self.patch_size, V * C, bias=True)

    def makeTransformerInput(self, x, x_timeF, y_timeF):
        M, _, P = x.size()
        N, _, F = x_timeF.size()
        # 1. enc和dec的输入
        # [M,TI,P]
        enc_in = x
        # [M,TL,P],[M,TO,P] -->[M,TL+TO,P]
        dec_in = torch.zeros([M, self.T_o, P]).float().to(self.device)
        dec_in = torch.cat([enc_in[:, self.T_i - self.T_label_i:, :], dec_in], dim=1).float().to(self.device)
        # 2. enc和dec输入的时间特征
        # [N,TI,F] -->[M=N*NP,TI,F]
        enc_timeF = x_timeF.repeat(self.num_patch, 1, 1)
        # [N,TO,F] -->[M=N*NP,T0,F]
        dec_timeF = y_timeF.repeat(self.num_patch, 1, 1)
        # [M,T0,F] -->[M,TL+TO,F]
        dec_timeF = torch.cat([enc_timeF[:, self.T_i - self.T_label_i:, :], dec_timeF], dim=1)
        # 3. transformer输入的掩码
        dec_mask = subsequent_mask(M, self.T_label_i + self.T_o).to(self.device)
        return enc_in, dec_in, enc_timeF, dec_timeF, dec_mask

    def forward(self, x, A, x_timeF, y_timeF):
        # N=Batch size
        # T=Time windows in
        # V=Num of Region
        # C=Channel
        N, T, V, C = x.size()

        # 3DGCNs
        # [N,T=TI,V,C] -->[N,TI,V*C]
        x = x.view(N, T, V * C)

        # patch
        if self.is_embed:
            # patch embed
            # [N,TI,V*C] -->[N,NP,TI,P]
            x = self.patch_embed(x)
        else:
            # patch divide
            # [N,TI,V*C] -->[N,TI,NP*P]
            x = self.proj1(x)
            # [N,TI,NP*P] -->[N,NP,TI,P]
            x = x.view(N, T, self.num_patch, self.patch_size).permute(0, 2, 1, 3).contiguous()
        # [N,NP,TI,P] -->[N*NP,TI,P]
        x = x.view(N * self.num_patch, T, self.patch_size)

        # make transformer input
        enc_in, dec_in, enc_timeF, dec_timeF, dec_mask = self.makeTransformerInput(x, x_timeF, y_timeF)

        # transformer
        # [N*NP,TI,P],[N*NP,TL+TO,P] -->[N*NP,TL+TO,d_model]
        dec_out = self.transformer(enc_in, dec_in, enc_timeF, dec_timeF, dec_mask=dec_mask)
        # [N*NP,TL+TO,d_model] -->[N*NP,TO,d_model]
        dec_out = dec_out[:, -self.T_o:, :]

        # output
        # [N*NP,TO,d_model] -->[N*NP,TO,P]
        if self.convert_dmodel_patchsize:
            y = self.proj2(dec_out)
        else:
            y = dec_out
        # [N*NP,TO,P] -->[N,TO,NP,P] -->[N,TO,NP*P]
        y = y.view(N, self.num_patch, self.T_o, self.patch_size).permute(0, 2, 1, 3).contiguous()
        y = y.view(N, self.T_o, self.num_patch * self.patch_size)
        # [N,TO,NP*P] -->[N,TO,V*C]
        y = self.proj3(y)

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