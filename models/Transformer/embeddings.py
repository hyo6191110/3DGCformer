import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np


class ValueEmbedding(nn.Module):
    """
    convert (batchsz, T, c_in) to (batchsz, T, d_model)
    """
    def __init__(self, d_model, c_in):
        super(ValueEmbedding, self).__init__()
        # self.lut = nn.Embedding(c_in, d_model) # 用于词嵌入,c_in是字典中词总数
        # self.lut(x) * math.sqrt(self.d_model) # x是long型
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')
        self.d_model = d_model

    def forward(self, x):
        # [N,T,c_in] -->[N,c_in, T]
        # [N,c_in, T] -->[N,d_model, T]
        # [N,d_model, T] -->[N,T,d_model]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class PositionalEncoding(nn.Module):
    """
    模型不包含递归和卷积结构，为了使模型能够有效利用序列的顺序特征，
    我们需要加入序列中各个Token间相对位置或Token在序列中绝对位置的信息。
    (将位置编码添加到编码器和解码器栈底部的输入Embedding)
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        pe.require_grad = False

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return Variable(self.pe[:, :x.size(1)])


class TimeFeatureEncoding(nn.Module):
    """
    input:x_timeF[N,T,5] type:long int
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TimeFeatureEncoding, self).__init__()
        self.freq = freq
        # x[:, :, 4] 取值 0-59
        minute_in_hour = 60
        # x[:, :, 3] 取值 0-23
        hour_in_day = 24
        # x[:, :, 2] 取值 0-6 (0对应星期1)
        day_in_week = 7
        # x[:, :, 1] 取值 0-30 (0对应X月1号)
        day_in_month = 31
        # x[:, :, 1] 取值 0-11 (0对应1月)
        month_in_year = 12

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        if freq == 'm':
            self.minute_embed = Embed(minute_in_hour, d_model)
        self.hour_embed = Embed(hour_in_day, d_model)
        self.weekday_embed = Embed(day_in_week, d_model)
        self.day_embed = Embed(day_in_month, d_model)
        self.month_embed = Embed(month_in_year, d_model)

    def forward(self, x_timeF):
        x_timeF = x_timeF.long()

        minute_x = self.minute_embed(x_timeF[:, :, 4]) if self.freq == 'm' else 0.
        hour_x = self.hour_embed(x_timeF[:, :, 3])
        weekday_x = self.weekday_embed(x_timeF[:, :, 2])
        day_x = self.day_embed(x_timeF[:, :, 1])
        month_x = self.month_embed(x_timeF[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(nn.Module):
    """
    接受Embedding的输出，对数据进行编码
    """
    def __init__(self, d_model, c_in, dropout, max_len=5000):
        super(DataEmbedding, self).__init__()
        self.skipValueEmbed = c_in == d_model
        self.value_embedding = ValueEmbedding(d_model=d_model, c_in=c_in)

        self.position_encoding = PositionalEncoding(d_model=d_model)
        self.time_encoding = TimeFeatureEncoding(d_model=d_model, embed_type='fixed', freq='h')

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_timeF):
        if not self.skipValueEmbed:
            x = self.value_embedding(x)
        # 由于位置编码与Embedding具有相同的维度 因此两者可以直接相加
        x = x + self.position_encoding(x) + self.time_encoding(x_timeF)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, c_in, patch_size, num_patch):
        super(PatchEmbedding, self).__init__()
        self.pad = nn.ZeroPad2d((0, patch_size * num_patch - c_in, 0, 0))
        self.conv = nn.Conv2d(in_channels=1, out_channels=patch_size,
                              kernel_size=(1, patch_size),
                              stride=(1, patch_size))

    def forward(self, x):
        # [N,TI,V*C] -->[N,1,TI,V*C]
        x = x.unsqueeze(1)
        # [N,1,TI,V*C] -->[N,1,TI,NP*P]
        x = self.pad(x)
        # [N,1,TI,NP*P] -->[N,P,TI,NP]
        x = self.conv(x)
        # [N,P,TI,NP] -->[N,NP,TI,P]
        x = x.permute(0, 3, 2, 1).contiguous()
        return x