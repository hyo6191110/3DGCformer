import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from models.Transformer.layers import clones

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'.
    """
    # Query和Key的维度
    d_k = query.size(-1)
    # 计算Query与各个Key的点积，使用矩阵并行运算
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 最后使用Softmax函数来获得Key的权重
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 返回Key权重与Value相乘的值
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        # 模型维度需要能被h整除
        assert d_model % h == 0
        # 假设 d_v == d_k 总是成立
        # 每个头负责d_k维度，加起来就是d_model
        self.d_k = d_model // h
        self.h = h
        # 0:W_Q[h*d_k, d_model] 1:W_K[h*d_k, d_model]
        # 2:W_V[h*d_v, d_model] 3:W_O[d_model, d_model]
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 同样的mask应用于全部h个head
            mask = mask.unsqueeze(1)
        # qkv的样本个数
        # X[nbatches,d_model]
        # Q[nbatches,d_k] K[nbatches,d_k] V[nbatches,d_v]
        nbatches = query.size(0)

        # 1) 对输入进行线性投影，并将d_model => h*d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) 对qkv应用attention，对结果进行拼接
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) 经过最后的linear层
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)