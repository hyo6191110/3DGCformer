import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np


def get_pad_mask(seq, pad_idx):
    # (batch, seqlen) -> (batch, 1, seqlen)
    return (seq != pad_idx).unsqueeze(-2)


def subsequent_mask(batchsz, T):
    """
    Mask out subsequent positions.
    """
    attn_shape = (batchsz, T, T)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def test1():
    attn_shape = (1, 5, 5)
    subsequent_mask1 = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask2 = (1 - torch.triu(torch.ones((attn_shape)), diagonal=1)).bool()
    print(torch.from_numpy(subsequent_mask1) == 0)
    print(subsequent_mask2)


def test2():
    a = torch.arange(9).reshape(1, 3, 3)
    print(a)
    print(a.repeat(2, 1, 1))


if __name__ == '__main__':
    list=range(10)
    print(list[:5])

