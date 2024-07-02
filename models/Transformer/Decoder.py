import numpy as np
import torch
import torch.nn as nn


from models.Transformer.layers import clones,LayerNorm,SublayerConnection


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder layer is made of self-attn, src-attn, and feed forward.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # 定义注意力方式
        self.self_attn = self_attn
        self.src_attn = src_attn
        # 定义FF层
        self.feed_forward = feed_forward
        # 每一层Encoder由3个子层组成，每一个应用res连接
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        # 第1个子层——实现了多头的 Self-attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第2个子层——实现了多头的 Src-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 第3个子层——简单的Positionwise的全连接前馈网络
        return self.sublayer[2](x, self.feed_forward)