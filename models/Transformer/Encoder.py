import torch.nn as nn


from models.Transformer.layers import clones,LayerNorm,SublayerConnection


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder layer is made up of self-attn and feed forward.
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        # 定义注意力方式
        self.self_attn = self_attn
        # 定义FF层
        self.feed_forward = feed_forward
        # 每一层Encoder由2个子层组成，每一个应用res连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # 第1个子层——实现了多头的 Self-attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第2个子层——简单的Positionwise的全连接前馈网络
        return self.sublayer[1](x, self.feed_forward)