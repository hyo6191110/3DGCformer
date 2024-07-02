import torch.nn as nn
import math, copy, time
from models.Transformer.Encoder import Encoder, EncoderLayer
from models.Transformer.Decoder import Decoder, DecoderLayer
from models.Transformer.EncoderDecoder import EncoderDecoder, Generator
from models.Transformer.utils import PositionwiseFeedForward
from models.Transformer.embeddings import ValueEmbedding, DataEmbedding
from models.Transformer.attention import MultiHeadedAttention


class Transformer(nn.Module):
    """
    Construct a model from hyperparameters.
    """
    def __init__(self, c_enc, c_dec, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        # 定义注意力机制
        attn = MultiHeadedAttention(h, d_model)
        # 定义前馈网络层
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 模型框架
        self.model = EncoderDecoder(
            encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            enc_embed=DataEmbedding(d_model, c_enc, dropout),
            dec_embed=DataEmbedding(d_model, c_dec, dropout)
        )
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_in, dec_in, enc_timeF, dec_timeF,
                enc_mask=None, dec_mask=None):
        return self.model(enc_in, dec_in, enc_timeF, dec_timeF, enc_mask, dec_mask)


if __name__ == '__main__':
    # Small example model.
    transformer=Transformer(10,10,2)
