import torch
import torch.nn as nn
from models.Transformer.utils import subsequent_mask
from models.baseline.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from models.baseline.layers.SelfAttention_Family import ProbAttention, AttentionLayer
from models.baseline.layers.Embed import DataEmbedding


class former_config():
    def __init__(self,C,V):
        self.moving_avg = 25
        self.enc_in = C*V
        self.dec_in = C*V
        self.c_out = C*V
        self.d_model = 512
        self.d_ff = 2048
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.05
        self.output_attention = False
        self.factor = 1
        self.activation = 'gelu'
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.distil = True


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, C, V,
                 T_i,
                 T_o,
                 T_label_i,
                 device):
        super(Informer, self).__init__()
        self.T_i = T_i
        self.T_label_i = T_label_i
        self.T_o = T_o
        self.device = device
        # param
        configs = former_config(C, V)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

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
        # make transformer input
        N, T, V, C = x.size()
        # [N,T=TI,V,C] -->[N,TI,V*C]
        x = x.view(N, T, V * C)
        x_enc, x_dec, x_mark_enc, x_mark_dec, dec_mask = self.makeTransformerInput(x, x_timeF, y_timeF)
        enc_self_mask = None
        dec_self_mask = dec_mask
        dec_enc_mask = None
        # informer
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # [B, L, D]
        y = dec_out[:, -self.T_o:, :]
        # [N,TO,V*C] -->[N,TO,V,C]
        y = y.view(N, self.T_o, V, C)

        return y
