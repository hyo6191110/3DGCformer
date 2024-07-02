import torch
import torch.nn as nn
from models.Transformer.utils import subsequent_mask
from models.baseline.layers.Embed import DataEmbedding_wo_pos
from models.baseline.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.baseline.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


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

class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, C, V,
                 T_i,
                 T_o,
                 T_label_i,
                 device):
        super(Autoformer, self).__init__()
        self.T_i = T_i
        self.T_label_i = T_label_i
        self.T_o = T_o
        self.device = device
        # param
        configs = former_config(C,V)

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
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
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.T_o, 1)
        zeros = torch.zeros([x_dec.shape[0], self.T_o, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.T_label_i:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.T_label_i:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        # [B, L, D]
        y = dec_out[:, -self.T_o:, :]
        # [N,TO,V*C] -->[N,TO,V,C]
        y = y.view(N, self.T_o, V, C)

        return y
