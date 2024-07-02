import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, enc_embed, dec_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_embed = enc_embed
        self.dec_embed = dec_embed

    def forward(self, enc_in, dec_in, enc_timeF, dec_timeF,
                enc_mask=None, dec_mask=None):
        enc_out = self.encode(enc_in, enc_timeF, enc_mask)
        dec_out = self.decode(enc_out, dec_in, dec_timeF, enc_mask, dec_mask)
        return dec_out

    def encode(self, enc_in, enc_timeF,
               enc_mask=None):
        enc_in = self.enc_embed(enc_in, enc_timeF)
        enc_out = self.encoder(enc_in, enc_mask)
        return enc_out

    def decode(self, enc_out, dec_in, dec_timeF,
               enc_mask=None, dec_mask=None):
        dec_in = self.dec_embed(dec_in, dec_timeF)
        dec_out = self.decoder(dec_in, enc_out, enc_mask, dec_mask)
        return dec_out


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step
    (used at the last step of transformer in NLP).
    vocab: num of words in dict
    output: probability of words(classification problem)
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)