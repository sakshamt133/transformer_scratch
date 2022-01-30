import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, embedding_dim, n_heads, forward_dim, num_encoder_layers, num_decoder_layers):
        super(Transformer, self).__init__()
        self.encoder = Encoder(embedding_dim, n_heads, forward_dim, num_encoder_layers)
        self.decoder = Decoder(embedding_dim, n_heads, forward_dim, num_decoder_layers)

    def forward(self, x, y):
        encoder_out = self.encoder(x)
        out = self.decoder(encoder_out, y)
        return out
