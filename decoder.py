import torch
import torch.nn as nn
from decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, forward_dim, num_decoder_layers):
        super(Decoder, self).__init__()
        self.num_decoder_layers = num_decoder_layers
        self.decoder = DecoderBlock(embedding_dim, n_heads, forward_dim)

    def forward(self, x, y):
        for _ in range(self.num_decoder_layers):
            y = self.decoder(x, y)
        return y

