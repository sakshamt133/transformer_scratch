import torch.nn as nn
from encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, forward_dim, num_encoder_layers):
        super(Encoder, self).__init__()
        self.encoder_block = EncoderBlock(embedding_dim, n_heads, forward_dim)
        self.num_encoder_layers = num_encoder_layers

    def forward(self, x, src_mask=None):
        for i in range(self.num_encoder_layers):
            x = self.encoder_block(x, src_mask)

        return x
