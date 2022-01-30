import torch
import torch.nn as nn
from self_attention import SelfAttention


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, forward_dim):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, n_heads)
        self.forward_expansion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * forward_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim * forward_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x, src_mask=None):
        out = self.attention(x, x, x, src_mask)
        out = out + x
        out = self.forward_expansion(out)
        return out

#
# eb = EncoderBlock(100, 5, 2)
# input = torch.randn((2, 15, 100))
# a = eb(input)
# print(a.shape)
