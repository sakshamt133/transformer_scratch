import torch
import torch.nn as nn
from self_attention import SelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, forward_dim):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, n_heads)
        self.forward_expansion = nn.Sequential(
            nn.Linear(embedding_dim, forward_dim * embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_dim * embedding_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x, y):
        out = self.attention(y, y, y)
        out = out + y
        out = self.attention(out, x, out)
        out = self.forward_expansion(out)
        return out

