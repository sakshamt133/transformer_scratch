import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.query_size = embedding_dim//self.n_heads
        self.linear = nn.Linear(self.query_size, self.query_size, bias=False)

    def forward(self, key, query, value, mask=None):
        batch, key_len, _ = key.shape
        query_len = query.shape[1]
        key = key.reshape(batch, key_len, self.n_heads, self.query_size)
        query = query.reshape(batch, -1, self.n_heads, self.query_size)
        value = value.reshape(batch, -1, self.n_heads, self.query_size)
        key = self.linear(key)
        query = self.linear(query)
        value = self.linear(value)
        filter_mat = torch.einsum("nqhe, nkhe -> nhqk", query, key)

        if mask is not None:
            pass

        softmax_filter = torch.softmax(filter_mat, dim=3)
        filter = torch.einsum("nhqk, nvhe -> nqhe", softmax_filter, value)
        filter = filter.reshape(batch, query_len, -1)
        return filter
