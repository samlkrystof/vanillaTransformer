import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim=512, max_length=5000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        self.pos_encoding = torch.empty(1, max_length, embedding_dim)

        indices = torch.arange(0, max_length).unsqueeze(1)
        pow_term = torch.pow(torch.full((max_length // 2,), 10000),
                             -torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        self.pos_encoding[0, :, 0::2] = torch.sin(indices * pow_term)
        self.pos_encoding[0, :, 1::2] = torch.cos(indices * pow_term)
        self.register_buffer("pos_encoding", self.pos_encoding)

    def forward(self, x: torch.Tensor):
        """

        :param x: (N,S,E)
        :return:
        """

        N, S, E = x.shape

        output = x + self.pos_encoding[0, :S]
        output = self.dropout(output)

        return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, embedding_dim, heads, dropout):
        super().__init__()

        assert embedding_dim % heads == 0
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.size_per_head = embedding_dim // heads
        self.dropout = nn.Dropout(dropout)

        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """

        :param query: (N, S, E)
        :param key: (N, T, E)
        :param value: (N, T, E)
        :param mask: (S, T)
        :return:
        """

        N, T, E = key.shape
        N, S, E = query.shape

        key_trans, value_trans, query_trans = [layer(x).view(N, -1, self.heads, self.size_per_head).transpose(1, 2)
                                               for layer, x in
                                               zip((self.key, self.value, self.query), (key, value, query))]

        scores = torch.matmul(query_trans, key_trans.transpose(-1, -2)) / torch.sqrt(self.size_per_head)
        if mask != None:
            scores = scores.masked_fill(mask.eq(0), -1e12)

        scores = self.dropout(scores)
        scores = torch.matmul(F.softmax(scores, dim=-1), value_trans)
        output = scores.transpose(1, 2).contiguos().view(N, T, E)

        return self.projection(output)


