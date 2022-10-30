import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim=512, max_length=5000, dropout=0.1):
        super(PositionalEncoder, self)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        self.pos_encoding = torch.empty(1, max_length, embedding_dim)

        indices = torch.arange(0, max_length).unsqueeze(1)
        pow_term = torch.pow(torch.full((max_length // 2,), 10000), -torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        self.pos_encoding[0, :, 0::2] = torch.sin(indices * pow_term)
        self.pos_encoding[0, :, 1::2] = torch.cos(indices * pow_term)
        self.register_buffer("pos_encoding", self.pos_encoding)

    def forward(self, x):
        """

        :param x (N,S,D):
        :return:
        """

        N, S, D = x.shape

        output = x + self.pos_encoding[0, :S]
        output = self.dropout(output)

        return output





