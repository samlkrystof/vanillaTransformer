import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


class PositionalEncoder(nn.Module):
    """
    Positional Encoder from "Attention is all you need"
    link: https://arxiv.org/abs/1706.03762

    """
    def __init__(self, embedding_dim: int =512, max_length: int =5000, dropout : int =0.1, **kwargs) -> None:
        super(PositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        self.pos_encoding = torch.empty(1, max_length, embedding_dim, **kwargs)

        indices = torch.arange(0, max_length, **kwargs).unsqueeze(1)
        pow_term = torch.pow(torch.full((embedding_dim // 2,), 10000, **kwargs),
                             -torch.arange(0, embedding_dim, 2, **kwargs).float() / embedding_dim)

        self.pos_encoding[0, :, 0::2] = torch.sin(indices * pow_term)
        self.pos_encoding[0, :, 1::2] = torch.cos(indices * pow_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (N,S,E) N: batch size, S: sequence length, E: embedding dimension
        :return: new tensor with positional encoding added
        """

        N, S, E = x.shape

        output = x + self.pos_encoding[0, :S]
        output = self.dropout(output)

        return output


class MultiHeadedAttention(nn.Module):
    """
    Multi Headed Attention from "Attention is all you need"
    """
    def __init__(self, embedding_dim, heads, dropout, **kwargs):
        super(MultiHeadedAttention, self).__init__()

        assert embedding_dim % heads == 0
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.size_per_head = embedding_dim // heads
        self.dropout = nn.Dropout(dropout)

        self.key = nn.Linear(embedding_dim, embedding_dim, **kwargs)
        self.query = nn.Linear(embedding_dim, embedding_dim, **kwargs)
        self.value = nn.Linear(embedding_dim, embedding_dim, **kwargs)

        self.projection = nn.Linear(embedding_dim, embedding_dim, **kwargs)

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """

        :param query: (N, S, E) N: batch size, S: sequence length, E: embedding dimension
        :param key: (N, T, E) T: sequence length
        :param value: (N, T, E)
        :param mask: (S, T)
        :return: output with shape (N, S, E)
        """

        N, S, E = query.shape

        key_trans, value_trans, query_trans = [layer(x).view(N, -1, self.heads, self.size_per_head).transpose(1, 2)
                                               for layer, x in
                                               zip((self.key, self.value, self.query), (key, value, query))]

        scores = torch.matmul(query_trans, key_trans.transpose(-1, -2)) / math.sqrt(self.size_per_head)
        if mask != None:
            scores = scores.masked_fill(mask.eq(0), -1e12)

        scores = self.dropout(scores)
        scores = torch.matmul(F.softmax(scores, dim=-1), value_trans)
        output = scores.transpose(1, 2).contiguous().view(N, S, E)

        return self.projection(output)


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, heads, feedforward_dim, dropout=0.1, **kwargs):
        super(EncoderBlock, self).__init__()
        num_sublayers = 2
        self.embedding_dim = embedding_dim
        self.heads = heads

        self.dropout = clones(nn.Dropout(dropout), num_sublayers)
        self.attention_module = MultiHeadedAttention(embedding_dim, heads, dropout, **kwargs)
        self.layer_norm = clones(nn.LayerNorm(embedding_dim, **kwargs), num_sublayers)
        self.feedforward = nn.Sequential(nn.Linear(embedding_dim, feedforward_dim, **kwargs),
                                         nn.ReLU(),
                                         nn.Linear(feedforward_dim, embedding_dim, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_headed = self.attention_module(x, x, x)
        multi_headed = self.dropout[0](multi_headed)

        norm = self.layer_norm[0](multi_headed + x)

        after_forward = self.feedforward(norm)
        after_forward = self.dropout[1](after_forward)

        second_norm = self.layer_norm[1](after_forward + norm)

        return second_norm


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, heads, feedforward_dim, dropout=0.1, **kwargs):
        super(DecoderBlock, self).__init__()
        num_sublayers = 3
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = clones(nn.Dropout(dropout), num_sublayers)
        self.masked_attention = MultiHeadedAttention(embedding_dim, heads, dropout, **kwargs)
        self.enc_dec_attention = MultiHeadedAttention(embedding_dim, heads, dropout, **kwargs)

        self.layer_norm = clones(nn.LayerNorm(embedding_dim, **kwargs), num_sublayers)
        self.feedforward = nn.Sequential(nn.Linear(embedding_dim, feedforward_dim, **kwargs),
                                         nn.ReLU(),
                                         nn.Linear(feedforward_dim, embedding_dim, **kwargs))

    def forward(self, encoder_x: torch.Tensor, decoder_x: torch.Tensor) -> torch.Tensor:
        """

        :param encoder_x: shape (N, T, E)
        :param decoder_x: shape (N, S, E)
        :return: shape (N, S, E)
        """
        N, S, E = decoder_x.shape
        mask = torch.tril(torch.ones(S, S, device=decoder_x.device, dtype=torch.uint8))
        result = self.masked_attention(decoder_x, decoder_x, decoder_x, mask)
        result = self.dropout[0](result)

        first_norm = self.layer_norm[0](decoder_x + result)

        cross_attention = self.enc_dec_attention(encoder_x, encoder_x, first_norm)
        cross_attention = self.dropout[1](cross_attention)

        second_norm = self.layer_norm[1](first_norm + cross_attention)

        feed_forward = self.feedforward(second_norm)
        feed_forward = self.dropout[2](feed_forward)

        third_norm = self.layer_norm[2](second_norm + feed_forward)

        return third_norm


class Transformer(nn.Module):
    def __init__(self, num_encoder_blocks, num_decoder_blocks, embedding_dim, heads, feedforward_dim, dropout,
                 max_length, num_classes, **kwargs):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim

        self.positional_encoding = PositionalEncoder(embedding_dim, max_length, dropout, **kwargs)
        self.encoder = nn.Sequential(*clones(EncoderBlock(embedding_dim, heads, feedforward_dim, dropout, **kwargs),
                                             num_encoder_blocks))
        self.decoder = nn.Sequential(*clones(DecoderBlock(embedding_dim, heads, feedforward_dim, dropout, **kwargs),
                                             num_decoder_blocks))
        self.projection = nn.Linear(embedding_dim, num_classes, **kwargs)

    def forward(self, encoder_x: torch.Tensor, decoder_x: torch.Tensor) -> torch.Tensor:
        pos_encoded_encoder = self.positional_encoding(encoder_x)
        encoder_y = self.encoder(pos_encoded_encoder)
        pos_encoded_decoder = self.positional_encoding(decoder_x)
        decoder_y = self.decoder(encoder_y, pos_encoded_decoder)
        output = self.projection(decoder_y)
        return output

