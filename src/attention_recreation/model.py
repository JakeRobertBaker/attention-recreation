import math
from collections.abc import Iterable
from typing import Callable

import torch
from torch import LongTensor, Tensor, nn
from torch.nn import Module


class SinePositionEncoder(Module):
    def __init__(self, d_model: int, p_dropout: float, max_seq_length: int = 5_000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_dropout)

        pe = torch.empty((max_seq_length, d_model))
        dim_index = torch.arange(0, d_model, 2)
        div_term = torch.exp(-math.log(10_000) * dim_index / d_model)  # shape (d_model/2)
        positions = torch.arange(0, max_seq_length).reshape(-1, 1)  # shape (max_seq_length, 1)
        terms = positions * div_term  # shape (max_seq_length, d_model/2)

        pe[:, 0::2] = torch.sin(terms)
        pe[:, 1::2] = torch.cos(terms)
        # generalise shape for batch sizes: (max_seq_length, d_model)-> (1,max_seq_length, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): shape (N, t, d_model)

        Returns:
            Tensor: shape (N, t, d_model)
        """
        x = x + self.pe[:, : x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class InputIdEncoder(Module):
    def __init__(self, vocab_size: int, d_model: int, positional_encoder: Module):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoder = positional_encoder

    def forward(self, x: LongTensor) -> Tensor:
        """Maps the input id space to the embedding space.

        Args:
            x (Tensor): (N, t) with batch_size N and sequence length t.
            Each element is an input id is an integer in the range [0, vocab_size).

        Returns:
            Tensor: (N, t, d_model)
        """
        embedding = self.token_encoder(x)  # shape (N, t, d_model)
        embedding = self.positional_encoder(embedding)
        return embedding


def make_input_id_encoder(
    d_model: int,
    p_dropout: float,
    vocab_size: int,
) -> InputIdEncoder:
    positional_encoder = SinePositionEncoder(d_model, p_dropout)
    return InputIdEncoder(vocab_size, d_model, positional_encoder)


def make_encoder(
    d_model: int,
    d_inner_layer: int,
    n_heads: int,
    n_stacks: int,
    p_dropout: float,
) -> Module:
    return Encoder(
        d_model=d_model,
        d_inner_layer=d_inner_layer,
        n_heads=n_heads,
        n_stacks=n_stacks,
        p_dropout=p_dropout,
        encoder_layer_factory=make_encoder_layer,
    )


def make_encoder_layer(
    d_model: int,
    d_inner_layer: int,
    n_heads: int,
    p_dropout: float,
) -> Module:
    return EncoderLayer(
        d_model=d_model,
        d_inner_layer=d_inner_layer,
        n_heads=n_heads,
        p_dropout=p_dropout,
    )


class EncoderDecoder(Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        n_encoder_stacks: int,
        n_decoder_stacks: int,
        source_vocab_size: int,
        target_vocab_size: int,
        input_id_encoder_factory: Callable[..., Module],
        encoder_factory: Callable[..., Module],
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_layer = d_inner_layer
        self.n_heads = n_heads
        self.n_encoder_stacks = n_encoder_stacks
        self.n_decoder_stacks = n_decoder_stacks
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        self.source_id_encoder = input_id_encoder_factory(vocab_size=source_vocab_size, d_model=d_model, p_dropout=p_dropout)
        self.target_id_encoder = input_id_encoder_factory(vocab_size=target_vocab_size, d_model=d_model, p_dropout=p_dropout)

        self.encoder = encoder_factory(
            d_model=d_model,
            d_inner_layer=d_inner_layer,
            n_stacks=n_encoder_stacks,
            n_heads=n_heads,
            p_dropout=p_dropout,
        )


class Encoder(Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        n_stacks: int,
        p_dropout: float,
        encoder_layer_factory: Callable[..., Module],
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_layer = d_inner_layer
        self.n_heads = n_heads
        self.n_stacks = n_stacks

        self.layers: Iterable[Module] = nn.ModuleList(
            [
                encoder_layer_factory(d_model=d_model, d_inner_layer=d_inner_layer, n_heads=n_heads, p_dropout=p_dropout)
                for i in range(n_stacks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Apply the Encoder laters to the encoded input ids.

        Args:
            x (Tensor): shape (N, t, d_model)

        Returns:
            Tensor: shape (N, t, d_model)
        """
        for layer in self.layers:
            x = layer.forward(x, mask)
        # we apply layer norm before each sublayer, thus need final layer norm after layer[-1]
        x = self.final_layer_norm(x)
        return x


class EncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        p_dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_layer = d_inner_layer
        self.n_heads = n_heads

        self.mha_layer_norm = nn.LayerNorm(d_model)
        self.mha_sublayer = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.mha_dropout = nn.Dropout(p_dropout)

        self.mlp_layer_norm = nn.LayerNorm(d_model)
        self.mlp_sublayer = nn.Sequential(nn.Linear(d_model, d_inner_layer), nn.Linear(d_inner_layer, d_model), nn.GELU())
        self.mlp_dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape (N, t, d_model)
            mask (Tensor): shape (t,t)

        Returns:
            Tensor: shape (N, t, d_model)
        """
        y = self.mha_layer_norm(x)
        y = self.mha_sublayer(y)
        y = self.mha_dropout(y) + x

        z = self.mlp_layer_norm(y)
        z = self.mlp_sublayer(z, mask)
        z = self.mlp_dropout(z) + y
        return z
