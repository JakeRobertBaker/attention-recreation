import math
from collections.abc import Iterable
from typing import Callable

import torch
from torch import LongTensor, Tensor, nn
from torch.nn import Module


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, d_q, d_k, d_v):
        super().__init__()
        assert d_model % n_heads == 0, "d_model is not divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        self.proj_q = nn.Linear(d_q, d_model)
        self.proj_k = nn.Linear(d_k, d_model)
        self.proj_v = nn.Linear(d_v, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        source_key_padding_mask: Tensor | None = None,
    ):
        """_summary_

        Args:
            query (Tensor): shape (N, t, d_q)
            key (Tensor): shape (N, s, d_k)
            value (Tensor): shape (N, t, d_v)
            source_key_padding_mask (Tensor): shape (N,s)
        """
        if isinstance(source_key_padding_mask, Tensor) and isinstance(attn_mask, Tensor):
            raise ValueError(
                "Cannot have both source_key_padding_mask and attn_mask as args since source_key_padding_mask defines a attn_mask."
            )

        batch_size, target_length, _ = query.shape
        _, source_length, _ = key.shape

        if isinstance(source_key_padding_mask, Tensor):
            # (N,s) -> (N,1,s) -> (N,t,s)
            attn_mask = source_key_padding_mask.unsqueeze(1).expand(batch_size, target_length, source_length).unsqueeze(1)

        # want 

        # query does: (N, t, d_q) -> (N, t, d_model) -> (N, t, n_heads, d_head) -> (N, n_heads, t, d_head)
        q = self.proj_q.forward(query).view(batch_size, target_length, self.n_heads, self.d_head).transpose(1,2)
        k = self.proj_k.forward(key).view(batch_size, source_length, self.n_heads, self.d_head).transpose(1,2)
        v = self.proj_v.forward(value).view(batch_size, source_length, self.n_heads, self.d_head).transpose(1,2)

        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        # (N, n_heads, t, d_head) -> (N, t, n_heads, d_head) -> (N, t, d_model)
        y = y.transpose(1,2).view(batch_size, target_length, self.d_model)
        return y


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
        self.mha_sublayer = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_q=d_model, d_k=d_model, d_v=d_model)
        self.mha_dropout = nn.Dropout(p_dropout)

        self.mlp_layer_norm = nn.LayerNorm(d_model)
        self.mlp_sublayer = nn.Sequential(nn.Linear(d_model, d_inner_layer), nn.Linear(d_inner_layer, d_model), nn.GELU())
        self.mlp_dropout = nn.Dropout(p_dropout)

    def forward(self, source: Tensor, source_key_padding_mask: Tensor) -> Tensor:
        """
        Args:
            source (Tensor): shape (N, s, d_model)
            source_key_padding_mask (Tensor): shape (N, s), masking applied to source pad keys

        Returns:
            Tensor: shape (N, s, d_model)
        """
        x = self.mha_layer_norm(source)
        y = self.mha_sublayer.forward(query=x, key=x, value=x, source_key_padding_mask=source_key_padding_mask)
        y = self.mha_dropout(y) + x

        x = self.mlp_layer_norm(y)
        y = self.mlp_sublayer(x)
        y = self.mlp_dropout(y) + x
        return y


class Encoder(Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        n_stacks: int,
        p_dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_layer = d_inner_layer
        self.n_heads = n_heads
        self.n_stacks = n_stacks

        self.layers: Iterable[EncoderLayer] = nn.ModuleList(
            [
                EncoderLayer(d_model=d_model, d_inner_layer=d_inner_layer, n_heads=n_heads, p_dropout=p_dropout)
                for i in range(n_stacks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, source: Tensor, source_key_padding_mask: Tensor) -> Tensor:
        """Apply the Encoder to encoded input IDs.

        Args:
            source (Tensor): shape (N, s, d_model)
            source_key_padding_mask (Tensor): shape (N, s), masking applied to source pad keys

        Returns:
            Tensor: shape (N, s, d_model)
        """
        for layer in self.layers:
            source = layer.forward(source=source, source_key_padding_mask=source_key_padding_mask)
        # we apply layer norm before each sublayer, thus need final layer norm after layer[-1]
        source = self.final_layer_norm(source)
        return source


class DecoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        p_dropout: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_layer = d_inner_layer
        self.n_heads = n_heads

        self.self_mha_layer_norm = nn.LayerNorm(d_model)
        self.self_mha_sublayer = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_q=d_model, d_k=d_model, d_v=d_model)
        self.self_mha_dropout = nn.Dropout(p_dropout)

        self.cross_mha_layer_norm = nn.LayerNorm(d_model)
        self.cross_mha_sublayer = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_q=d_model, d_k=d_model, d_v=d_model)
        self.cross_mha_dropout = nn.Dropout(p_dropout)

        self.mlp_layer_norm = nn.LayerNorm(d_model)
        self.mlp_sublayer = nn.Sequential(nn.Linear(d_model, d_inner_layer), nn.Linear(d_inner_layer, d_model), nn.GELU())
        self.mlp_dropout = nn.Dropout(p_dropout)

    def forward(self, target: Tensor, source: Tensor, source_key_padding_mask: Tensor):
        """Decoder layer applies


        Args:
            target (Tensor): shape (N, t, d_model), used for queries in cross mha
            source (Tensor): shape (N, s, d_model), used for keys and values in cross mha
            source_key_padding_mask (Tensor): shape (N, s), masking applied to source pad keys
        """

        # target self attention has causal mask activated.
        target_x = self.self_mha_layer_norm(target)
        target_y = self.self_mha_sublayer.forward(query=target_x, key=target_x, value=target_x, is_causal=True)
        target_y = self.self_mha_dropout(target_y) + target_x

        # cross attention has a padding mask
        target_x = self.cross_mha_layer_norm(target_y)
        y = self.cross_mha_sublayer.forward(
            query=target_x,
            key=source,
            value=source,
            source_key_padding_mask=source_key_padding_mask,
        )
        y = self.cross_mha_dropout(y) + target_x

        x = self.mlp_layer_norm(y)
        y = self.mlp_sublayer(x)
        y = self.mlp_dropout(y) + x

        return y


class Decoder(Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        n_stacks: int,
        p_dropout: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_layer = d_inner_layer
        self.n_heads = n_heads
        self.n_stacks = n_stacks

        self.layers: Iterable[DecoderLayer] = nn.ModuleList(
            [
                DecoderLayer(d_model=d_model, d_inner_layer=d_inner_layer, n_heads=n_heads, p_dropout=p_dropout)
                for i in range(n_stacks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, target: Tensor, source: Tensor, source_key_padding_mask: Tensor):
        """Applies decoing to shifted right target and source tensor

        Args:
            target (Tensor): shape (N, t, d_model)
            source (Tensor): shape (N, s, d_model)
            source_key_padding_mask (Tensor): shape (N, s), masking applied to source pad keys
        """
        for layer in self.layers:
            target = layer.forward(target=target, source=source, source_key_padding_mask=source_key_padding_mask)

        target = self.final_layer_norm(target)
        return target


class EncoderDecoder(Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        n_encoder_stacks: int,
        n_decoder_stacks: int,
        source_vocab_size: int,
        target_vocab_size: int | None,
        input_id_encoder_factory: Callable[..., InputIdEncoder],
        encoder_factory: Callable[..., Encoder],
        decoder_factory: Callable[..., Decoder],
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

        # if target vocab size is None then assume target and source are same language
        if target_vocab_size:
            self.target_id_encoder = input_id_encoder_factory(vocab_size=target_vocab_size, d_model=d_model, p_dropout=p_dropout)
        else:
            self.target_id_encoder = self.source_id_encoder

        self.encoder = encoder_factory(
            d_model=d_model,
            d_inner_layer=d_inner_layer,
            n_stacks=n_encoder_stacks,
            n_heads=n_heads,
            p_dropout=p_dropout,
        )

        self.decoder = decoder_factory(
            d_model=d_model,
            d_inner_layer=d_inner_layer,
            n_stacks=n_encoder_stacks,
            n_heads=n_heads,
            p_dropout=p_dropout,
        )

    def forward(self, target_ids: LongTensor, source_ids: LongTensor, source_key_padding_mask: Tensor):
        """
        Apply full encoder decoder.

        Args:
            source (Tensor): shape (N, s, d_model)
            target (Tensor): shape (N, t, d_model)
        """

        source = self.source_id_encoder.forward(source_ids)
        target = self.target_id_encoder.forward(target_ids)

        encoded_source = self.encoder.forward(
            source=source,
            source_key_padding_mask=source_key_padding_mask,
        )

        decoded_target = self.decoder.forward(
            target=target,
            source=encoded_source,
            source_key_padding_mask=source_key_padding_mask,
        )

        # TODO final projection into target embed space


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
) -> Encoder:
    return Encoder(
        d_model=d_model,
        d_inner_layer=d_inner_layer,
        n_heads=n_heads,
        n_stacks=n_stacks,
        p_dropout=p_dropout,
    )


def make_decoder(
    d_model: int,
    d_inner_layer: int,
    n_heads: int,
    n_stacks: int,
    p_dropout: float,
) -> Decoder:
    return Decoder(
        d_model=d_model,
        d_inner_layer=d_inner_layer,
        n_heads=n_heads,
        n_stacks=n_stacks,
        p_dropout=p_dropout,
    )
