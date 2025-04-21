import torch
from torch import Tensor
from torch import nn
import math


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        n_encoder_stacks: int,
        n_decoder_stacks: int,
        input_id_encoder: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_later = d_inner_layer
        self.n_heads = n_heads
        self.n_encoder_stacks = n_encoder_stacks
        self.n_decoder_stacks = n_decoder_stacks
        self.input_id_encoder = input_id_encoder

        self.encoder = Encoder(d_model=d_model, d_inner_layer=d_inner_layer, n_heads=n_heads, n_stacks=n_encoder_stacks)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
        n_stacks: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_later = d_inner_layer

        self.n_heads = n_heads
        self.n_stacks = n_stacks

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, d_inner_layer=d_inner_layer, n_heads=n_heads) for i in range(n_stacks)]
        )
    def forward(self, x:Tensor)->Tensor:
        """Apply the Encoder laters to the encoded input ids.

        Args:
            x (Tensor): shape (N, t, d_model)

        Returns:
            Tensor: shape (N, t, d_model)
        """

        pass


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner_layer: int,
        n_heads: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner_later = d_inner_layer
        self.n_heads = n_heads

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """

        Args:
            x (Tensor): shape (N, t, d_model)

        Returns:
            Tensor: shape (N, t, d_model)
        """


class SinePositionEncoder(nn.Module):
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
        return x + self.pe[:, : x.size(1), :].requires_grad(False)


class InputIdEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, positional_encoder: SinePositionEncoder):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoder = positional_encoder

    def forward(self, x: Tensor) -> Tensor:
        """Maps the input id space to the embedding space.

        Args:
            x (Tensor): (N, t, vocab_size)

        Returns:
            Tensor: (N, t, d_model)
        """
        embedding = self.token_encoder(x)  # shape (N, t, d_model)
        embedding = self.positional_encoder(embedding)
        return embedding

