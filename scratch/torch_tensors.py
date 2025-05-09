from sympy import im
import torch
from torch import nn
import math


d_model = 512
max_seq_length = 5000

pe = torch.empty((max_seq_length, d_model))

dim_index = torch.arange(0, d_model, 2)
div_term = torch.exp(-math.log(10_000) * dim_index / d_model) # shape (d_model/2)

positions = torch.arange(0, max_seq_length).reshape(-1, 1)  # shape (max_seq_length, 1)
terms = positions * div_term # shape (max_seq_length, d_model/2)

pe[:,0::2] = torch.sin(terms)
pe[:,1::2] = torch.cos(terms)

# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
# Activate module
layer_norm(embedding)
# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)