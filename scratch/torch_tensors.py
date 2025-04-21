import torch
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