import numpy as np
import timeit
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(1)
np.random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

nt = torch.nested.nested_tensor(
    [torch.arange(12).reshape(2, 6), torch.arange(18).reshape(3, 6)],
    dtype=torch.float,
    device=device,
)

padded_out_tensor = torch.nested.to_padded_tensor(nt, padding=0.0)

# access nested property
nt.is_nested
padded_out_tensor.is_nested

# indexing works like usual
nt[0]
# elt 1 all rows last col
nt[1, :, -1]

# When indexing a nestedtensor's 0th dimension, the result is a regular tensor.
nt[0].is_nested


# 2 corresponds to 2 elements of the nested tensor
# -1 is inherited and then infered for each tensor element
# so in effect we are applying nt[i].reshape(-1,2,3) for i=0,1
nt_reshaped = nt.reshape(2, -1, 2, 3)

# cannot swap zero but can apply 1,2 swap to each of the nested tensors
nt_transposed = nt_reshaped.transpose(1, 2)

nt_mm = torch.nested.nested_tensor([torch.randn((2, 3, 4)), torch.randn((2, 3, 5))], device=device)

# broadcasting allows this generally
torch.matmul(torch.randn((2, 2, 3)), torch.randn((2, 3, 4)))


nt3 = torch.matmul(nt_transposed, nt_mm)
print(f"Result of Matmul:\n {nt3}")

nt4 = F.dropout(nt3, 0.1)
print(f"Result of Dropout:\n {nt4}")

nt5 = F.softmax(nt4, -1)
print(f"Result of Softmax:\n {nt5}")

## experiment with padded vs nested

vocabulary = {"goodbye": 1.0, "padding": 2.0, "embrace": 3.0, "nested": 4.0, "tensor": 5.0}
sentences = [["goodbye", "padding"], ["embrace", "nested", "tensor"]]

max_sentence_length = max([len(sentence) for sentence in sentences])

padded_sentences = [
    [vocabulary[word] for word in sentence] + (max_sentence_length - len(sentence)) * [0] for sentence in sentences
]
padded_sentences = torch.tensor(padded_sentences)

nested_sentences = torch.nested.nested_tensor(
    [[vocabulary[word] for word in sentence] for sentence in sentences], layout=torch.jagged
)

print(f"{padded_sentences=}")
print(f"{nested_sentences=}")

padded_sentences_for_softmax = [
    [vocabulary[word] for word in sentence] + (max_sentence_length - len(sentence)) * [float("-inf")]
    for sentence in sentences
]
padded_sentences_for_softmax = torch.tensor(padded_sentences_for_softmax)

print(F.softmax(padded_sentences_for_softmax, -1))
print(F.softmax(nested_sentences, -1))


## Transformer Example
class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout_p (float, optional): Dropout probability. Default: 0.0
    """

    def __init__(self, E_q: int, E_k: int, E_v: int, E_total: int, nheads: int, dropout_p: float = 0.0):
        super().__init__()
        self.nheads = nheads
        self.dropout_p = dropout_p
        self.query_proj = nn.Linear(E_q, E_total)
        self.key_proj = nn.Linear(E_k, E_total)
        self.value_proj = nn.Linear(E_v, E_total)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_t, E_q)
            key (torch.Tensor): key of shape (N, L_s, E_k)
            value (torch.Tensor): value of shape (N, L_s, E_v)

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        # TODO: demonstrate packed projection
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout_p, is_causal=True)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


## Parameters
# following_paper
N = 1000
E_q, E_k, E_v, E_total = 512, 512, 512, 512
E_out = E_q
nheads = 8
# done in pytorch tutorial
dropout_p = 0.0


## fake datagen
def zipf_sentence_lengths(alpha: float, batch_size: int) -> torch.Tensor:
    # generate fake corpus by unigram Zipf distribution
    # from wikitext-2 corpus, we get rank "." = 3, "!" = 386, "?" = 858
    sentence_lengths = np.empty(batch_size, dtype=int)
    for ibatch in range(batch_size):
        sentence_lengths[ibatch] = 1
        word = np.random.zipf(alpha)
        while word != 3 and word != 386 and word != 858:
            sentence_lengths[ibatch] += 1
            word = np.random.zipf(alpha)
    return torch.tensor(sentence_lengths)


def gen_batch(N, E_q, E_k, E_v, device):
    # generate semi-realistic data using Zipf distribution for sentence lengths
    sentence_lengths = zipf_sentence_lengths(alpha=1.2, batch_size=N)

    # Note: the torch.jagged layout is a nested tensor layout that supports a single ragged
    # dimension and works with torch.compile. The batch items each have shape (B, S*, D)
    # where B = batch size, S* = ragged sequence length, and D = embedding dimension.
    query = torch.nested.nested_tensor(
        [torch.randn(s.item(), E_q, device=device) for s in sentence_lengths], layout=torch.jagged
    )

    key = torch.nested.nested_tensor(
        [torch.randn(s.item(), E_k, device=device) for s in sentence_lengths], layout=torch.jagged
    )

    value = torch.nested.nested_tensor(
        [torch.randn(s.item(), E_v, device=device) for s in sentence_lengths], layout=torch.jagged
    )

    return query, key, value, sentence_lengths


query, key, value, sentence_lengths = gen_batch(N, E_q, E_k, E_v, device)


def jagged_to_padded(jagged_tensor: torch.Tensor, padding_val):
    # TODO: do jagged -> padded directly when this is supported
    return torch.nested.to_padded_tensor(torch.nested.nested_tensor(list(jagged_tensor.unbind())), padding_val)


padded_query, padded_key, padded_value = (jagged_to_padded(t, 0.0) for t in (query, key, value))
mha = MultiHeadAttention(E_q, E_k, E_v, E_total, nheads, dropout_p).to(device=device)


def benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    begin = timeit.default_timer()
    output = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return output, (end - begin)


output_nested, time_nested = benchmark(mha, query, key, value)
output_padded, time_padded = benchmark(mha, padded_query, padded_key, padded_value)

# padding-specific step: remove output projection bias from padded entries for fair comparison
for i, entry_length in enumerate(sentence_lengths):
    output_padded[i, entry_length:] = 0.0

print("=== without torch.compile ===")
print(
    "nested and padded calculations differ by",
    (jagged_to_padded(output_nested, 0.0) - output_padded).abs().max().item(),
)
print("nested tensor multi-head attention takes", time_nested, "seconds")
print("padded tensor multi-head attention takes", time_padded, "seconds")

# warm up compile first...
compiled_mha = torch.compile(mha)
compiled_mha(query, key, value)
# ...now benchmark
compiled_output_nested, compiled_time_nested = benchmark(compiled_mha, query, key, value)

# warm up compile first...
compiled_mha(padded_query, padded_key, padded_value)
# ...now benchmark
compiled_output_padded, compiled_time_padded = benchmark(compiled_mha, padded_query, padded_key, padded_value)

# padding-specific step: remove output projection bias from padded entries for fair comparison
for i, entry_length in enumerate(sentence_lengths):
    compiled_output_padded[i, entry_length:] = 0.0

print("=== with torch.compile ===")
print(
    "nested and padded calculations differ by",
    (jagged_to_padded(compiled_output_nested, 0.0) - compiled_output_padded).abs().max().item(),
)
print("nested tensor multi-head attention takes", compiled_time_nested, "seconds")
print("padded tensor multi-head attention takes", compiled_time_padded, "seconds")

print(f"Nested speedup: {compiled_time_padded / compiled_time_nested:.3f}")

# nested tensor test

variable_seq = torch.nested.nested_tensor(
    [torch.arange(7), torch.arange(13)],
    dtype=torch.int,
    device=device)

embedder = nn.Embedding(num_embeddings=13, embedding_dim=512, device=device)

variable_encoding = embedder.forward(variable_seq)
variable_encoding[0].size()

# variable_encoding + variable_encoding[0]