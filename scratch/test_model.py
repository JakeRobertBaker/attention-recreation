import os
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from attention_recreation.model import EncoderDecoder, make_decoder, make_encoder, make_input_id_encoder
from tokenizers import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# load data
ds = load_dataset("Helsinki-NLP/opus-100", "de-en", streaming=False)
en_generator = (x["translation"]["en"] for x in ds["train"])
de_generator = (x["translation"]["de"] for x in ds["train"])

# get tokenizers

print(os.getcwd())
en_tokenizer: Tokenizer = Tokenizer.from_file("./tokenizers/basic_en_tokenizer.json")
de_tokenizer: Tokenizer = Tokenizer.from_file("./tokenizers/basic_de_tokenizer.json")


# Preprocess and tokenize
def tokenize(example):
    en_encoded = en_tokenizer.encode(example["translation"]["en"])
    de_encoded = de_tokenizer.encode(example["translation"]["de"])
    return {"en_input_ids": en_encoded.ids, "de_input_ids": de_encoded.ids, "en_length": len(en_encoded.ids)}


tokenized_ds = ds.map(tokenize, remove_columns="translation", num_proc=os.cpu_count())
# filter out very long and short sentences they appear to be bad!
tokenized_ds = tokenized_ds.filter(lambda x: (x["en_length"] >= 4) & (x["en_length"] <= 45), num_proc=os.cpu_count())

en_pad_id = en_tokenizer.token_to_id("[PAD]")
de_pad_id = de_tokenizer.token_to_id("[PAD]")

train_en_lengths = tokenized_ds["train"]["en_length"]


class BatchSampler:
    def __init__(self, lengths, batch_size):
        """Shuffle idea taken from https://pi-tau.github.io/posts/transformer/#token-embedding-layer"""
        self.lengths = lengths
        self.batch_size = batch_size

    def __iter__(self):
        indicies = list(range(len(self.lengths)))
        shuffle(indicies)

        # yeild a pool of size 200 batch size
        step = 200 * self.batch_size
        for pool_start in range(0, len(self.lengths), step):
            pool = indicies[pool_start : pool_start + step]
            pool_sorted_by_lengths = sorted(pool, key=lambda i: self.lengths[i])

            for i in range(0, step, self.batch_size):
                yield pool_sorted_by_lengths[i : i + self.batch_size]

    def __len__(self):
        return len(self.lengths) // self.batch_size


def collate_fn(batch):
    seq_en_token_ids = [Tensor(x["en_input_ids"]) for x in batch]
    en_tokens_padded = pad_sequence(seq_en_token_ids, batch_first=True, padding_value=en_pad_id)

    seq_de_token_ids = [Tensor(x["de_input_ids"]) for x in batch]
    de_tokens_padded = pad_sequence(seq_de_token_ids, batch_first=True, padding_value=de_pad_id)

    return en_tokens_padded, de_tokens_padded


dataloader = DataLoader(
    dataset=tokenized_ds["train"].remove_columns("en_length").with_format("torch", device=device),
    batch_sampler=BatchSampler(train_en_lengths, BATCH_SIZE),
    collate_fn=collate_fn,
)

for en_tokens_padded, de_tokens_padded in dataloader:
    break

source_key_padding_mask = en_tokens_padded == en_pad_id


d_model = 512
d_inner_layer = 2048
n_heads = 8
n_encoder_stacks = 7
n_decoder_stacks = 7
source_vocab_size = en_tokenizer.get_vocab_size()
target_vocab_size = de_tokenizer.get_vocab_size()


encoder_decoder_model = EncoderDecoder(
    d_model=d_model,
    d_inner_layer=d_inner_layer,
    n_heads=n_heads,
    n_encoder_stacks=n_encoder_stacks,
    n_decoder_stacks=n_decoder_stacks,
    source_vocab_size=source_vocab_size,
    target_vocab_size=target_vocab_size,
    input_id_encoder_factory=make_input_id_encoder,
    encoder_factory=make_encoder,
    decoder_factory=make_decoder,
).to(device)

encoder_decoder_model.forward(
    target_ids=de_tokens_padded, source_ids=en_tokens_padded, source_key_padding_mask=source_key_padding_mask
)