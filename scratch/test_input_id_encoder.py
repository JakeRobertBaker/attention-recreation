from torch import LongTensor, Tensor

from attention_recreation.model import InputIdEncoder, SinePositionEncoder
from tokenizers import Tokenizer

en_tokenizer: Tokenizer = Tokenizer.from_file("tokenizers/basic_en_tokenizer.json")
en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
en_vocab_size = en_tokenizer.get_vocab_size()

# TODO check bert base cased vocab size for comparison

d_model = 512
p_dropout = 0

sin_pos_enc = SinePositionEncoder(d_model=d_model, p_dropout=p_dropout)
input_id_encoder = InputIdEncoder(vocab_size=en_vocab_size, d_model=d_model, positional_encoder=sin_pos_enc)
outputs = en_tokenizer.encode_batch(["Hello my name is Jake", "Donald Trump is VERY smart!!", "I love cheese!"])
token_ids = LongTensor([out.ids for out in outputs])
encoded_ids: Tensor = input_id_encoder(token_ids)

print("Token IDs")
print(token_ids)
print("\nEncoded tokens shape")
print(encoded_ids.shape)
