from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = Whitespace()
tokenizer.decoder = decoders.BPEDecoder()


print("done")