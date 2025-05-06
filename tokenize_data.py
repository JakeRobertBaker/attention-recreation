from pathlib import Path
from typing import Generator, Tuple

from datasets import load_dataset

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, Trainer


def basic_tokenizer_trainer() -> Tuple[Tokenizer, Trainer]:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    # TODO post proceesing will vary on inference role
    # TODO add padding when padding is needed.

    trainer = BpeTrainer(
        min_frequency=2,
        show_progress=True,
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
    )

    return tokenizer, trainer


def train_and_save_tokenizer(
    tokenizer: Tokenizer,
    trainer: Trainer,
    generator: Generator,
    save_name: str | Path,
    save_dir: str | Path = "./tokenizers",
):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    tokenizer.train_from_iterator(generator, trainer)

    save_dir = Path(save_dir)
    save_path = str(save_dir / save_name)

    if not save_dir.is_dir():
        save_dir.mkdir()

    tokenizer.save(save_path)
    print(f"\nTokenizer saved to {save_path}")


if __name__ == "__main__":
    print("Loading Dataset")
    ds = load_dataset("Helsinki-NLP/opus-100", "de-en", streaming=True)
    en_generator = (x["translation"]["en"] for x in ds["train"])
    de_generator = (x["translation"]["de"] for x in ds["train"])

    en_tokenizer, en_trainer = basic_tokenizer_trainer()
    de_tokenizer, de_trainer = basic_tokenizer_trainer()

    print("\nTraining EN")
    train_and_save_tokenizer(en_tokenizer, en_trainer, en_generator, "basic_en_tokenizer.json")
    print("\nTraining DE")
    train_and_save_tokenizer(de_tokenizer, de_trainer, de_generator, "basic_de_tokenizer.json")
