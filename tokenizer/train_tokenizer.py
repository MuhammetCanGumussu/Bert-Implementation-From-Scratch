"""trains wordpiece tokenizer from scratch on turkish data (trwiki)"""

import os
import sys

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

# add root directory to sys path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if root_dir not in sys.path else None


from data.data_aux import get_merged_files
from config import get_train_tokenizer_py_config


def main():
    cfg = get_train_tokenizer_py_config()

    VOCAB_SIZE = cfg.vocab_size   
    LIMIT_ALPHABET = cfg.limit_alphabet
    MIN_FREQUENCY = cfg.min_frequency
    CASED = cfg.cased

    SAVE_PATH = root_dir + f"/tokenizer/tr_wordpiece_tokenizer_{'cased' if CASED else 'uncased'}.json"


    if os.path.exists(SAVE_PATH):
        print(f"[INFO] {SAVE_PATH} already exists. Skipping tokenizer training...")
        sys.exit(0)

    merged_file_content = get_merged_files()

    merged_file_content = merged_file_content.splitlines()


    tokenizer = Tokenizer(models.WordPiece(vocab={"[PAD]":0, "[UNK]":1}, unk_token="[UNK]"))

    normalizer_list = [normalizers.NFKC()] if CASED else [normalizers.NFKC(), normalizers.Lowercase()]

    tokenizer.normalizer = normalizers.Sequence(normalizer_list)

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), 
                                               pre_tokenizers.Digits(individual_digits=True),
                                               pre_tokenizers.Punctuation()])

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    trainer = trainers.WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens, 
                                        min_frequency=MIN_FREQUENCY,
                                        continuing_subword_prefix="##", 
                                        limit_alphabet=LIMIT_ALPHABET)

    print("[INFO] training is started...")

    tokenizer.train_from_iterator(iterator=merged_file_content, trainer=trainer, length=len(merged_file_content))

    tokenizer.decoder = decoders.WordPiece(prefix="##")

    print(f"[INFO] tokenizer will be saved {SAVE_PATH}...")

    tokenizer.save(SAVE_PATH, pretty=True)




if __name__ == "__main__":
    main()

