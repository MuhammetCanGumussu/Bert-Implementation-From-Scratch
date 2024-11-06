"""trains wordpiece tokenizer from scratch on turkish data (trwiki)"""

import os

from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

from ..data.data_aux import get_merged_files
from ..config import get_train_tokenizer_py_config





root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = root_dir + "/tokenizer/tr_wordpiece_tokenizer_cased.json"



def get_tokenizer(custom=True):

    if not custom:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    

    # if custom tokenizer, we need to check if there is a custom tokenizer file
    if not os.path.exists(SAVE_PATH):
        print(f"[INFO] there is no tokenizer file to wrap with fast tokenizer in {SAVE_PATH} Please train tokenizer first...")
        import sys
        sys.exit(0)
    
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file = SAVE_PATH, # You can load from the tokenizer file, alternatively
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        clean_up_tokenization_spaces=True   # default olarak ta True ancak future warning ilerde False olacağını belirtti.
                                            # ilerde problem olmaması için (ve tabiki future warning almamak için) açıkca True yaptık
    )
    return tokenizer
    


def main():
    cfg = get_train_tokenizer_py_config()

    VOCAB_SIZE = cfg.vocab_size   
    LIMIT_ALPHABET = cfg.limit_alphabet
    MIN_FREQUENCY = cfg.min_frequency


    if os.path.exists(SAVE_PATH):
        print(f"[INFO] {SAVE_PATH} already exists. Skipping tokenizer training...")
        exit()

    merged_file_content = get_merged_files()

    merged_file_content = merged_file_content.splitlines()


    tokenizer = Tokenizer(models.WordPiece(vocab={"[PAD]":0, "[UNK]":1}, unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(),
         normalizers.Lowercase()]   # bunu kullanmayacağım, kaldıracağım zaman her şeyi baştan exec etmeliyim...
    )

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

    print("[INFO] tokenizer will be saved {SAVE_PATH}...")

    tokenizer.save(SAVE_PATH, pretty=True)




if __name__ == "__main__":
    main()

