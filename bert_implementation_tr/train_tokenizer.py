"""trains wordpiece tokenizer from scratch on turkish data (trwiki)"""

import os

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

from .data.data_aux import get_merged_files



# if os.getcwd() == os.path.dirname(__file__):
#     from data.data import get_merged_files
# else:
#     # GEÇİCİ BUGFİX
#     # cwd data içinde olduğunda, örn data dir içindeki random_word_set.py'i execute edersek cwd data içinde olacak ve yukarıdaki statement hata verecek
#     from data import get_merged_files


VOCAB_SIZE = 32_000   
LIMIT_ALPHABET = 1_000  # TODO: 100 yeterli
MIN_FREQUENCY = 2
SAVE_PATH = "tr_wordpiece_tokenizer_cased.json"






def main():
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

    tokenizer.add_tokens(["..."])

    print("[INFO] tokenizer will be saved...")

    tokenizer.save(SAVE_PATH, pretty=True)


if __name__ == "__main__":
    main()


# # load tokenizer
# tokenizer = Tokenizer.from_file(SAVE_PATH)
# 
# deneme = """Hasekisultan, (bilinen adıyla Haseki) İstanbul, Fatih İlçesi'nde, Millet ve Cerrahpaşa caddeleri arasında Fındıkzade, Cerrahpaşa, Aksaray semtlerinin çevrelediği semt.
# İstanbul'un fethinden sonra oluşturulan Müslüman mahallelerinden biridir. Semtin bugün kullanılan ismi alması 1538'de I. Süleyman'ın (Kanuni) hasekisi Hürrem Sultan tarafından Mimar Sinan'a bir külliye yaptırmasıyla başlar. Zaman içinde yangınlar, imar faailiyetleri ve Haseki Hastanesi'nin genişlemesi sebebiyle orijinal dokusunu kaybetmiştir"""
# 
# import code; code.interact(local=locals())
# 
# print(tokenizer(deneme))
# print(tokenizer.vocab_size)
# 
# sys.exit()