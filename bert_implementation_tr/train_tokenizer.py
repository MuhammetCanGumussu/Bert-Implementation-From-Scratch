"""trains wordpiece tokenizer from scratch on turkish language data (trwiki)"""

# postproc vs ayarlamalar yapabilirim, son veri X,Y oluşumunda baya işten kurtarabilir (hemde hızlı olur)
# tabi yaptıktan sonra çalışıp çalışmadığını dene, encode_batch'te dene!


from data.data import split_titles_and_docs 
import os

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

_, docs_list = split_titles_and_docs("data/merged.raw")


VOCAB_SIZE = 32_000
LIMIT_ALPHABET = 1_000
MIN_FREQUENCY = 2
SAVE_PATH = "tr_wordpiece_tokenizer_cased.json"





if os.path.exists(SAVE_PATH):
    print(f"[INFO] {SAVE_PATH} already exists. Skipping tokenizer training...")
    exit()

tokenizer = Tokenizer(models.WordPiece(vocab={"[PAD]":0, "[UNK]":1}, unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFKC(),
     normalizers.Lowercase()]
)

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), 
                                           pre_tokenizers.Digits(individual_digits=True),
                                           pre_tokenizers.Punctuation()])


special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

trainer = trainers.WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens, 
                                    min_frequency=MIN_FREQUENCY,
                                    continuing_subword_prefix="##", 
                                    limit_alphabet=LIMIT_ALPHABET)

print("[INFO] training is started...")

tokenizer.train_from_iterator(docs_list, trainer=trainer, length=len(docs_list))

tokenizer.decoder = decoders.WordPiece(prefix="##")

print("[INFO] tokenizer will be saved...")


tokenizer.save(SAVE_PATH, pretty=True)


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