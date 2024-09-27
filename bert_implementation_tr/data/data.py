# standard library
import os
import sys
import json
import logging
import random
import multiprocessing as mp
from multiprocessing.managers import DictProxy
from typing import List, Tuple


# third party library
import spacy
import tqdm
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, decoders







# DEFAULT VALUES (değiştirilebilir/config/param)
NUM_SHARDS = 100
BLOCK_SIZE = 256
OVERLAP = BLOCK_SIZE // 2 # pencere hareketi/deltası kesinlikle 4 dene!
TAMPON = 10  # block penceresi sınırlarını tamponluyoruz gibi düşünülebilir (tampon içinde kalan sent idx'ler kullanılmayacak)



# multi language sentence tokenizer
sent_seperator = spacy.load("xx_sent_ud_sm")

NUM_PROCESSES = (os.cpu_count() - 1) if os.cpu_count() > 1 else 1

def get_tokenizer(tokenizer_path="../tr_wordpiece_tokenizer_cased.json", fast=True):

    if not os.path.exists(tokenizer_path):
        print(f"[INFO] there is no tokenizer file to wrap with fast tokenizer in {tokenizer_path} Please train tokenizer first...")
        exit(0)
    
    if fast:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file = tokenizer_path, # You can load from the tokenizer file, alternatively
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            clean_up_tokenization_spaces=True   # default olarak ta True ancak future warning ilerde False olacağını belirtti.
                                                # ilerde problem olmaması için (ve tabiki future warning almamak için) açıkca True yaptık
        )
             
    else:
        # default olarak ta True ancak future warning ilerde False olacağını belirtti.
        # ilerde problem olmaması için (ve tabiki future warning almamak için) açıkca True yaptık
        tokenizer = Tokenizer.from_file(tokenizer_path, clean_up_tokenization_spaces=True)

    return tokenizer


tokenizer = get_tokenizer(fast=True)


def appy_seed(number=13013):
    random.seed(42) # reproducibility

def get_merged_files():

    raw_dir = os.path.join(os.path.dirname(__file__), "raw")

    files = os.listdir(raw_dir)

    print(f"[INFO] Files in dir: {files}...")

    merged_file_content = ""

    for raw_file in files:
        with open(os.path.join(raw_dir, raw_file), encoding="utf-8") as raw:
            merged_file_content += (raw.read() + "\n")

    return merged_file_content

def delete_subtitles_from_docs(docs):
    """
    Yaklaşıl 4 kelimeden oluşan cümleler hep alt başlık gibi (wiki dump kaynaklı). Veriye bakarken bu alt başlıkların paragraflar arasında anlam bozukluğuna sebep olabileceğini düşündüm
    Küçük veri seti kullanacak olan modellerde bu sinyallerin performansı kötü etkileyeceğini düşündüğümden (deney yapılabilir)
    bunları silmeye karar kıldım.
    """
    new_docs = []
    for doc in docs:
        new_doc = []
        for line in doc.splitlines():
            if len(line.split(" ")) > 5:
                new_doc.append(line)
        if new_doc: # for example, doc can have 1 sentence and it has less then 5 words
            # let's convert in the same format (join lines again)
            new_docs.append("\n".join(new_doc))
    
    
    assert not any(len(doc) == 0 for doc in new_docs), "[INFO] Unexpected, empty docs list has empty doc list..."

    return new_docs

def split_titles_and_docs(content):
    """Returns a list of titles and a list of documents from merged file"""

    titles = []
    docs = []

    
    lines = content.splitlines()
    doc_lines = []

    for line in lines:

        if line.startswith('== ') and line.endswith(' == '):
            # New title found
            titles.append(line)

            # doc_lines will not has any lines for the firs elements, so this block will be passed (list is empty so cond is False)
            if doc_lines:
                # append the previous doc into docs
                # needs to be joined with \n to make it a 1 string not list of strings
                docs.append("\n".join(doc_lines))
                # Start a new document for this new title
                doc_lines = []

        else:
            # Continue accumulating lines for the current document
            doc_lines.append(line)

    # Append the last document
    docs.append("\n".join(doc_lines))

    assert len(titles) == len(docs), f"[ERROR] len(titles) and len(docs) are not same!... num titles: {len(titles)}, num docs: {len(docs)}"

    return titles, docs

def get_docs_df():
    
    print(f"[INFO] Docs dataframe is being created...")

    merged_content = get_merged_files()
    _, docs = split_titles_and_docs(merged_content)

    del merged_content

    # inplace operation
    random.shuffle(docs)

    docs = delete_subtitles_from_docs(docs)

    return pd.DataFrame(docs, columns=["doc"])

def create_doc_shards(docs_df):

    num_shards = NUM_SHARDS
    num_docs_per_shard = len(docs_df) // num_shards
    extra_shard_len = 0

    if len(docs_df) % num_shards != 0:
        extra_shard_len = len(docs_df) % num_shards
        num_shards += 1
        


    os.makedirs("doc_shards", exist_ok=True)
    files = os.listdir("./doc_shards")


    if len(files) == num_shards:
        print(f"[INFO] Expected number of shards are already exists. Exiting...")
        return
    
    # fazlalık dosyaları (eğer olursa) sil (evet biraz brute force)
    if num_shards < len(files):

        files_finded = set(files)
        shard_files_expected = {f"doc_shard_{shard_idx}.json" for shard_idx in range(num_shards)}
        files_will_deleted = list(files_finded - shard_files_expected)
        if files_will_deleted:
            print(f"[INFO] These files are unexpected, so they are gonna be deleted: {files_will_deleted}")
            for file in files:
                os.remove("./doc_shards/" + file)


    print(f"[INFO] Total number of doc: {len(docs_df)}")
    print(f"[INFO] Each shard will have {num_docs_per_shard} number of docs...")
    print(f"[INFO] Number of shard will be: {num_shards}")

    if extra_shard_len:
        print(f"[INFO] Extra shard needed. Number of doc will be placed: {extra_shard_len}")

    
    df_start_idx = 0
    df_end_idx = num_docs_per_shard


    with tqdm.tqdm(total=num_shards, desc="Doc shards are being created") as pbar:
        for shard_idx in range(0, num_shards):
            # last iteration (extra shard)
            if shard_idx == num_shards - 1:
                df_end_idx = df_start_idx + extra_shard_len

                assert df_end_idx == len(docs_df)
                assert (df_end_idx - df_start_idx) == extra_shard_len

                docs_df.iloc[df_start_idx:df_end_idx].to_json("./doc_shards/" + f"doc_shard_{shard_idx}.json",
                                                          orient="records",
                                                          lines=True,
                                                          force_ascii=False)
                pbar.update()
                break

            docs_df.iloc[df_start_idx:df_end_idx].to_json("./doc_shards/" + f"doc_shard_{shard_idx}.json",
                                                          orient="records",
                                                          lines=True,
                                                          force_ascii=False)
            df_start_idx = df_end_idx
            df_end_idx += num_docs_per_shard
            pbar.update()

def read_shard(shard_dir, shard_idx, return_type="pd"):

    if shard_dir.startswith("doc"):
        prefix = "doc"
    elif shard_dir.startswith("ab"):
        prefix = "ab"

    if shard_idx >= 0 and shard_idx < len(os.listdir(prefix + "_shards")):
        
        temp = pd.read_json(shard_dir + prefix + f"_shard_{shard_idx}.json",
                            orient="records",
                            lines=True,
                            encoding="utf-8")
        
        if return_type == "pd":
            return temp

        elif return_type == "dict":
            return temp.to_dict("list")

        else:
            print(f"[ERROR] Unvalid input for return type, must be 'pd' or 'dict'...")

    else:
        raise IndexError(f"shard_idx >= 0 and shard_idx < {len(os.listdir('doc_shards'))}, shard_idx you gave was: {shard_idx}")

def tokenize_and_sent_idx(doc: pd.Series)-> pd.Series:
    idx = []

    # doc asında iterrow kaynaklı tuple objesi, içinedn doc'u (Series) çıkartalım:
    doc = doc[1]
    temp = {"doc":doc["doc"], "token_ids":[], "word_ids":[], "sent_idx":[]}

    old_idx = 0
    last_word_id = -1
    
    for sentence in list(sent_seperator(temp["doc"]).sents):
        encoding = tokenizer(sentence.text)
        if encoding["input_ids"] == []:
            continue
        old_idx = len(encoding["input_ids"]) + old_idx
        
        temp["sent_idx"].append(old_idx)
        temp["token_ids"].extend(encoding["input_ids"])
        temp["word_ids"].extend((np.array(encoding.word_ids()) + last_word_id + 1).tolist())
        last_word_id = temp["word_ids"][-1]


    # because of zero index system
    temp["sent_idx"] = [ idx - 1 for idx in temp["sent_idx"]]

    return temp

def _get_random_sample(docs: DictProxy | dict[str, list], len_of_random_b):
    
    random_b_dict = {"token_ids":[], "word_ids":[]}

    if len_of_random_b <= 0:
        raise ValueError(f"[ERROR] Len of number cannot be less than or equal to zero! {len_of_random_b} ")

    # rasgele tüm doclara gidilecek (without replacement şekilde)
    for rand_doc_idx in random.sample(range(len(docs["doc"])), k=len(docs["doc"])):
        # en sonki nokta/sent_idx gereksiz: doc sonu random sample penceremin başı olamayacağına göre
        # diğer durumların aksine bu sefer doc başı kullanışlı olabilir: random sample pencerem doc başından başlayabilir
        doc_sent_idx_list = [0] + docs["sent_idx"][rand_doc_idx][:-1]

        # rasgele tüm noktalara/sent_idx'lere gidilecek (without replacement şekilde)
        for rand_sent_start_idx in random.sample(doc_sent_idx_list, k=len(doc_sent_idx_list)):
            # rand_sent_start_idx + 1' deki +1 : nokta/sent_idx tokeni dahil etmemeliyiz B'ye, A sonunda hali hazırda nokta/sent_idx tokeni var
            # yoksa ab cümlelerinde A ve B arasında 2 tane nokta görürüz diye basitçe özetleyebiliriz
            if len(docs["token_ids"][rand_doc_idx][rand_sent_start_idx + 1:]) >= len_of_random_b:
                rand_sent_end_idx = rand_sent_start_idx + len_of_random_b
                random_b_dict["token_ids"] = docs["token_ids"][rand_doc_idx][rand_sent_start_idx + 1:rand_sent_end_idx + 1]
                random_b_dict["word_ids"] = docs["word_ids"][rand_doc_idx][rand_sent_start_idx + 1:rand_sent_end_idx + 1]
                return random_b_dict


    print(f"[INFO] Random sample couldn't be found in this shard interestingly...")
    return None

def _update_random_word_id(random_b_word_ids:List, last_word_id_of_A:int):
    # return updated_random_b_word_ids
    # 32  88
    temp = np.array(random_b_word_ids)
    
    if random_b_word_ids[0] > last_word_id_of_A:
        random_b_word_ids = temp - (random_b_word_ids[0] - last_word_id_of_A - 1)
    else:
        random_b_word_ids = temp + (last_word_id_of_A - random_b_word_ids[0] + 1)
    return random_b_word_ids.tolist()

def _fill_ab_from_block(ab_dict, doc, block_start_idx, mid_sent_idx, block_end_idx):
    ab_dict["A_token_ids"].append(doc["token_ids"][block_start_idx:mid_sent_idx])
    ab_dict["A_word_ids"].append(doc["word_ids"][block_start_idx:mid_sent_idx])
    ab_dict["B_token_ids"].append(doc["token_ids"][mid_sent_idx:block_end_idx])
    ab_dict["B_word_ids"].append(doc["word_ids"][mid_sent_idx:block_end_idx])
    ab_dict["isNext"].append(True)

def _fill_a_from_block_b_from_random(ab_dict, doc, docs, block_start_idx, mid_sent_idx, len_of_random_b):
    random_b = _get_random_sample(docs, len_of_random_b)
    if random_b is not None:
        ab_dict["A_token_ids"].append(doc["token_ids"][block_start_idx:mid_sent_idx])
        ab_dict["A_word_ids"].append(doc["word_ids"][block_start_idx:mid_sent_idx])
        ab_dict["B_token_ids"].append(random_b["token_ids"][:])
        ab_dict["B_word_ids"].append(_update_random_word_id(random_b["word_ids"][:], ab_dict["A_word_ids"][-1][-1])) # updt_rnd_b(list, int)
        ab_dict["isNext"].append(False)

    # eğer random sample "bir şekilde" bulunamadıysa (çok zor bir ihtimal) hiç birşey yapma

def convert_doc_to_ab(args: Tuple, block_size: int = BLOCK_SIZE, 
                      overlap: int = OVERLAP, tampon:int = TAMPON)-> dict[str, list]:
    """ 
        args[0]:   docs (DictProxy or dict[str, list])
        args[1]:   doc_idx (int)
    """
    
    docs = args[0]
    doc_idx = args[1]
    doc = {key: value[doc_idx] for key, value in docs.items()}

    if type(docs) == pd.DataFrame:
        raise TypeError("[ERROR] Docs must be DictProxy or dict[str, list] type...")


    block_size_raw = block_size - 3     # special token counts (cls, sep, sep)
    block_start_idx = 0
    block_end_idx = block_size_raw

    ab_dict = {"A_token_ids":[], "B_token_ids":[], "A_word_ids":[], "B_word_ids":[], "isNext": []}


    doc_len = len(doc["token_ids"])
    while block_end_idx <= (doc_len):

        # block içerisindeki sent indexleri al
        block_sent_idx = [sent_idx for sent_idx in doc["sent_idx"] if (block_start_idx + tampon) < sent_idx and (block_end_idx - tampon) > sent_idx]
        number_of_sent = len(block_sent_idx)


        # block içerisinde sent idx yok
        if number_of_sent == 0:

            # block konumu güncelle
            block_start_idx += overlap
            block_end_idx += overlap

            # bir sonraki iterasyona atla
            continue

         # tek sayı olduğunda
        if number_of_sent % 2 != 0:
            # +1 nedeni: nokta/sent_idx A'nın içinde olsun
            mid_sent_idx = block_sent_idx[number_of_sent // 2] + 1  
        # çift sayı olduğunda
        else:
            # a ve b noktalarından hangisi blocksize ortasına yakın ise o nokta (sent idx) mid_sent_idx olarak alınacak
            a = number_of_sent // 2
            b = a - 1
            # +1 nedeni: nokta/sent_idx A'nın içinde olsun
            # unutma: mid_sent_idx, nokta/sent_idx token'ını indexler! ve bu token A'ya dahil olacak (fill fonk'larda görebilirsin)
            mid_sent_idx = (block_sent_idx[a] + 1) if abs(block_sent_idx[a] - block_size_raw // 2) < abs(block_sent_idx[b] - block_size_raw // 2) else (block_sent_idx[b] + 1)

        if random.random() > 0.5:
            # + 1 şurdan geldi: A nokta/sentidx token'ı da alacağından 1 birim sağdan kaydırdık
            len_of_random_b = block_size_raw - (mid_sent_idx - block_start_idx)

            _fill_a_from_block_b_from_random(ab_dict, doc, docs, block_start_idx, mid_sent_idx, len_of_random_b)

            # block konum güncellemesi yapılmadığına dikkat
            continue 
        

        _fill_ab_from_block(ab_dict, doc, block_start_idx, mid_sent_idx, block_end_idx)

        block_start_idx += overlap
        block_end_idx += overlap
    
    # BURASI AÇIK OLDUĞUNDA FALSE SAYISI ~3000 DAHA FAZLA OLUYOR (DOC SAYISI KADAR). CLASS IMBALANCE'A SEBEP OLABİLİR. GERÇİ FOCAL LOSS KULLANABİLİRİZ...
    # if (block_start_idx + tampon) < doc_len:
    #     len_of_random_b = block_end_idx - doc_len  # A midpoint yani nokta/sent_idx tokenınıda içine alacağından bir sağ shiftledik
    #     # random b için istenen token sayısı tampondan az ise o bu ab sample adayı kullanılmayacak (B'de çok çok az sayıda token var demektir)
    #     if len_of_random_b < tampon:
    #        return ab_dict
    #     _fill_a_from_block_b_from_random(ab_dict, doc, docs, block_start_idx, doc_len, len_of_random_b)

    return ab_dict

def visualize_ab(ab: pd.Series):
    """visualize one ab sample"""

    print(f"A: {ab['A_token_ids']}")
    print(f"B: {ab['B_token_ids']}")
    print(f"A_decoded: {tokenizer.decode(ab['A_token_ids'])}")
    print(f"B_decoded: {tokenizer.decode(ab['B_token_ids'])}")
    print(f"len_of_A: {len(ab['A_token_ids'])}")
    print(f"len_of_B: {len(ab['B_token_ids'])}")
    print(f"A_word_ids: {ab['A_word_ids']}")
    print(f"B_word_ids: {ab['B_word_ids']}")
    print(f"len_of_A_word_ids: {len(ab['A_word_ids'])}")
    print(f"len_of_B_word_ids: {len(ab['B_word_ids'])}")
    print(f"sum_of_AB_tokens: {len(ab['B_word_ids']) + len(ab['A_word_ids'])}")
    print(f"isNext: {ab['isNext']}")
    
    print("---------------\n")


    pass

def create_ab_shards():

    os.makedirs(f"ab_shards_{BLOCK_SIZE}", exist_ok=True)

    # if shard counts are equal, then exit, if not, create new shards (may do overwriting)
    if len(os.listdir(f"ab_shards_{BLOCK_SIZE}")) == len(os.listdir("doc_shards")):
        print("[INFO] ab_shards already exists. Exiting...")
        sys.exit(0)


    for shard_idx in range(len(os.listdir(f"doc_shards_{BLOCK_SIZE}"))):
        docs_shard_df = read_shard(f"doc_shards_{BLOCK_SIZE}/", shard_idx)
    
        docs_shard_df["token_ids"] = None
        docs_shard_df["word_ids"] = None
        docs_shard_df["sent_idx"] = None

        
        with mp.Pool(NUM_PROCESSES) as pool:
            # list[series]
            docs_tokenized_pool = pool.imap(tokenize_and_sent_idx, docs_shard_df.iterrows(), chunksize= 100)
            list_of_docs_tokenized = list(tqdm.tqdm(docs_tokenized_pool, total=len(docs_shard_df), desc=f"[INFO] Tokenization and sent_idx of docs, shard_{shard_idx}"))

        del docs_shard_df
        del docs_tokenized_pool

        # manager = mp.Manager()
        # shared_docs_tokenized_dict = manager.dict(pd.DataFrame(list_of_docs_tokenized).to_dict("list"))
        shared_docs_tokenized_dict = pd.DataFrame(list_of_docs_tokenized).to_dict("list")
        len_of_docs = len(list_of_docs_tokenized)
        del list_of_docs_tokenized


        with mp.Pool(NUM_PROCESSES) as pool:
            # ab_pool = pool.imap(convert_doc_to_ab, [(shared_docs_tokenized_dict, idx) for idx in range(0, len_of_docs)], chunksize= 100)
            ab_pool = pool.imap(convert_doc_to_ab, [(shared_docs_tokenized_dict, idx) for idx in range(0, len_of_docs)], chunksize= 100)
            list_of_ab_dicts = list(tqdm.tqdm(ab_pool, total=len_of_docs, desc=f"[INFO] Converting docs to ab, shard_{shard_idx}"))
            ab_df = pd.concat([pd.DataFrame(each_ab_dict) for each_ab_dict in list_of_ab_dicts])

        del ab_pool
        del list_of_ab_dicts

        # shuffle
        ab_df = ab_df.sample(frac=1)

        # save
        ab_df.to_json(f"./ab_shards_{BLOCK_SIZE}/" + f"ab_shard_{shard_idx}.json",
                       orient="records",
                       lines=True,
                       force_ascii=False)


        
def merge_shards():
    pass

if __name__ == "__main__":

    print(f"[INFO] Number of processes: {NUM_PROCESSES}")

    appy_seed()

    docs_df = get_docs_df()

    create_doc_shards(docs_df)

    # free resource
    del docs_df

    create_ab_shards()