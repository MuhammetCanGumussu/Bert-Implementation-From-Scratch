"""
tokenizer alfabeyi azalt
tokenizer cased hale getir
tokenizer (daha kök ve ekler şeklinde olması için farklı tokenizer'lar vs kullanılabilir bakılacak) [pretokenize aşamasında yapılabilir ancak henüz bunu yapabilecek bir sistem bulamadım]

exluded notNext sample sayısını track edebiliriz
stat'da oranlar da olabilir (update yerine en son kaydedilirken)
weighted loss net gerekiyor gözüküyor
tokenizer, random word set vs ayrıca test edilecek

"""

# standard library
import os
import random
import multiprocessing as mp
from typing import List, Tuple


# third party library
import spacy
import tqdm
import numpy as np
import pandas as pd


# local
from .data_aux import FillInput, OneSampleStat, ModelInput, Stat, get_merged_files, get_last_shard_idx, get_tokenizer
from ..random_word_set.random_word_set import get_random_word_set
from ..tokenizer.train_tokenizer import get_tokenizer
from ..config import get_data_py_config


# aslında en ilk dosyalar arası import olayına bakmalı
# flags: DİR, TOKENİZER, HATA, MAGIC NUMBERS, tokenizer'da "..." tokeni eklemeyi kaldır! 31999 yerine 32000 alıyoruz emb hatası oluyor!
# train val split
# configurable
# ignore idx [X]
# tokenizer
# random word set
# dir name'e tokenizer (dosyaların en sağında tokenizer postfix koyma sıkıntı yaratmaz gibi gözüküyor, hiç "_" ile split yapılmamış ne data da ne dataaux'da) (dosyalar yerine dir'de olsa da yeter gibi)
# system path işleri 
# extra: farklı veri setleri içinde çalışabilme (shard'lı şekilde)
# extra2: uniform dist for isnext, configurable

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = get_data_py_config()

# DEFAULT VALUES (değiştirilebilir/config/param)
BLOCK_SIZE = cfg.block_size
NUM_OF_DOCS_PER_SHARD = cfg.num_of_docs_per_shard       # doc_shards ve ab_shards için
NUM_TOKENS_PER_SHARD = cfg.num_tokens_per_shard   # xy_shards için, örn: 40_000 sample * 256 token ~ 10_000_000 tokens
OVERLAP = cfg.overlap          # pencere hareketi/deltası kesinlikle 4 dene!
EDGE_BUFFER = cfg.edge_buffer                        # block penceresi sınırlarını tamponluyoruz gibi düşünülebilir (tampon içinde kalan sent idx'ler kullanılmayacak)
SEED = cfg.seed
RATE_OF_UNTOUCHED_WORDS = cfg.rate_of_untouched_words
MASK_RATIO = cfg.mask_ratio
REPLACE_RATIO = cfg.replace_ratio
IDENTITY_RATIO = cfg.identity_ratio
TOKENIZER_TYPE = cfg.tokenizer_type



# TODO TODO TODO BAKILACAK:
# prepare data for do_eval parametresi yapabiliriz, tek bir xy shardı hazırlar (block size'ı vs gene yukarıdaki global ile belirlenecek no problemo)
# daha iyi moduler çözüm get_tokenizer'ı güncelleyip hf'de return edebiliriz
# daha sonra shard count limit tanımlayabiliriz gibi gibi
# tabi xy klasörlerinde ek tokenizer postifx kullanabiliriz (buna gerek kalmayacak şekilde yapalım ya, do eval klasörü oluşsun yeter gibi)

NUM_PROCESSES = (os.cpu_count() - 1) if os.cpu_count() > 1 else 1

tokenizer = get_tokenizer(custom=TOKENIZER_TYPE)
word_dict = get_random_word_set()

CLS_TOKEN_ID = tokenizer.convert_tokens_to_ids("[CLS]")
SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids("[SEP]")
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids("[PAD]")
MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[MASK]")
UNK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[UNK]")

# multi language sentence tokenizer
sent_seperator = spacy.load("xx_sent_ud_sm")


def appy_seed():
    random.seed(SEED) # reproducibility


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

def get_docs_df() -> pd.DataFrame:
    
    print(f"[INFO] Docs dataframe is being created...")

    merged_content = get_merged_files()
    _, docs = split_titles_and_docs(merged_content)

    del merged_content

    # inplace operation
    random.shuffle(docs)

    docs = delete_subtitles_from_docs(docs)

    return pd.DataFrame(docs, columns=["doc"])





#def clean_shards_dir(shards_dir:str, num_shards:int) -> None:
#    files = os.listdir(shards_dir)
#    
#    for file in files:
#
#        # eğer xy_shards dizininde isek, extra stat.txt dosyası olacak, bu dosyanın silinmesini istemiyoruz
#        if file == "stat.txt":
#            continue
#
#        # dizin ile aynı prefix'e sahip olmayan dosyaları sil 
#        if not file.startswith(shards_dir.split("_")[0]):
#            print(f"[INFO] {shards_dir}/{file} removed, not expected file in this dir...")
#            os.remove(shards_dir + "/" + file)
#            continue
#
#        # dosyanın shard index'ini al
#        last_shard_idx = int(file.split("_")[2].split(".")[0])
#
#        # beklenen aralık dışındaki indexli dosyaları sil
#        if last_shard_idx >= num_shards or last_shard_idx < 0:
#            os.remove(shards_dir + "/" + file)
#            print(f"[INFO] {shards_dir}/{file} removed, not expected index range: {last_shard_idx} >= {num_shards} or {last_shard_idx} < 0...")
#            continue





def create_doc_shards(docs_df):


    num_shards = len(docs_df) // NUM_OF_DOCS_PER_SHARD
    extra_shard_len = 0

    if len(docs_df) % num_shards != 0:
        extra_shard_len = len(docs_df) % NUM_OF_DOCS_PER_SHARD
        num_shards += 1

    # DİR
    doc_shards_dir = root_dir + "/doc_shards"
    os.makedirs(doc_shards_dir, exist_ok=True)

    #clean_shards_dir(doc_shards_dir, num_shards)
    last_shard_idx = get_last_shard_idx(doc_shards_dir) + 1

    # beklenen tüm dosyalar hali hazırda mevcut ise, çık
    if last_shard_idx == num_shards:
        print(f"[INFO] Expected number of doc shard files are already exists. Exiting...")
        return
    
    print(f"[INFO] Creating doc shard files, starts from 0...")

    if extra_shard_len:
        print(f"[INFO] Extra shard needed. Number of doc will be placed in extra shard: {extra_shard_len}")

    
    df_start_idx = 0
    df_end_idx = NUM_OF_DOCS_PER_SHARD



    with tqdm.tqdm(total=num_shards, desc="Doc shards are being created") as pbar:
        for last_shard_idx in range(0, num_shards):

            # last iteration (extra shard)
            if last_shard_idx == num_shards - 1:
                df_end_idx = df_start_idx + extra_shard_len

                assert df_end_idx == len(docs_df), f"df_end_idx: {df_end_idx}, len(docs_df): {len(docs_df)}"
                assert (df_end_idx - df_start_idx) == extra_shard_len
                #DIR
                docs_df.iloc[df_start_idx:df_end_idx].to_json(doc_shards_dir + f"/doc_shard_{last_shard_idx}.json",
                                                          orient="records",
                                                          lines=True,
                                                          force_ascii=False)
                pbar.update()
                break
            #DIR
            docs_df.iloc[df_start_idx:df_end_idx].to_json(doc_shards_dir + f"/doc_shard_{last_shard_idx}.json",
                                                          orient="records",
                                                          lines=True,
                                                          force_ascii=False)
            df_start_idx = df_end_idx
            df_end_idx += NUM_OF_DOCS_PER_SHARD
            pbar.update()



def read_shard(shard_dir: str, shard_idx: int, return_type: str) -> pd.DataFrame | dict[str, list] | np.ndarray:
    # TODO: read_xy, read_doc, read_ab diye fonk'lar oluştur, burada onları çağır.

    last_idx = get_last_shard_idx(shard_dir)

    if last_idx == -1:
        raise FileNotFoundError(f"{shard_dir} is empty...")

    if shard_idx > last_idx or shard_idx < 0:
        raise IndexError(f"shard idx must be >= 0 and <= {last_idx}, shard_idx you gave was: {shard_idx}")
    
    if shard_dir.split("/")[-1].startswith("xy"):
        if return_type != "np":
            raise ValueError(f"'return_type' parameter must be 'np' for xy_shards dir...")
        #DİR
        return np.load(shard_dir + f"/xy_shard_{shard_idx}.npy")
    

    prefix = "doc" if shard_dir.split("/")[-1].startswith("doc") else "ab"
    temp_df = pd.read_json(shard_dir + f"/{prefix}_shard_{shard_idx}.json",
                       orient="records",
                       lines=True,
                       encoding="utf-8")
    
    if return_type == "pd":
        return temp_df
    if return_type == "dict":
        return temp_df.to_dict("list")

    raise ValueError(f"'return_type' parameter must be 'pd' or 'dict' for doc or ab shards...")
    


def tokenize_and_sent_idx(doc: pd.Series)-> dict[str, list]:

    if type(doc) == tuple:
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


    # shift sent_idx left by 1
    temp["sent_idx"] = [ idx - 1 for idx in temp["sent_idx"]]

    return temp

def _get_random_sample(docs: dict[str, list], len_of_random_b):
    
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
    # word_id'lerin her ab sample için 0'dan başlaması gerek. (bunu burada yapmayı unuttuğumdan xy oluşturma esnasına kaldı...)
    ab_dict["isNext"].append(True)

def _fill_a_from_block_b_from_random(ab_dict, doc, docs, block_start_idx, mid_sent_idx, len_of_random_b):
    random_b = _get_random_sample(docs, len_of_random_b)
    if random_b is not None:
        ab_dict["A_token_ids"].append(doc["token_ids"][block_start_idx:mid_sent_idx])
        ab_dict["A_word_ids"].append(doc["word_ids"][block_start_idx:mid_sent_idx])
        ab_dict["B_token_ids"].append(random_b["token_ids"][:])
        ab_dict["B_word_ids"].append(_update_random_word_id(random_b["word_ids"][:], ab_dict["A_word_ids"][-1][-1])) # updt_rnd_b(list, int)
        # word_id'lerin her ab sample için 0'dan başlaması gerek. (bunu burada yapmayı unuttuğumdan xy oluşturma esnasına kaldı...)
        ab_dict["isNext"].append(False)

    # eğer random sample "bir şekilde" bulunamadıysa (çok zor bir ihtimal) hiç birşey yapma

def convert_doc_to_ab(args: Tuple)-> dict[str, list]:
    """ 
        args[0]:   docs (DictProxy or dict[str, list])
        args[1]:   doc_idx (int)
    """
    block_size = BLOCK_SIZE
    overlap = OVERLAP
    edge_buffer = EDGE_BUFFER
    
    docs = args[0]
    doc_idx = args[1]
    #HATA mp'de her bir proc'a docs ref bir şekilde yollayabileceğimi düşünmüştüm (bu diğerlerinde bu hatayı yapmadım diye hatırlıyorum),
    # bundan dolayı burada slice'lama/indexleme yapmışım. mp.pool'da tuple olarak idx ve docs döndermişim. Bunun yerine gene iterrows kullanmak
    # daha iyi olacaktır
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
        block_sent_idx = [sent_idx for sent_idx in doc["sent_idx"] if (block_start_idx + edge_buffer) < sent_idx and (block_end_idx - edge_buffer) > sent_idx]
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
    
    # Genellikle doc sonlarında block dışında kalan alanlar olur. Burayı random B ile kullanabiliriz. Bu durumda çoğu durumda toplam doc sayısı kadar 
    # fazladan notNext sample'ımız olur (class imbalance denebilecek kadar orantısal etki olmaz (veri seti büyük olduğundan ve tabi doclar'da geniş olduğundan)
    # ama mesela doc sayısı çok olsa, doclar kısa olsa (bir doc'tan çok fazla normal sample çıkmaz ise), veri seti küçük olursa o zaman problem olabilir
    # ben gene de class (isNext or not) weight hesaplayan vs fonklar tanımlayacağım, bariz bir oran farkı varsa o zaman uygulamak önemli olur 
    if (block_start_idx + edge_buffer) < doc_len:
        len_of_random_b = block_end_idx - doc_len  # A midpoint yani nokta/sent_idx tokenınıda içine alacağından bir sağ shiftledik
        # random b için istenen token sayısı edge_bufferdan az ise o bu ab sample adayı kullanılmayacak (B'de çok çok az sayıda token var demektir)
        if len_of_random_b < edge_buffer:
           return ab_dict
        _fill_a_from_block_b_from_random(ab_dict, doc, docs, block_start_idx, doc_len, len_of_random_b)

    return ab_dict




def create_ab_shards() -> None:

    #DİR
    ab_dir = root_dir + f"/ab_shards_{TOKENIZER_TYPE}_{BLOCK_SIZE}"
    os.makedirs(ab_dir, exist_ok=True)

    #DİR
    doc_dir = root_dir + "/doc_shards"
    num_shards = len(os.listdir(doc_dir))
    # clean_shards_dir(ab_dir, num_shards)

    # son var olan dosya idx'ini aldık bir sonrakinden devam edileceği için 1 ekledik
    last_shard_idx = get_last_shard_idx(ab_dir) + 1
    
    # beklenen tüm dosyalar hali hazırda mevcut ise, çık
    if last_shard_idx == num_shards:
        print(f"[INFO] Expected number of ab shard {BLOCK_SIZE} files are already exists in {ab_dir}. Exiting...")
        return
    
    print(f"[INFO] Creating ab shard files, continues (or starts) from {last_shard_idx}...")

    for shard_idx in range(last_shard_idx, num_shards):
        #DİR
        docs_shard_df = read_shard(doc_dir, shard_idx, "pd")
    
        docs_shard_df["token_ids"] = None
        docs_shard_df["word_ids"] = None
        docs_shard_df["sent_idx"] = None

        
        with mp.Pool(NUM_PROCESSES) as pool:
            # list[series]
            docs_tokenized_pool = pool.imap(tokenize_and_sent_idx, docs_shard_df.iterrows(), chunksize= 200)
            list_of_docs_tokenized = list(tqdm.tqdm(docs_tokenized_pool, total=len(docs_shard_df), desc=f"[INFO] Tokenization and sent_idx of docs, shard_{shard_idx} / {num_shards - 1}"))

        del docs_shard_df
        del docs_tokenized_pool

        # manager = mp.Manager()
        # shared_docs_tokenized_dict = manager.dict(pd.DataFrame(list_of_docs_tokenized).to_dict("list"))
        shared_docs_tokenized_dict = pd.DataFrame(list_of_docs_tokenized).to_dict("list")
        len_of_docs = len(list_of_docs_tokenized)
        del list_of_docs_tokenized


        with mp.Pool(NUM_PROCESSES) as pool:
            # ab_pool = pool.imap(convert_doc_to_ab, [(shared_docs_tokenized_dict, idx) for idx in range(0, len_of_docs)], chunksize= 200)
            ab_pool = pool.imap(convert_doc_to_ab, [(shared_docs_tokenized_dict, idx) for idx in range(0, len_of_docs)], chunksize= 200)
            list_of_ab_dicts = list(tqdm.tqdm(ab_pool, total=len_of_docs, desc=f"[INFO] Converting docs to ab, shard_{shard_idx}"))
            ab_df = pd.concat([pd.DataFrame(each_ab_dict) for each_ab_dict in list_of_ab_dicts])

        del ab_pool
        del list_of_ab_dicts

        # shuffle each ab sample (doc/title related ab samples will not be consecutive, so every sample will talk about diffrent topic)
        ab_df = ab_df.sample(frac=1)

        #DİR
        # save
        ab_df.to_json(ab_dir + f"/ab_shard_{shard_idx}.json",
                       orient="records",
                       lines=True,
                       force_ascii=False)



def get_random_tokens(number_of_token_need: int)-> np.ndarray | None:
    temp = f"token_group_{number_of_token_need}"
    if temp not in word_dict.keys():
        # edge case
        # print(f"[INFO] Token group not found for this value: {number_of_token_need}")
        return None
    return np.array(random.choice(word_dict[temp]), dtype=np.uint16)

def _fill_mask(fill_input: FillInput, word_ids: np.ndarray, x: np.ndarray, y: np.ndarray, one_sample_stat: OneSampleStat=None) -> None:

    mask_word_array = fill_input.mask_word_array

    for mask_word_id in mask_word_array:
        # binary search
        word_token_slice_start_idx = np.searchsorted(word_ids, mask_word_id, side="left")
        word_token_slice_end_idx = word_token_slice_start_idx + 1

        for idx in range(word_token_slice_start_idx + 1, len(word_ids)):
            if word_ids[idx] != mask_word_id:
                break
            word_token_slice_end_idx += 1

        if one_sample_stat is not None:
            one_sample_stat.number_of_mask_token_count += word_token_slice_end_idx - word_token_slice_start_idx
        
        word_token_slice = x[word_token_slice_start_idx:word_token_slice_end_idx].copy()
        x[word_token_slice_start_idx:word_token_slice_end_idx] = MASK_TOKEN_ID
        y[word_token_slice_start_idx:word_token_slice_end_idx] = word_token_slice
    


def _fill_replace(fill_input: FillInput, word_ids: np.ndarray, x: np.ndarray, y: np.ndarray, one_sample_stat: OneSampleStat=None) -> None:
    
    replace_word_array = fill_input.replace_word_array

    for replace_word_id in replace_word_array:
        # binary search
        word_token_slice_start_idx = np.searchsorted(word_ids, replace_word_id, side="left")
        word_token_slice_end_idx = word_token_slice_start_idx + 1

        for idx in range(word_token_slice_start_idx + 1, len(word_ids)):
            if word_ids[idx] != replace_word_id:
                break
            word_token_slice_end_idx += 1

        number_of_token_need = word_token_slice_end_idx - word_token_slice_start_idx
        random_token_slice = get_random_tokens(number_of_token_need)

        # ihtiyaç duyulan random token sayısı eldekinden fazla ise None döndürecek, 
        # bu durumda bir sonraki replace kelimesine atla
        if random_token_slice is None:
            if one_sample_stat is not None:
                one_sample_stat.number_of_not_accepted_word += 1
            continue

        if one_sample_stat is not None:
            one_sample_stat.number_of_replace_token_count += number_of_token_need


        word_token_slice = x[word_token_slice_start_idx:word_token_slice_end_idx].copy()
        y[word_token_slice_start_idx:word_token_slice_end_idx] = word_token_slice
        x[word_token_slice_start_idx:word_token_slice_end_idx] = random_token_slice


def _fill_identity(fill_input: FillInput, word_ids: np.ndarray, x: np.ndarray, y: np.ndarray, one_sample_stat: OneSampleStat=None) -> None:

    identity_word_array = fill_input.identity_word_array

    for identity_word_id in identity_word_array:
        # binary search
        word_token_slice_start_idx = np.searchsorted(word_ids, identity_word_id, side="left")
        word_token_slice_end_idx = word_token_slice_start_idx + 1

        for idx in range(word_token_slice_start_idx + 1, len(word_ids)):
            if word_ids[idx] != identity_word_id:
                break
            word_token_slice_end_idx += 1

        if one_sample_stat is not None:
            one_sample_stat.number_of_identity_token_count += word_token_slice_end_idx - word_token_slice_start_idx
        
        word_token_slice = x[word_token_slice_start_idx:word_token_slice_end_idx]
        y[word_token_slice_start_idx:word_token_slice_end_idx] = word_token_slice


def _fill_xy(fill_input: FillInput, word_ids: np.ndarray, x: np.ndarray, y: np.ndarray, one_sample_stat: OneSampleStat = None) -> None:
        
        funcs = [_fill_mask, _fill_identity, _fill_replace]

        for func in funcs:
            if one_sample_stat is not None:
                func(fill_input, word_ids, x, y, one_sample_stat)
            else:
                func(fill_input, word_ids, x, y)


def _create_fill_input_for_sample(word_ids, one_sample_stat: OneSampleStat = None) -> FillInput | Tuple[FillInput, OneSampleStat]:
        # kaç kelime var
        word_count = word_ids[-1]
        
    
        # kelimelere olasılık veriliyor
        word_array = np.arange(word_count, dtype=np.uint16)
        prop_of_words = np.random.rand(word_count)

        

        mask_bound = (1 - RATE_OF_UNTOUCHED_WORDS) * MASK_RATIO
        replace_bound = (1 - mask_bound) * REPLACE_RATIO + mask_bound
        # not needed
        #identity_bound = (1 - replace_bound) * IDENTITY_RATIO + replace_bound

        mask_of_mask = (prop_of_words > RATE_OF_UNTOUCHED_WORDS) & (prop_of_words <= mask_bound)          # 0.12'lik dilim
        mask_of_replace =  (prop_of_words > mask_bound) & (prop_of_words <= 0.985)     # 0.015'lik dilim
        mask_of_identity = (prop_of_words > replace_bound)                               # 0.015'lik dilim

        
        mask_word_array = word_array[mask_of_mask]
        replace_word_array = word_array[mask_of_replace]
        identity_word_array = word_array[mask_of_identity]

        # eğer tüm sample da hiç mask, identity veya replace yok ise rastgele bir mask kelimesi seciliyor
        if np.all(prop_of_words <= 0.85):
            mask_word_array = np.array([np.random.randint(word_count)])

        if one_sample_stat is not None:

            one_sample_stat.number_of_word = word_count
            one_sample_stat.number_of_mask_word = len(mask_word_array)
            one_sample_stat.number_of_replace_word = len(replace_word_array)
            one_sample_stat.number_of_identity_word = len(identity_word_array)

            return FillInput(mask_word_array, identity_word_array, replace_word_array), one_sample_stat
        
        return FillInput(mask_word_array, identity_word_array, replace_word_array)


def _create_model_inputs(x: np.ndarray, y: np.ndarray, len_A_tokens: int, isNext: bool, blocks_full: bool = True) -> ModelInput:

        # CLS token append
        x = np.concatenate((np.array([CLS_TOKEN_ID], dtype=np.uint16), x), dtype=np.uint16)
        # yeni düzenleme:::::::::::: BAKILACAK TODO
        y = np.concatenate((np.array([PAD_TOKEN_ID], dtype=np.uint16), y), dtype=np.uint16)
        # SEP token append middle (+ 1 başa cls token eklendiği için len 1 artmıştı)
        x = np.concatenate((x[:len_A_tokens + 1], np.array([SEP_TOKEN_ID], dtype=np.uint16), x[len_A_tokens + 1:]), dtype=np.uint16)
        y = np.concatenate((y[:len_A_tokens + 1], np.array([PAD_TOKEN_ID], dtype=np.uint16), y[len_A_tokens + 1:]), dtype=np.uint16)
        # SEP token append end
        x = np.concatenate((x, np.array([SEP_TOKEN_ID], dtype=np.uint16)), dtype=np.uint16)
        y = np.concatenate((y, np.array([PAD_TOKEN_ID], dtype=np.uint16)), dtype=np.uint16)
    

        # segment ids
        segment_ids = np.zeros(x.shape, dtype=np.uint16)
        segment_ids[len_A_tokens + 2:] = 1

        # geliştirdiğim algoritma sonucunda, tüm blocklarım ne olursa olsun full dolu olacak,
        # bu durumda PAD tokenı içermeyeceğinden attention mask'te full 1'lerden oluşacak.
        attention_mask = np.ones(x.shape, dtype=np.uint16)

        # ancak block'lar full dolmadı ise yani PAD tokenlarına sahip ise o zaman ona göre
        # attention mask ayarlanmalı
        if blocks_full == False:
            for i in range((len(x) - 1), -1, -1):
                if x[i] != PAD_TOKEN_ID:
                    attention_mask[i:] = 0

        # hiçbir kelime fill'lenmemişse rasgele bir tanesini mask ile fill'le
        
        return ModelInput(input_ids=x, labels=y, attention_mask=attention_mask, token_type_ids=segment_ids, next_sentence_label=isNext)


def _create_xy(ab_row: pd.Series | tuple[int, pd.Series], stat_needed: bool = True)-> ModelInput | Tuple[ModelInput, OneSampleStat]:
    "ab_row: A_token_ids, B_token_ids, A_word_ids, B_word_ids, isNext"

    # mp pool kullanım, iterrow tuple döndürür
    if isinstance(ab_row, tuple):
        ab_row = ab_row[1]

    len_A_tokens = len(ab_row["A_token_ids"])
    len_B_tokens = len(ab_row["B_token_ids"])


    if PAD_TOKEN_ID == 0:
        y = np.zeros((len_A_tokens + len_B_tokens), dtype=np.uint16)
    else:
        y = np.ones((len_A_tokens + len_B_tokens), dtype=np.uint16) * PAD_TOKEN_ID
    

    # special tokenlar eklenmedi daha (en son yapılacak)
    a_array = np.array(ab_row["A_token_ids"], dtype=np.uint16)
    b_array = np.array(ab_row["B_token_ids"], dtype=np.uint16)
    x = np.concatenate([a_array, b_array], dtype=np.uint16)
    del a_array
    del b_array


    # special tokenlar eklenmedi daha (en son yapılacak)
    a_word_ids = np.array(ab_row["A_word_ids"], dtype=np.uint16)
    b_word_ids = np.array(ab_row["B_word_ids"], dtype=np.uint16)
    word_ids = np.concatenate([a_word_ids, b_word_ids], dtype=np.uint16)
    del a_word_ids
    del b_word_ids
    
    # 0'dan başlaması gerek. Bunu ab oluşumunda yapmayı unutmuşum. İlerde bunu oraya taşıyabiliriz.
    word_ids -= word_ids[0] 

    one_sample_stat = OneSampleStat(isNext=ab_row["isNext"])    
    fill_input, one_sample_stat = _create_fill_input_for_sample(word_ids, one_sample_stat)

    # fills with mask, random or identity tokens
    _fill_xy(fill_input, word_ids, x, y, one_sample_stat) 
    
    if stat_needed:
        return _create_model_inputs(x, y, len_A_tokens, ab_row["isNext"], len_A_tokens), one_sample_stat

    return _create_model_inputs(x, y, len_A_tokens, ab_row["isNext"], len_A_tokens)



def get_num_lines_from_ab_dir() -> int:
    """read and count all lines of spesified dir"""
    num_lines = 0
    #DİR
    for shard_id in os.listdir(root_dir + f"/ab_shards_{TOKENIZER_TYPE}_{BLOCK_SIZE}"):
        with open(root_dir + f"/ab_shards_{TOKENIZER_TYPE}_{BLOCK_SIZE}/ab_shard_{shard_id}", "r", encoding="utf-8") as f:
            for _ in f:
                num_lines += 1
    return num_lines



def save_xy_shard(placeholder_array, shard_idx) -> None:
    #DİR
    np.save(root_dir + f"/xy_shards_{TOKENIZER_TYPE}_{BLOCK_SIZE}/xy_shard_{shard_idx}.npy", placeholder_array)



def create_xy_shards() -> None:

    #DİR
    xy_dir = root_dir + f"/xy_shards_{TOKENIZER_TYPE}_{BLOCK_SIZE}"
    ab_dir = root_dir + f"/ab_shards_{TOKENIZER_TYPE}_{BLOCK_SIZE}"
    os.makedirs(xy_dir, exist_ok=True)

    # örn: 40_000 sample * 256 token ~ 10_000_000 tokens
    number_of_sample_per_shard = NUM_TOKENS_PER_SHARD // BLOCK_SIZE
    
    # if there is no shard file in ab_shards, then raise error (xy shards can't be created)
    if get_last_shard_idx(ab_dir) == -1:
        raise FileNotFoundError(f"There is no shard file in {ab_dir}. To create xy shards please create ab_shards first.")

    # total_lines_in_ab will be used als
    total_lines_in_ab = get_num_lines_from_ab_dir() 
    number_of_shard = total_lines_in_ab // number_of_sample_per_shard if total_lines_in_ab % number_of_sample_per_shard == 0 else (total_lines_in_ab // number_of_sample_per_shard) + 1

    # clean_shards_dir(xy_dir, number_of_shard)
    last_shard_idx_of_xy = get_last_shard_idx(xy_dir) + 1

    
    # beklenen tüm dosyalar hali hazırda mevcut ise, çık
    if last_shard_idx_of_xy == number_of_shard:
        print(f"[INFO] Expected number of files are already exists in {xy_dir}. Exiting...")
        return
    
    print(f"[INFO] Creating xy shard {TOKENIZER_TYPE}, {BLOCK_SIZE} files, starts from 0...")

    last_shard_idx_of_xy = 0

    # for stat.txt
    stat = Stat(block_size = BLOCK_SIZE)

    with mp.Pool(NUM_PROCESSES) as pool:

        # np array genişliği: x + y + segment_ids + attention_mask + 1 --> (BLOCK_SIZE * 4) + 1 
        # + 1 is for next_sentence_label (isNext:1, notNext:0) 
        width = (BLOCK_SIZE * 4) + 1
        
        placeholder_array = np.empty((number_of_sample_per_shard, width), dtype=np.uint16)
        #placeholder_array = np.empty((1000, width), dtype=np.uint16)
        last_row_index = 0
        
        for ab_shard_idx in range(len(os.listdir(ab_dir))): 
            #DİR
            ab_shard_df = read_shard(ab_dir, ab_shard_idx, return_type="pd")

            xy_map_iterator = pool.imap(_create_xy, ab_shard_df.iterrows(), chunksize= 200)
            # xy_map_iterator = pool.imap(_create_xy, ab_shard_df.iloc[:500].iterrows(), chunksize= 200)


            for model_input, one_sample_stat in tqdm.tqdm(xy_map_iterator, total=len(ab_shard_df), desc=f"[INFO] Converting ab shards {ab_shard_idx} to xy shards {last_shard_idx_of_xy} / {number_of_shard - 1}..."):
                
                stat.update_stat_with_another_stat(one_sample_stat)

                if last_row_index < number_of_sample_per_shard:
                    placeholder_array[last_row_index] = np.concatenate([model_input.input_ids, model_input.labels, model_input.token_type_ids, model_input.attention_mask, model_input.next_sentence_label], dtype=np.uint16)
                    last_row_index += 1
                else:
                    save_xy_shard(placeholder_array, last_shard_idx_of_xy)
                    placeholder_array = np.empty((number_of_sample_per_shard, width), dtype=np.uint16)
                    last_shard_idx_of_xy += 1
                    last_row_index = 0
            
        
        # save last shard (slice last placeholder_array)
        print(f"[INFO] Converting ab shards {ab_shard_idx} block size {BLOCK_SIZE} to xy shards {last_shard_idx_of_xy} / {number_of_shard - 1} block size {BLOCK_SIZE}, Last shard...")
        save_xy_shard(placeholder_array[:last_row_index], last_shard_idx_of_xy)

        #DİR
        # save stat
        stat.save_stat(xy_dir + "/stat.txt")








if __name__ == "__main__":

    print(f"[INFO] Number of processes: {NUM_PROCESSES}")

    appy_seed()

    docs_df = get_docs_df()

    create_doc_shards(docs_df)

    # free resource
    del docs_df

    create_ab_shards()

    create_xy_shards()

    