"""

ab_raw.txt
ab_dataframe [A, B, isNext]
ab_dataframe [A, B, isNext, A_tokenized, B_tokenized, A_word_ids, B_word_ids, A_token_len, B_token_len]
total_token_num

ab_block1_df [A, B, isNext, A_tokenized, B_tokenized, A_word_ids, B_word_ids, A_token_len, B_token_len]
ab_block2_df [A, B, isNext, A_tokenized, B_tokenized, A_word_ids, B_word_ids, A_token_len, B_token_len]
total_token_num, excluded_token_num


xy_block1_df [x, att_mask, token_type_ids, y] : shards (#token_per_shard must be defined, numpy, dtype important)
xy_block2_df [x, att_mask, token_type_ids, y] : shards

visualization tool


--Çıkarımlar--

Genel bir epoch kavramı olamaz: toplam iterasyon sayısı belirli yüzdeler ile kısa, uzun
block'lara paylaşılacak. block'lardaki sample sayısı eşit olmayabilir. Kısa block içerisinde
bazı uzun block sample'ları kesilmiş/truncated biçimde olabilir, bu da tam olarak tekrar aynı
sample'ı görmek olmasada kesilmiş kısım kadar aynı tokenlar tekrar görülecek demektir uzun block'ta 
(yarı overfitting de diyebiliriz, yarı epoch'ta.). Bu tarz durumlardan dolayı genel epoch
kavramı yerine blocklara özel epoch kavramı kullanılabilir (ki bu da 2.bahsettiğim nedenden dolayı tam
doğru değil). Ne olursa olsun blocklarda toplam kaç token var, ortak kaç sample var vs istatistiği
çıkartılmalıdır. !Önemli!:[overfit vs olabilir demiştim ancak, tam tersi iyi de olabilir sonuçta kısa 
block'tan öğrendiği haldeki ilişkilendirme uzun block'ta dramatik değişikliklere sebep olabilir.]


Daha uzun olan block, kısa bloğun super set'i olamaz/olmamalıdır. Bunun nedeni uzun
bloğun amacının long term örüntüleri yakalamasıdır. Kısa block example'larının
buna bir faydası yok.

Kısa blockta uzun block sample'ları kesilmiş/truncated halde bulunacak. Bu iki block
arasında kesişim olacağı anlamına gelir. id kavramı oluştur ve toplam kaç tane sample'ın
kesişimde olduğunu stat olarak sun

kavram karmaşası önleme: kısa block, uzun block sample'ları vardır. Bazı uzun block sample'ları (
aslında bazı değil baya fazla) kısa block'ta kesilmiş/truncate edilmiş halde bulunacaklar (genede
ben bunlara uzun block sample'ları diyorum bu kafanı karıştırmasın)

Vocabulary dict'i oluşumunu düzeltmeli, kelime tokenizasyonu hali hazırda eğitilmiş tokenizer ile 
yapılmalı (uzun kelimelerde baya hataylı idi.)

pandas df'de apply mp kullanmıyor, dask df'lerde mp özelliği var
kendim dfler üzerinde mp yapmayı denedim, çok verimli ya da güzel değildi ondan dolayı
ownership olayında dask deneyebilirsin. 

(aslında zor değil progress bar ile uğraşmak istemediğim için.
df chunklara(dask'te partition deniyor) bölünüyor sonra ne yapmak istiyorsan o chunklara onu yap mp ile!)

longer block amacı, daha uzun ilişkiler kurabilmeyi öğrenme ve tabiki block_s'ten uzun olan positional embedding'leri de eğitme

"""



# standard library
import os
import sys
import multiprocessing as mp

# third party library
import pandas as pd
from tqdm import tqdm


# my library





BLOCK_SHORT = 128
BLOCK_LONG = 256    # TODO: paper değerlerine göre degistirilecek

NUM_WORKER = os.cpu_count() - 1

tokenizer_path = "../tr_wordpiece_tokenizer_cased.json"

tokenizer_wrapped = get_fast_tokenizer(tokenizer_path)


def get_ab_df():
    """
	    Retrieves or generates a pandas DataFrame containing sentence pairs and their corresponding next flags.
	    The DataFrame is loaded from a JSON file if it exists, otherwise it is created by processing a raw text file.
	    The raw text file is expected to contain sentence pairs separated by '[SEP]' and a next flag.
	    The function returns a pandas DataFrame with columns 'A', 'B', and 'Next' representing the sentence pairs and their next flags.
	"""
    
    

    if os.path.exists("dataframe/ab_df.json"):
        print(f"[INFO] dataframe/ab_df.json exists. Loading...")
        return pd.read_json("dataframe/ab_df.json", orient="records", lines=True)
    

    print(f"[INFO] dataframe/ab_df.json does not exist. Creating...")

    

    # Process each line
    with open("preprocess_and_stats/ab_string.raw", 'r', encoding='utf-8') as file:

        sentences1 = []
        sentences2 = []
        next_flags = []

        for line in file:
            # Strip whitespace and split by [SEP]
            parts = line.strip().split(' [SEP] ')

            assert len(parts) == 3, 'Line has to be 3 parts in format: sentence1 [SEP] sentence2 [SEP] next_flag' + f"\nthis is the problematic line: {line}"
            
            sentence1, sentence2, is_next = parts
            sentences1.append(sentence1)
            sentences2.append(sentence2)
            next_flags.append(is_next)

    # Create a DataFrame
    ab_df = pd.DataFrame({
        'A': sentences1,
        'B': sentences2,
        'Next': next_flags
    })

    # save dataframe (as json lines) before return
    os.makedirs("dataframe", exist_ok=True)
    ab_df.to_json("dataframe/ab_df.json", orient="records", lines=True)

    return ab_df


def dump_stage1_stat(excluded_df, blockS_df, blockL_df):    
    """
    number of tokens in each block
    number of samples in each block
    number of common samples in blockS and blockL

    Inputs:
        - blockS_df [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len] df
        - blockL_df [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len] df
        - excluded_df [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len] df
    """
    pass


def truncate_S(row):
    """
    Some rows need to be truncated to a maximum length defined by the global constant 'BLOCK_SHORT'. 

    Parameters:
        row (pandas.Series): A row of a pandas DataFrame.

    Returns:
        pandas.Series: The modified row with truncated fields.
    """
    # unpack pandas series object from tuple(index, pandas.Series)
    row = row[1]

    # no need to truncate (already compatible with BLOCK_SHORT)
    if row["A_token_len"] + row["B_token_len"] + 3 <= BLOCK_SHORT:
        return row
    
    # truncate
    b_len_after_truncation = BLOCK_SHORT - 3 - row["A_token_len"]

    row["B_tokens"] = row["B_tokens"][:b_len_after_truncation]
    row["B_word_ids"] = row["B_word_ids"][:b_len_after_truncation]
    row["B_token_len"] = b_len_after_truncation


    return row

def truncate_L(row):
    """
    Truncates the 'A_tokens', 'B_tokens', 'A_word_ids', and 'B_word_ids' fields of a given row
    in a pandas DataFrame to a maximum length defined by the global constant 'BLOCK_LONG'.

    Parameters:
        row (pandas.Series): A row of a pandas DataFrame.

    Returns:
        pandas.Series: The modified row with truncated fields.
    """
    # unpack pandas series object from tuple(index, pandas.Series)
    row = row[1]

    # no need to truncate (already compatible with BLOCK_LONG)
    if row["A_token_len"] + row["B_token_len"] + 3 <= BLOCK_LONG:
        return row
    
    # truncate
    b_len_after_truncation = BLOCK_LONG - 3 - row["A_token_len"]

    row["B_tokens"] = row["B_tokens"][:b_len_after_truncation]
    row["B_word_ids"] = row["B_word_ids"][:b_len_after_truncation]
    row["B_token_len"] = b_len_after_truncation

    return row



def find_ownership(row):
    """
        each token seq will have additional 3 special tokens (the reason for "+ 3")

        Inputs:
            - row [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len]) pandas.Series

        Returns:
            - row [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len, group] pandas.Series

        for each sample/row in ab_df:
                - if (A_token_len + 3) >= BLOCK_LONG                      (excluded) 0
                - elif (A_token_len + B_token_len + 3) <= BLOCK_SHORT     (blockS) 1
                - elif (A_token_len + 3) < BLOCK_SHORT                    (both for blockS and blockL) 2
                - else                                                    (blockL) 3

                after slicing subdataframes from ab_df by this conditions we should get subdataframes: 
                    excluded: will not be used, might be used for stat
                    blockS: this subdataframe is just for blockS
                    both: this subdataframe is for both blockS and blockL (truncation operation should be considered)
                    other/blockL: remaining subdataframe is just for blockL

                both df will be concatenated to blockS and blockL so in the end we should get blockS, blockL, excluded
    
    """
    # row ise tuple object right now, but we need pandas.Series so lets unpack
    # Tuple(index:int, pandas.Series) -> pandas.Series
    row = row[1]

    # excluded
    if (row["A_token_len"] + 3) >= BLOCK_LONG:
        row["group"] = 0
        return row
    # blockS
    elif (row["A_token_len"] + row["B_token_len"] + 3) <= BLOCK_SHORT:
        row["group"] = 1
        return row
    # both for blockS and blockL
    elif (row["A_token_len"] + 3) < BLOCK_SHORT:
        row["group"] = 2
        return row
    # blockL
    else:
        row["group"] = 3
        return row



def split_blocks(ab_df):
    """        
    Inputs:
        - ab_df [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len] df
    Returns:
        - blockS, blockL, excluded with additional new "group" column/feature
    """
    # append new group column with default value 0 (excluded group)
    ab_df["group"] = 0 
    total_len = len(ab_df)

    # Create a multiprocessing pool
    with mp.Pool(processes=NUM_WORKER) as pool:
        ab_df_grouped = pool.imap(find_ownership, ab_df.iterrows(), chunksize=NUM_WORKER * 2)
        ab_df_grouped = pd.DataFrame(list(tqdm(ab_df_grouped, total=total_len, desc="Find ownership...")),
                                     columns=["A", "B", "Next", "A_tokens", "B_tokens", "A_word_ids", "B_word_ids", "A_token_len", "B_token_len", "group"])



    # split ab_df_grouped into blockS, both, excluded by "group" column
    excluded = ab_df_grouped[ab_df_grouped["group"] == 0]
    blockS = ab_df_grouped[ab_df_grouped["group"] == 1]
    blockL = ab_df_grouped[ab_df_grouped["group"] == 3]
    both = ab_df_grouped[ab_df_grouped["group"] == 2]


    blockS = pd.concat([blockS, both], axis=0)
    blockL = pd.concat([blockL, both], axis=0)

    with mp.Pool(processes=NUM_WORKER) as pool:
        # truncation operation
        blockS_len = len(blockS)
        blockS = pool.imap(truncate_S, blockS.iterrows(), chunksize=NUM_WORKER * 2)
        blockS = pd.DataFrame(list(tqdm(blockS, total=blockS_len, desc="Truncate blockS...")),
                              columns=["A", "B", "Next", "A_tokens", "B_tokens", "A_word_ids", "B_word_ids", "A_token_len", "B_token_len", "group"])


        blockL_len = len(blockL)
        blockL = pool.imap(truncate_L, blockL.iterrows(), chunksize=NUM_WORKER * 2)
        blockL = pd.DataFrame(list(tqdm(blockL, total=blockL_len, desc="Truncate blockL...")),
                              columns=["A", "B", "Next", "A_tokens", "B_tokens", "A_word_ids", "B_word_ids", "A_token_len", "B_token_len", "group"])


    
    assert (blockS["A_token_len"] + blockS["B_token_len"] <= BLOCK_SHORT).all()   
    assert (blockL["A_token_len"] + blockL["B_token_len"] <= BLOCK_LONG).all()

    print(f"[INFO] Split blocks done!...")


    return blockS, blockL, excluded

def tokenize_row(row):
    """
    Inputs:
        - row [A, B, Next] pandas.Series GÜNCELLE
        - row dict(A, B, Next)
    Returns:
        - row [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len] pandas.Series GÜNCELLE
        - row dict(A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len) 
    """
    tokenizer_wrapped.add_tokens()

    A_encoding = tokenizer_wrapped(row["A"])
    B_encoding = tokenizer_wrapped(row["B"])

    row["A_tokens"] = A_encoding["input_ids"]
    row["B_tokens"] = B_encoding["input_ids"]
    row["A_word_ids"] = A_encoding.word_ids()
    row["B_word_ids"] = B_encoding.word_ids()
    row["A_token_len"] = len(A_encoding["input_ids"])
    row["B_token_len"] = len(B_encoding["input_ids"])

    return row


def tokenize_df(ab_df):
    """
    Inputs:
        - ab_df [A, B, Next] df
    Returns:
        - ab_df [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len] df
    """
    print(f"[INFO] Tokenizing ab_df...")

    ab_records = ab_df.to_dict('records') # list of dicts / dicts are samples/rows
    del ab_df


    # Create a multiprocessing pool
    with mp.Pool(NUM_WORKER) as pool:
        results = pool.imap(tokenize_row, ab_records, chunksize=1024)
        results = list(tqdm(results, total=len(ab_records), desc="Tokenize ab_df rows..."))    

    print(f"[INFO] Tokenizing ab_df done!...")

    return pd.DataFrame(results, columns=["A", "B", "Next", "A_tokens", "B_tokens", "A_word_ids", "B_word_ids", "A_token_len", "B_token_len"])


def apply_stage1(ab_df):
    """
    tokenize df -> add ownership -> split blocks-> dump stat

    Inputs:
        - ab_df [A, B, Next] df

    tokenize_df will return [A, B, Next, A_tokens, B_tokens, A_word_ids, B_word_ids, A_token_len, B_token_len] df

    Returns:
        TODO
        
    """
    # First imap operation with a progress bar

    print(f"[INFO] Applying stage1...")

    ab_df = tokenize_df(ab_df.iloc[:1_000_000]) # lets use 100_000 samples for testing

    # ab_df = tokenize_df(ab_df)

    blockS, blockL, excluded = split_blocks(ab_df)
    del ab_df

    print(f"[INFO] Total excluded rows: {len(excluded)}")

    # geçici stat, stage2 sonrası dosya vs bişiler ayarlarım 
    print(f"[INFO] Number of tokens of blockS: {blockS['A_token_len'].sum()}")
    print(f"[INFO] Number of tokens of blockL: {blockL['A_token_len'].sum()}")
    print(f"[INFO] Number of total tokens: {blockS['A_token_len'].sum() + blockL['A_token_len'].sum()}")

    print(blockS.head(2))
    print(blockL.head(2))

    raise NotImplementedError("Do not go further bruh!")

    # TODO: drop unnecessary columns
    blockS.drop(columns=["A", "B"], inplace=True)
    blockL.drop(columns=["A", "B"], inplace=True)


    # dump_stage1_stat(blockS, blockL, excluded)

    return blockS, blockL
    """
    Truncates the 'A_tokens', 'B_tokens', 'A_word_ids', and 'B_word_ids' fields of a given row
    in a pandas DataFrame to a maximum length defined by the global constant 'BLOCK_LONG'.

    Parameters:
        row (pandas.Series): A row of a pandas DataFrame.

    Returns:
        pandas.Series: The modified row with truncated fields.
    """

def create_xy_shards(random_word_dict):

    ab_df = get_ab_df()
    print(f"[INFO] ab_df has been loaded.")

    try:
        blockS, blockL = apply_stage1(ab_df) 
        del ab_df 
    except NotImplementedError as e:
        print(f"[WARNING] {e}, i am out of here bruh!")
        sys.exit(1)

    # Batch processing can reduce the overhead of task distribution and improve performance, particularly when combined with efficient numerical libraries.
    # load ab_string into dataframe [A, B, isNext/notNext]
    # get block_s_df, block_l_df
    # bu blockların statlarını dumpla
    # -------2.stage başlayacak--------



    pass

    