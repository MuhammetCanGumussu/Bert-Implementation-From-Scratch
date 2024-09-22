"""
The purpose of this script is to create the data for the BERT training

NSP taski büyük veri setlerinde aslında avantajlı değil aksine model performansını düşürüyor (electra paper'i sanırsam buna bak)
Ancak küçük veri setleri için kullanımı (ki benim durumum) genel model performansını olumlu yönde etkiler.
NSP taskininde düzgün çalışabilmesi için A sonu, B başının gerçek cümlelerdeki gibi olması önemlidir.
Diğer türlü A cümle ortası ile biter, B de cümle ortası ile başlarsa (A: Ali ata (bak) B: (Sevdiğim) şarkılar bunlar..., notNext)
Model nsp taskinde bu tarz example'ları basit gramatik bilgiler ile (ki genelde modelin ilk öğreneceği şeylerdir) halledebilir, 
complex ilişkilendirmelere gerek duymaz.

görünüşe göre pad token'ına ihtiyacım olmayacak.

(., !, ?, ...) cümleleri bu tokenlara göre split edeceğim.

"kelime" açısından istenilen oranlar sağlansa da, span corruption yapıldığından "token" açısından
istatistiği merak ediyorum yüzdesel olarak ne kadar artacak?

toplam veri seti default overlap param ile yaklaşık 2 x artacak (overlap yarı old'dan 0,5 x gelecek, notNext old yani random sample
olduğunda kernel/block hareketi/posizyonu esasında değişmeyecek bunu da havadan yeni bir example oluşturma gibi düşünebilirsin + 0,5 x
daha toplamda yaklaşık olarak 2x artmış olur) 

tokenların yaklaşık yarısı bir epoch'da 3 kere gözükecek (overlap + random sample sayesinde)
kalan yarısı da 2 kere gözükecek (sadece overlap sayesinde) 

random word set stage:
    * bunu script şekilde yazdım. burada (data.py) eğer beklenen dosya yok ise bu exec edilsin
    * word_count_per_token_len_group (örn 500 ise her token grubu için en fazla 500 tane kelime olacak )
    * 
    * 
    *


Her stage hali hazırda yapılmış ise atlansın
Her stage kalınıldığı noktadan devam edebilmeli (shard index ile)
1st Stage(documents):
    * read trwiki-67 files and merge them (as a big one string object)
    * split titles and docs (just keep docs btw, i will not use titles for training)(boş/null doc olmamalı)
    * tüm docs list shuffle edilsin (raw dosya dizilimi random şekilde mi bilmiyorum, ondan her türlü bir shuffling yapalım)
    * delete subtitles from docs (aproximately if line has less than 4 words) (list of docs)
    * save this list of docs json in shards
    * bu stage'in düzgün çalışıp çalışmadığını kontrol et
    * free all resources

Her stage hali hazırda yapılmış ise atlansın
Her stage kalınıldığı noktadan devam edebilmeli (shard index ile)
2nd Stage (ab_tokenized):
    * 2.1 Stage (tokenization)
        + load docs json file as pandas dataframe (fonk yaz belirtilen shard indexini yükleyen)
        + en son kalınılan shard'tan devam edebilmeli
        + her doc "string" tokenize edilecek yeni kolonlar türeyecek: token_ids, word_ids
        + düz string kolonu dropla (list[str]) olan
        + her doc'da sentence indexlerini bulmalı yeni kolona bunları koymalı (doc uzunluğu en sonki index'mi öyle değil ise ekle)
        + kısaca token len için ayrı kolon olmayacak, sentence index'lerdeki en son elemandan bu çıkartılabilmeli
        + 2 cümleden az olan (yani tek cümleli) doc'lar filtrelenecek
        + doc [token_ids, word_ids, sent_idx]
        + bu stage'in düzgün çalışıp çalışmadığını kontrol et
        + 2.2'ye geçmeden free resources

    * 2.2 Stage (ab creation)
        + block_size, overlap, shard başına kaç ab sample (shard_size) gibi parametrelere göre çalışacak
        + isnext olasılığı al, next ise:
            - bloğu doldurulabildiği kadar doldurulacak, yer kalırsa (truncation sayesinde çoğu durumda kalmasa da document'ın sonuna
              geldiğimizde dolduramama ihtimali var) [PAD] token'ları ile doldurulacak
        + notNext ise:
            - random sample alınacak, sample genişliği belli kısıtlamalar ile olacak, eğer kısıtlamalar sağlanamıyorsa başka random sample
              yapılacak
            - random sample alındıktan sonra bulunulan kernel/block posizyonunda gene alınabilecek kadar alınacak (sonu cümle gibi bitmeli)
            - kalan yerler pad ile doldurulacak (bunda pad olasılığı yüksek)
        + tüm shard shuffle edilmeli (ab halde shuffle et shard'ı)
        + 2.stage sonu, shard kayıtları (ab sayılarına göre yapılabilir (her doc farklı sayıda ab çıkarabilir sonuçta) ancak aynı kalsın, zaten son
          aşamada her shard'ı token sayısı eşit olacak biçimde tasarlayacağım (o aşamaya kadar şard'larımız hep aynı sayıda doc olacak şekilde))
        + bu stage'in düzgün çalışıp çalışmadığını kontrol et
        + free all resources

        ++ toplam veri seti default overlap param ile yaklaşık 2 x artacak (overlap yarı old'dan 0,5 x gelecek, notNext old yani random sample
           olduğunda kernel/block hareketi/posizyonu esasında değişmeyecek bunu da havadan yeni bir example oluşturma gibi düşünebilirsin + 0,5 x
           daha toplamda yaklaşık olarak 2x artmış olur) 
           tokenların yaklaşık yarısı bir epoch'da 3 kere gözükecek (overlap + random sample sayesinde)
        ++ kalan yarısı da 2 kere gözükecek (sadece overlap sayesinde) 
    

Her stage hali hazırda yapılmış ise atlansın
Her stage kalınıldığı noktadan devam edebilmeli (shard index ile)
3rd Stage(xy_numpy):
    * xy'ler numpy arrayler vs oluşturma, SON aşama
    * % kaç [MASK] token'i var?, % kaç yer değiştirme, aynı bırakma var vs ("kelime" açısından istenilen oranlar sağlansa da, span corruption yapıldığından "token" açısından
    istatistiği merak ediyorum yüzdesel olarak ne kadar artacak?) 
    * ekstrem kelimelerde (8 token misal) random kelime yoksa kaçsın ya da MASK yapsın
    * random word json, unpackleneerek sadece list[list[int]] şekline çevirip kullanmayı deneyelim (idx'ler token len)
      böylece manager dict yerine düz Array kullanabiliriz
    * random kelimeleri negative sample gibi düşünebiliriz



"""

# standard library
import os
import sys
import json
import logging
import random
import multiprocessing as mp
from typing import List


# third party library
import tqdm
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast











tr_wiki_prefix = "trwiki-67"
preprocess_and_stats_dir = "preprocess_and_stats"
random_words_set_dir = "random_words_set"

merged_filename = preprocess_and_stats_dir + "/merged.raw"
ab_string_path = preprocess_and_stats_dir + "/ab_string.raw"
merged_preprocess_path = preprocess_and_stats_dir + "/merged_preprocessed.raw"



random_words_set_path = random_words_set_dir + "/random_words_set.json"
tokenizer_path = "../tr_wordpiece_tokenizer_cased.json"


NUM_PROCESSES = (os.cpu_count() - 1) if os.cpu_count() > 1 else 1

def get_merged_files():

    
    raw_dir = os.path.join(os.path.dirname(__file__), "raw")

    files = os.listdir(raw_dir)

    print(f"[INFO] Files in dir: {files}...")

    merged_file_content = ""

    for raw_file in files:
        with open(os.path.join(raw_dir, raw_file), encoding="utf-8") as raw:
            merged_file_content += (raw.read() + "\n")

    return merged_file_content

    

def get_tokenizer(tokenizer_path, fast=True):

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
        )

    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer


def appy_seed(number=13013):
    random.seed(42) # reproducibility


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



def split_text_to_words(text: str) -> List[str]:
    # isalnum() -> checks if all characters are alphanumeric (a-z, A-Z, 0-9)]
    # word_tokenize sees non alphanumeric characters as words so we need to remove them by isalnum
    # for example: ( "[SEP]" ) -> ( "[", "SEP", "]" )

    words = [word_cand for word_cand in word_tokenize(text, language="turkish") if word_cand.isalnum()]
   
    # lets convert SEP -> [SEP] again, and return
    return ["[SEP]" if word == "SEP" else word for word in words]




def get_num_chars(docs):
    "docs: list[str, str, str, ...]"
    return len(''.join(docs))
    
def get_num_words(docs):
    "docs: list[str, str, str, ...]"
    import re

    # Regular expression to match words (alphanumeric characters and apostrophes)
    word_pattern = r'\b\w+\b'

    return len(re.findall(word_pattern, ' '.join(docs)))

def get_num_sents(docs):
    "docs: list[str, str, str, ...]"
    return len(sent_tokenize('. '.join(docs), language='turkish'))

def dump_stat(file):
    """file: [preprocess_and_stats/merged.raw] or [preprocess_and_stats/merged_preprocessed.raw] or ..."""
    stat = {"num_sentences": 0, "num_words": 0, "num_chars": 0}

    # en başta dosyanın kendisi olmayabilir
    if not os.path.exists(file):
        print(f'[INFO] {file} is not exists. Stat cannot be tracked...')
        return

    if "merged" in file :

        # [preprocess_and_stats/merged_stat.json] or [preprocess_and_stats/merged_preprocessed_stat.json]
        file_with_stat = file.split(".")[0] + "_stat.json"

        if os.path.exists(file_with_stat):
            print(f'[INFO] {file_with_stat} already exists. Skipping stat tracking...')
            return
        
        print(f"[INFO] {file_with_stat} is not exists. Stat will be tracked...")
        
        titles, docs = split_titles_and_docs(file)

        stat["num_titles_and_docs"] = len(titles)
        print(f"[INFO] merged file stat: num_titles and docs: {stat['num_titles_and_docs']}")

        stat["num_sentences"] = get_num_sents(docs)
        print(f"[INFO] merged file stat: num_sentences: {stat['num_sentences']}")

        stat["num_words"] = get_num_words(docs)
        print(f"[INFO] merged file stat: num_words: {stat['num_words']}")
        
        stat["num_chars"] = get_num_chars(docs)
        print(f"[INFO] merged file stat: num_chars: {stat['num_chars']}")

        # Write stat dict to JSON file
        with open(file_with_stat, 'w') as json_file:
            json.dump(stat, json_file, indent=4)

        print(f"[INFO] Stat has been written to {file_with_stat} ...")

        return
    

    elif "ab_string" in file:
        # file: preprocess_and_stats/ab_string.raw
        # file_with_stat:preprocess_and_stats/ab_string_stat.json
        file_with_stat = file.split(".")[0] + "_stat.json"

        if os.path.exists(file_with_stat):
            print(f'[INFO] {file_with_stat} already exists. Skipping stat tracking...')
            return
        
        print(f"[INFO] {file_with_stat} is not exists. Stat will be tracked...")

        # load ab_strings (file: preprocess_and_stats/ab_string.raw)
        ab_strings = load_ab_string(file)

        # Create a multiprocessing pool
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            # ab_strings without seperator [SEP] and isNext/notNext
            ab_strings = pool.map(_del_seperator_and_isNext, ab_strings)

        stat["num_ab_string_row"] = len(ab_strings)
        print(f"[INFO] {file} stat: num_ab_string_row: {stat['num_ab_string_row']}")

        stat["num_sentences"] = get_num_sents(ab_strings)
        print(f"[INFO] {file} stat: num_sentences: {stat['num_sentences']}")

        stat["num_words"] = get_num_words(ab_strings)
        print(f"[INFO] {file} stat: num_words: {stat['num_words']}")
        
        stat["num_chars"] = get_num_chars(ab_strings)
        print(f"[INFO] {file} stat: num_chars: {stat['num_chars']}")

        # Write stat dict to JSON file
        with open(file_with_stat, 'w') as json_file:
            json.dump(stat, json_file, indent=4)

        print(f"[INFO] Stat has been written to {file_with_stat} ...")



        return
    
    # belirsiz muallak
    elif "ab_tokenized" in file:
        return
    else:
        print(f'[INFO] {file}, filename is not correct...')
        return
    
    
    return

def merge_files(raw_files, merged_file):
    if os.path.exists(merged_file):
        print(f'[INFO] {merged_file} already exists. Skipping merge.')
        return

    with open(merged_file, 'w', encoding="utf-8") as merged:
        for raw_file in raw_files:
            with open(raw_file, "r", encoding="utf-8") as raw:
                merged.write(raw.read())
                merged.write("\n")

    print(f'[INFO] Files {raw_file} have been merged into {merged_file}')

def delete_if_empty_doc(titles, docs):
    """if there is empty doc, it will be deleted with its title"""

    print(f"[INFO] delete_if_empty_doc()...")

    for i in range(len(docs)):
        if docs[i] == "\n" or docs[i] == "":
            print(f"[INFO] doc: {docs[i]}is empty. Deleting it with its title: {titles[i]}...")
            titles.pop(i)
            docs.pop(i)

    # lets make sure that titles and docs are same size before returning
    assert len(titles) == len(docs), "[ERROR] len(titles) and len(docs) are not same!..."

    return titles, docs

def delete_subtitles(merged_file):
    """delete subtitles (if line has less than 4 words lets consider it as subtitle)
       and also delete empty docs and related titles
    """
    
    print(f"[INFO] preprocessing is started...")

    # Configure logging to output to a file
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        handlers=[logging.FileHandler(preprocess_and_stats_dir + '/preprocessing.log', 'w', 'utf-8')])

    titles, docs = split_titles_and_docs(merged_file)
    titles, docs = delete_if_empty_doc(titles, docs) 

    before_doc_len = len(docs)
    before_tit_len = len(titles)
    before_total_lines = len(" ".join(docs).splitlines())

    logging.info(f"[INFO] Before delete_subtitles: num_docs: {before_doc_len:,}")
    logging.info(f"[INFO] Before delete_subtitles: num_titles: {before_tit_len:,}")
    logging.info(f"[INFO] Before delete_subtitles: total_lines: {before_total_lines:,}")

    

    import re

    # Regular expression to match words (alphanumeric characters and apostrophes)
    word_pattern = r'\b\w+\b'

    i = 0
    while i < len(docs):
        doc_lines = docs[i].split("\n")

        # Calculate progress percentage
        progress = (i / len(docs)) * 100
        progress = round(progress, 2)  # Round to 2 decimal places
    
        # Print progress bar
        print(f"Progress: [{int(progress)}%]", end='\r', flush=True)

        j = 0
        while j < len(doc_lines):

            # print(j, len(doc_lines))
            doc_line_len = len(re.findall(word_pattern, doc_lines[j]))

            if doc_line_len < 4:
                # just one line in the doc and it's less than 4 words, burn it!
                if len(doc_lines) == 1:
                    titles.pop(i)
                    docs.pop(i)
                    i -= 1
                    break

                doc_lines.pop(j)
                j -= 1


            j += 1

        docs[i] = "\n".join(doc_lines)
        i += 1  

    after_total_lines = len(" ".join(docs).splitlines())

    logging.info(f"[INFO] After delete_subtitles: num_docs: {len(docs):,}  diff: {(before_doc_len - len(docs)):,}")
    logging.info(f"[INFO] After delete_subtitles: num_titles: {len(titles):,}  diff: {(before_tit_len - len(titles)):,}")
    logging.info(f"[INFO] After delete_subtitles: total_lines: {after_total_lines:,}  diff: {(before_total_lines - after_total_lines):,}")

    assert len(titles) == len(docs), "[ERROR] len(titles) and len(docs) are not same!..."

    return (titles, docs)

def save_preprocessed(titles, docs):
    """takes preprocessed titles and docs, merges them and save them as new file"""
    
    assert len(titles) == len(docs), "[ERROR] len(titles) and len(docs) are not same!..."

    if os.path.exists(merged_preprocess_path):
        print(f'[INFO] {merged_preprocess_path} already exists. Skipping preprocessing...')
        return
    
    print(f"[INFO] Saving preprocessed merged file...")

    for i in range(len(titles)):
        
        # Calculate progress percentage
        progress = (i / len(titles)) * 100
        progress = round(progress, 2)  # Round to 2 decimal places
    
        # Print progress bar
        print(f"Progress: [{int(progress)}%]", end='\r', flush=True)

        with open(merged_preprocess_path, 'a', encoding='utf-8') as merged:
            merged.write(titles[i])
            merged.write("\n" + docs[i] + "\n\n")

    print(f"[INFO] Preprocessed merged file saved as {merged_preprocess_path}...")

    return

def sliding_doc(doc):
    """ str -> sent_tokenization -> if len(sentences)> 1 return list[str(A [SEP] B), ...] else return None
        this is a function that will be used as "mapping" function
    """
    sentences = sent_tokenize(doc, language='turkish') # doclardaki ilk sentenceların başında 2 tane \n\n bulunuyor bunu temizlemeli
                                                       # ya burda yada buraya gelmeden docs üzerinde bu characterleri temizlemeli
    ab_for_spesific_doc = []

    if len(sentences) > 1:
        for i in range(len(sentences) - 1):
            ab_for_spesific_doc.append(sentences[i] + " [SEP] " + sentences[i+1])

        return ab_for_spesific_doc
    else:
        return None


def shuffle_ab(ab_and_random_tuple_list):
    """ if 0.5 prob occurs, b of ab will be replaced by random_sentence
        also append str with notNext
        if 0.5 prob not occurs, do not touch ab and append str with isNext

        this is a function that will be used as "mapping" function
    """
    ab_string, random_sentence = ab_and_random_tuple_list

    if random.random() < 0.5:
        #                          A           [SEP]         Random_B        [SEP] notNext
        ab = ab_string.split(" [SEP] ")[0] + " [SEP] " + random_sentence + " [SEP] notNext"
        return ab
    else:
        return ab_string + " [SEP] isNext"

def _del_seperator_and_isNext(ab_string):
    """ mapping function for list[str(A [SEP] B [SEP] isNext), ...] -> list[str(AB), ...]
        takes one string (A [SEP] B [SEP] isNext) and returns just plain ( AB ) 
    """
    return "".join(ab_string).replace(" [SEP] ", " ").replace(" notNext", "").replace(" isNext", "")

def load_ab_string(ab_string_path, with_seperator=True):
    """returns ab_string list[str(A [SEP] B [SEP] isNext), ...] if file is already exists else it will create it and returns"""

    if os.path.exists(ab_string_path):
        print(f'[INFO] {ab_string_path} already exists. AB_string is loading...')
        with open(ab_string_path, 'r', encoding='utf-8') as f:
            return f.read().splitlines()

    print(f"[INFO] {ab_string_path} does not exist. AB_string is creating...")

    _, docs = split_titles_and_docs(merged_preprocess_path)


    # Create a multiprocessing pool
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        ab_strings = list(tqdm.tqdm(pool.imap(sliding_doc, docs, chunksize=NUM_PROCESSES * 2), total=len(docs), desc="Sliding window processing..."))

    print(f"[INFO] number of docs deleted bcs of len(sen)==1 (because they can not be ab): {ab_strings.count(None)}")

    # clean None's
    ab_strings = list(filter(lambda x: x is not None, ab_strings))


    # unpack ab_strings list[list[str]]--> list[str]
    ab_strings = [ab for ab_list in ab_strings for ab in ab_list]


    random_sentences = [ab.split(" [SEP] ")[1] for ab in ab_strings]
    ab_and_random_tuple_list = list(zip(random.sample(ab_strings, len(ab_strings)),
                                        random.sample(random_sentences, len(ab_strings))))
    
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        ab_strings = list(tqdm.tqdm(pool.imap(shuffle_ab, ab_and_random_tuple_list, chunksize=NUM_PROCESSES * 2), total=len(ab_strings), desc="Shuffling AB strings"))

    
    print(f"[INFO] deleting consecutive newline characters...")
    # ardışık \n\n'ları sil (doc başlangıcınlarından kaynaklı preprocess)
    # ab_strings ve all_string aynı mem'i şişiriyor gerekirse birinden biri atılabilir (tabi tekrar list[str, str] ops yapmak gerekicek)
    all_string = '\n'.join(ab_strings)
    all_string = all_string.replace("\n\n", "")

    # number of isNext and notNext
    print(f"[INFO] number of notNext: {all_string.count('notNext')}")

    # elde edilen liste dosyaya yazılacak ve list return edilecek
    with open(ab_string_path, 'w', encoding="utf-8") as f:
        f.write(all_string)
        print(f"[INFO] {ab_string_path} has been created.")


    print(f"[INFO] finally, total number of ab example: {len(ab_strings)}.")

    
    return ab_strings # list[str, str, ...]

  

if __name__ == '__main__':

    print(f"[INFO] number of core available: {NUM_PROCESSES}")


    files_to_merge = [ f'{"raw" + "/" + tr_wiki_prefix}-train.raw',
                   f'{"raw" + "/" + tr_wiki_prefix}-val.raw', 
                   f'{"raw" + "/" + tr_wiki_prefix}-test.raw' ]


    merge_files(files_to_merge, merged_filename)


    # if preprocessed merged file is not exists, then preprocess it and save it
    if not os.path.exists(merged_preprocess_path):
        print(f'[INFO] {merged_preprocess_path} is not exists. Preprocessing merged file (delete subtitles)...')

        titles, docs = delete_subtitles(merged_filename)
        save_preprocessed(titles, docs)
        del titles
        del docs

    else:
        print(f"[INFO] {merged_preprocess_path} already exists. Skipping preprocessing (delete subtitles)...")

    if not os.path.exists(random_words_set_path):
        print(f'[INFO] {random_words_set_path} is not exists. random_word_set.py gonna executed...')

        import subprocess
        subprocess.run(["python", "random_word_set.py"])

    # lets load random words set (if it was not created already, random_word_set.py will create it above)
    with open(random_words_set_path, 'r', encoding="utf-8") as json_file:
        random_words_dict = json.load(json_file)
        print(f'[INFO] {random_words_set_path} has been loaded.')


    # let's dump the stat of merged file
    dump_stat(merged_filename)
    # let's take stat of preprocessed merged file
    dump_stat(merged_preprocess_path)
    # let's dump the stat of ab_string.raw
    dump_stat(ab_string_path)

   
    # TODO hali hazırda var ise shardlar ellenemeyecek
    print(f"[INFO] Creating x,y shards...")
    create_xy_shards(random_words_dict)
    
    

    

