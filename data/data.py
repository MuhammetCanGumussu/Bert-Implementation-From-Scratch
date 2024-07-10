# şimdilik main isminde sonradan düzenlemeler yapılacak

# TODOS:
# ŞU MERGED, AB VS DOSYALARI VS STAT DOSYALARINI DÜZENLE!!!
# LOG dosyasında hangi preprocess old belirtilmeli! + aslında log olarak tutmak iyi değil gibi neyse bak
# PATH VAR'LARINI Düzenle
# "preprocess" kelimesini çok fazla ve belirsiz şekilde kullanmışım (delete subtitle ile vs anlamlı isimlendirmelere çevir)
# dump_stat baya modularize edilebilir/edilmeli
# map fonksiyonların başına "_" koy
# stat fonk'ları mp'lenebilir
# shardinge aslında gerek yoktu, kendimi denemek için koydum

"""
29.06.2024

dosyaları al birleştir
başlık ve paragrafları birbirinden ayır
paragrafları all_dataset_string dosyasına yaz

toplam kaç tane example(paragraf/doc), kelime, sentence var onlara bak hatta kaydet bu bilgileri
merged.raw dosyası için ayrı,
preprocessed merged için ayrı,
ab.raw dosyası için ayrı istatistik tutulacak

AB_string dosyası oluştur (sliding olacak)

github push yapılacak

# şimdilik bunlar ---------------------------------------

subtitle'ları temizlemek gerekiyor!!!! 


"""


from nltk.tokenize import sent_tokenize
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
# import pandas as pd

import json
import os
import sys
import logging
import time
import tqdm
import random

random.seed(42) # reproducibility

tr_wiki_prefix = "trwiki-67"
preprocess_and_stats_dir = "preprocess_and_stats"
random_words_set_dir = "random_words_set"

merged_filename = preprocess_and_stats_dir + "/merged.raw"
ab_string_path = preprocess_and_stats_dir + "/ab_string.raw"
merged_preprocess_path = preprocess_and_stats_dir + "/merged_preprocessed.raw"

random_words_set_path = random_words_set_dir + "/random_words_set.json"


if mp.current_process().name == 'MainProcess':
    NUM_PROCESSES = (os.cpu_count() - 1) if os.cpu_count() > 1 else 1
    print(f"[INFO] number of core available: {NUM_PROCESSES}")


CHUNK_SIZE = 16


def split_titles_and_docs(filename):
    """Returns a list of titles and a list of documents from merged file"""

    titles = []
    docs = []

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        doc_lines = []

        for line in lines:

            if line.startswith('== ') and line.endswith('== \n'):
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

    assert len(titles) == len(docs), "[ERROR] len(titles) and len(docs) are not same!..."

    return titles, docs

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
        ab_strings = list(tqdm.tqdm(pool.imap(sliding_doc, docs, chunksize=CHUNK_SIZE // 2), total=len(docs), desc="Sliding window processing..."))

    print(f"[INFO] number of docs deleted bcs of len(sen)==1 (because they can not be ab): {ab_strings.count(None)}")

    # clean None's
    ab_strings = list(filter(lambda x: x is not None, ab_strings))


    # unpack ab_strings list[list[str]]--> list[str]
    ab_strings = [ab for ab_list in ab_strings for ab in ab_list]


    random_sentences = [ab.split(" [SEP] ")[1] for ab in ab_strings]
    ab_and_random_tuple_list = list(zip(random.sample(ab_strings, len(ab_strings)),
                                        random.sample(random_sentences, len(ab_strings))))
    
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        ab_strings = list(tqdm.tqdm(pool.imap(shuffle_ab, ab_and_random_tuple_list, chunksize=CHUNK_SIZE*5), total=len(ab_strings), desc="Shuffling AB strings"))

    
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
    

    files_to_merge = [ f'{"raw" + "/" + tr_wiki_prefix}-train.raw',
                   f'{"raw" + "/" + tr_wiki_prefix}-val.raw', 
                   f'{"raw" + "/" + tr_wiki_prefix}-test.raw' ]


    merge_files(files_to_merge, merged_filename)


    # if preprocessed merged file is not exists, then preprocess it and save it
    if not os.path.exists(merged_preprocess_path):
        print(f'[INFO] {merged_preprocess_path} is not exists. Preprocessing merged file (delete subtitles)...')

        titles, docs = delete_subtitles(merged_filename)
        save_preprocessed(titles, docs)

    else:
        print(f"[INFO] {merged_preprocess_path} already exists. Skipping preprocessing (delete subtitles)...")


    ab_strings = load_ab_string(ab_string_path) # list[str, str, ...]

    

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


    # ----------------------------------------------------------------------------------------
    # batched map
    # sharding (split yüzdeleri burada belirlenecek, shard numpy dosyaları isimleri için andreje bak)
    # tokenize ab
    # block scheduling (bl_size1 kırmızı, bl_size2 mavi, küme operasyonları)


