# şimdilik main isminde sonradan düzenlemeler yapılacak

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
# import pandas as pd

import json
import os
import sys
import logging
import time
import random


tr_wiki_prefix = "trwiki-67"
merged_filename = 'merged.raw'
ab_string_path = "ab_string.txt"


NUM_PROCESSES = os.cpu_count()
print(f"[INFO] number of core available: {NUM_PROCESSES}")

# global scope is also executed for all subprocesses induvidually
# print(f"proc id: Global scope Before main block: {os.getpid()}")
  
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
    stat = {"num_sentences": 0, "num_words": 0, "num_chars": 0}

    if not os.path.exists(file):
        print(f'[INFO] {file} is not exists. Stat cannot be tracked...')
        return

    # merged.raw,  merged_preprocessed.raw
    if "merged" in file :

        if os.path.exists(file.split(".")[0] + "_stat.json"):
            print(f'[INFO] {file.split(".")[0] + "_stat.json"} already exists. Skipping stat tracking...')
            return
        
        titles, docs = split_titles_and_docs(file)

        stat["num_titles"] = len(titles)
        print(f"[INFO] merged file stat: num_titles: {stat['num_titles']}")

        stat["num_sentences"] = get_num_sents(docs)
        print(f"[INFO] merged file stat: num_sentences: {stat['num_sentences']}")

        stat["num_words"] = get_num_words(docs)
        print(f"[INFO] merged file stat: num_words: {stat['num_words']}")
        
        stat["num_chars"] = get_num_chars(docs)
        print(f"[INFO] merged file stat: num_chars: {stat['num_chars']}")

        # Write stat dict to JSON file
        with open(file.split(".")[0] + "_stat.json", 'w') as json_file:
            json.dump(stat, json_file, indent=4)

        print(f"[INFO] Stat has been written to 'merged_stat.json'...")


        return
    
     # belirsiz muallak
    elif "ab_string" in file:
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

def preprocess_merged(merged_file):
    """delete subtitles (if line has less than 4 words lets consider it as subtitle)"""
    
    print(f"[INFO] preprocessing is started...")

    # Configure logging to output to a file
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        handlers=[logging.FileHandler('preprocessing.log', 'w', 'utf-8')])

    titles, docs = split_titles_and_docs(merged_file)
    titles, docs = delete_if_empty_doc(titles, docs) 

    before_doc_len = len(docs)
    before_tit_len = len(titles)
    before_total_lines = len(" ".join(docs).splitlines())

    logging.info(f"[INFO] Before preprocessing: num_docs: {before_doc_len:,}")
    logging.info(f"[INFO] Before preprocessing: num_titles: {before_tit_len:,}")
    logging.info(f"[INFO] Before preprocessing: total_lines: {before_total_lines:,}")

    

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

    logging.info(f"[INFO] After preprocessing: num_docs: {len(docs):,}  diff: {(before_doc_len - len(docs)):,}")
    logging.info(f"[INFO] After preprocessing: num_titles: {len(titles):,}  diff: {(before_tit_len - len(titles)):,}")
    logging.info(f"[INFO] After preprocessing: total_lines: {after_total_lines:,}  diff: {(before_total_lines - after_total_lines):,}")

    assert len(titles) == len(docs), "[ERROR] len(titles) and len(docs) are not same!..."

    return (titles, docs)

def save_preprocessed(titles, docs):
    """takes preprocessed titles and docs, merges them and save them as new file"""
    
    assert len(titles) == len(docs), "[ERROR] len(titles) and len(docs) are not same!..."

    if os.path.exists("merged_preprocessed.raw"):
        print(f'[INFO] merged_preprocessed.raw already exists. Skipping preprocessing...')
        return
    
    print(f"[INFO] Saving preprocessed merged file...")

    for i in range(len(titles)):
        
        # Calculate progress percentage
        progress = (i / len(titles)) * 100
        progress = round(progress, 2)  # Round to 2 decimal places
    
        # Print progress bar
        print(f"Progress: [{int(progress)}%]", end='\r', flush=True)

        with open("merged_preprocessed.raw", 'a', encoding='utf-8') as merged:
            merged.write(titles[i])
            merged.write("\n" + docs[i] + "\n\n")

    print(f"[INFO] Preprocessed merged file saved as merged_preprocessed.raw...")

    return

def sliding_doc(doc):
    """ str -> sent_tokenization -> if len(sentences)> 1 return list[str(A <---> B), ...] else return None
        this is a function that will be used as "mapping" function
    """
    sentences = sent_tokenize(doc, language='turkish')

    ab_for_spesific_doc = []

    if len(sentences) > 1:
        for i in range(len(sentences) - 1):
            ab_for_spesific_doc.append(sentences[i] + " <---> " + sentences[i+1])

        return ab_for_spesific_doc
    else:
        return None

def get_random_sentence(ab_string):
    """ returns random B sentence from ab_string """
    random_index = random.randint(0, len(ab_string) - 1)

    return ab_string[random_index].split(" <---> ")[1] 
def shuffle_ab(ab_string):
    """ if 0.5 prob occurs, select random sentence from sentence_list and change b of ab with that sentence
        also append str with notNext
        if 0.5 prob not occurs, do not touch ab and append str with isNext

        this is a function that will be used as "mapping" function
    """

    if random.random() < 0.5:
        random_sentence = get_random_sentence()
        ab = ab_string.split(" <---> ")[0] + " <---> " + random_sentence + " <---> notNext"
        return ab
    else:
        return ab + " <---> isNext"

def load_ab_string(ab_string_path):
    """returns ab_string list[str(A <---> B <---> isNext), ...] if file is already exists else it will create it and returns"""

    if os.path.exists(ab_string_path):
        print(f'[INFO] {ab_string_path} already exists. AB_string is loading...')
        with open(ab_string_path, 'r') as f:
            return f.read().splitlines()

    print(f"[INFO] {ab_string_path} does not exist. AB_string is creating...")

    _, docs = split_titles_and_docs("merged_preprocessed.raw")

    # sliding window stage
    # her bir doc sent tokenize edilecek 
    # docs list[str] --> after mapping with sliding_doc --> list[list[str(A <---> B)], ...]
    # sliding_doc takes str -> sent tokenizasyon yapar -> len(sen)>1 ise sentencelar üzerinde sliding yaparak ab return eder (eğer len(sen)==1 ise None return eder)

    # Create a multiprocessing pool
    with mp.Pool(processes=(NUM_PROCESSES - 1) if NUM_PROCESSES > 1 else 1) as pool:
        # Apply the sliding_doc function to each doc_str using pool.map
        ab_strings = pool.map(sliding_doc, docs)

    # print number of None (number of docs that will be deleted bcs of len(sen)==1 (they can not be ab)) )
    print(f"[INFO] number of docs that will be deleted bcs of len(sen)==1 (they can not be ab): {ab_strings.count(None)}")

    # clean None's
    ab_strings = list(filter(lambda x: x is not None, ab_strings))

    # shuffling stage
    # list unpack edilecek --> list[str(A <---> B), str(A <---> B), str(A <---> B), ...]
    # list shuffle edilecek --> list[str(A <---> B <---> isNext), str(A <---> B <---> notNext), str(A <---> B <---> isNext), ...]
    

    # unpack ab_strings list[list[str]]--> list[str]
    ab_strings = [ab for ab_list in ab_strings for ab in ab_list]


    with mp.Pool(processes=(NUM_PROCESSES - 1) if NUM_PROCESSES > 1 else 1) as pool:
        # Apply the sliding_doc function to each doc_str using pool.map
        ab_strings = pool.map(shuffle_ab, ab_strings)

    # number of isNext and notNext
    print(f"[INFO] number of notNext: {ab_strings.count(' <---> notNext')}")

    # elde edilen liste dosyaya yazılacak ve list return edilecek

    with open(ab_string_path, 'w') as f:
        f.write('\n'.join(ab_strings))
        print(f"[INFO] {ab_string_path} has been created.")

    print(f"[INFO] finally total number of ab example: {len(ab_strings)}.")

    return ab_strings # list[str, str, ...]

  

if __name__ == '__main__':
    

    files_to_merge = [ f'{"raw" + "/" + tr_wiki_prefix}-train.raw',
                   f'{"raw" + "/" + tr_wiki_prefix}-val.raw', 
                   f'{"raw" + "/" + tr_wiki_prefix}-test.raw' ]

    

    merge_files(files_to_merge, merged_filename)


    # let's dump the stat of merged file
    dump_stat(merged_filename)


    # if preprocessed merged file is not exists, then preprocess it and save it
    if not os.path.exists("merged_preprocessed.raw"):
        print(f'[INFO] merged_preprocessed.raw is not exists. Preprocessing merged file...')

        titles, docs = preprocess_merged(merged_filename)
        save_preprocessed(titles, docs)

        # let's take stat of preprocessed merged file
        dump_stat("merged_preprocessed.raw")

    else:
        print(f"[INFO] merged_preprocessed.raw already exists. Skipping preprocessing...")
    
    
    #---------------------- Data Preprocessing Pipeline -------------------------
    # AB (sliding wind.) oluşturma componenti ile isNext/notNext shuffling componenti fuse'la text dosyası olarak kaydet (pandas df kullanılabilir, ayraç olarak bu <----> örüntü kullanılacak, text dosyası olarak kaydedilebilir)
    # Oluşturulan AB dosyasının stat bilgilerini getir
    # vocab'taki her token için token length bilgisi oluştur (daha sonra kullanılacak)
    # 

   

    ab_string_list = load_ab_string(ab_string_path) # list[str, str, ...]

    # dump_stat(ab_string_path) not implemented yet






#print(f"proc id: Global scope After main block: {os.getpid()}")



#----------------------------------------------------------------------








# sys.exit()
# 
# deneme_string = """
# == Çizgili arı şahini == 
# 
# Çizgili arı şahini ("Pernis celebensis"), atmacagiller (Accipitridae) familyasından yırtıcı bir kuş türü.
# Endonezya ve Filipinler'de bulunur. Doğal habitatları subtropikal veya tropikal nemli ova ormanları ve subtropikal veya tropikal nemli dağ ormanlarıdır
# 
# == Marcia Trimble == 
# 
# Marcia Trimble (1966 - 1975) 25 Şubat 1975 günü ailesinin Nashville,Tennessee,Amerika'daki evinden kaçtığında yalnızca 9 yaşındaydı.Marcia,komşularına Sincap Kız kurabiyeleri satıyordu.Kızın cesedi 33 gün sonra Paskalya Pazar'ında (30 Mart),evinden sadece 200 kilometre uzaklıktaki bir garajda bulundu.Otopsi sonucunda kıza cinsel tecavüz edildiği ve boğazlandığı öğrenildi.
# Çözülmemiş Cinayet Araştırması.
# Kızın nasıl kaçışı ve öldürülüşü bilinmemektedir.1979 yazında,cinayet işlendiği zaman 15 yaşında olan Jeffrey Womack,kızı öldürdüğü için hapse atıldı.Womack arkadaşlarına hep Trimble'ı öldürdüğünü böbürlenerek anlatıyordu.Bazı komşu çocuklarına göre,Marcia'nın evden kaçtığı gün yanında Womack de vardı.Womack bazı dedektiflere cinayet ile ilgili bazı ipuçları da söyledi.Womack iki tane yalan makinesi tarafından sorgulandı.1980 yılında Womack serbest bırakıldı
# """
# 
# for line in deneme_string.splitlines():
#     if line.startswith('== ') and line.endswith('== '):
#         print(line)
# 