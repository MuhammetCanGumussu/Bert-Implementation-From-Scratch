# şimdilik main isminde sonradan düzenlemeler yapılacak

"""
29.06.2024

dosyaları al birleştir
başlık ve paragrafları birbirinden ayır
paragrafları all_dataset_string dosyasına yaz

toplam kaç tane example(paragraf/doc), kelime, sentence var onlara bak hatta kaydet bu bilgileri
merged.raw dosyası için ayrı, ab.raw dosyası için ayrı istatistik tutulacak

AB_string dosyası oluştur (sliding olacak)

github push yapılacak

# şimdilik bunlar ---------------------------------------

subtitle'ları temizlemek gerekiyor!!!! 


"""


from nltk.tokenize import sent_tokenize

import json
import os
import sys

raw_dir_path = "raw"
tr_wiki_prefix = "trwiki-67"



deneme = """Ömer Duran (d. 16 Temmuz 1972) Türk sinema ve dizi oyuncusudur.
Kariyeri.
İstanbul Üniversitesi Fransız Dili Ve Edebiyat Fakültesi mezunudur. Kartal Halk Eğitim Merkezinde ilk gerçek eğitimini aldı. Sonrasında birçok workshopa katıldı. Amatör gruplarla defalarca sahneye çıktı. Sadri Alışık Tiyatrosu ile 4 oyunda rol aldı. Selahattin Taşdöğen Tiyatrosunda sahne aldı. TV kariyerine 1994 yılında Kaygısızlar (dizi) isimli tv dizisinde başladı.Bu diziden sonra çeşitli dizilerde görev aldı. 2011 yılında vizyona giren "72.Koguş" filminde Kaya Ali rolünü canlandırdı, 2012 yılında Babalar ve Evlatlar adlı dizide Kambur Ziya rolü ile Star Tv'de yer alıyor.
Duran bir dönem sahaflık yaptı. 19 Kasım 2018 tarihinde yayınlanan Kelime Oyunu yarışma programına katıldı. 2 kız çocuğu babasıdır"""

import code; code.interact(local=locals())

sys.exit()




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
                pass

        # Append the last document
        docs.append("\n".join(doc_lines))

    return titles, docs

# titles, docs = split_titles_and_docs("merged.raw")
# 
# print(f"number of titles: {len(titles)}")
# print(f"number of docs: {len(docs)}")
# 
# import code; code.interact(local=locals())
# 
# print("exit...")
# sys.exit()


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

files_to_merge = [ f'{raw_dir_path + "/" + tr_wiki_prefix}-train.raw',
                   f'{raw_dir_path + "/" + tr_wiki_prefix}-val.raw', 
                   f'{raw_dir_path + "/" + tr_wiki_prefix}-test.raw' ]

merged_filename = 'merged.raw'

merge_files(files_to_merge, merged_filename)

def get_num_chars(docs):
    "docs: list[str, str, str, ...]"
    pass
def get_num_words(docs):
    "docs: list[str, str, str, ...]"
    word_tokenize(docs, language="turkish", )
    pass
def get_num_sents(docs):
    "docs: list[str, str, str, ...]"
    pass


def get_stat(file):
    stat = {"num_sentences": 0, "num_words": 0, "num_chars": 0}

    if not os.path.exists(file):
        print(f'[INFO] {file} is not exists. Stat cannot be tracked...')
        return

    if "merged" in file:
        titles, docs = split_titles_and_docs(file)

        stat["num_titles"] = len(titles)
        stat["num_sentences"] = get_num_sents(docs)
        stat["num_words"] = get_num_words(docs)
        stat["num_chars"] = get_num_chars(docs)

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
















sys.exit()

deneme_string = """
== Çizgili arı şahini == 

Çizgili arı şahini ("Pernis celebensis"), atmacagiller (Accipitridae) familyasından yırtıcı bir kuş türü.
Endonezya ve Filipinler'de bulunur. Doğal habitatları subtropikal veya tropikal nemli ova ormanları ve subtropikal veya tropikal nemli dağ ormanlarıdır

== Marcia Trimble == 

Marcia Trimble (1966 - 1975) 25 Şubat 1975 günü ailesinin Nashville,Tennessee,Amerika'daki evinden kaçtığında yalnızca 9 yaşındaydı.Marcia,komşularına Sincap Kız kurabiyeleri satıyordu.Kızın cesedi 33 gün sonra Paskalya Pazar'ında (30 Mart),evinden sadece 200 kilometre uzaklıktaki bir garajda bulundu.Otopsi sonucunda kıza cinsel tecavüz edildiği ve boğazlandığı öğrenildi.
Çözülmemiş Cinayet Araştırması.
Kızın nasıl kaçışı ve öldürülüşü bilinmemektedir.1979 yazında,cinayet işlendiği zaman 15 yaşında olan Jeffrey Womack,kızı öldürdüğü için hapse atıldı.Womack arkadaşlarına hep Trimble'ı öldürdüğünü böbürlenerek anlatıyordu.Bazı komşu çocuklarına göre,Marcia'nın evden kaçtığı gün yanında Womack de vardı.Womack bazı dedektiflere cinayet ile ilgili bazı ipuçları da söyledi.Womack iki tane yalan makinesi tarafından sorgulandı.1980 yılında Womack serbest bırakıldı
"""

for line in deneme_string.splitlines():
    if line.startswith('== ') and line.endswith('== '):
        print(line)
