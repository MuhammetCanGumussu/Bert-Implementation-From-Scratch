# şimdilik main isminde sonradan düzenlemeler yapılacak

"""
29.06.2024

dosyaları al birleştir
başlık ve paragrafları birbirinden ayır
paragrafları all_dataset_string dosyasına yaz

toplam kaç tane example(paragraf/doc), kelime, sentence var onlara bak hatta kaydet bu bilgileri

AB_string dosyası oluştur (sliding olacak)

# şimdilik bunlar ---------------------------------------


"""

import os
import sys

raw_dir_path = "raw"
tr_wiki_prefix = "trwiki-67"

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



