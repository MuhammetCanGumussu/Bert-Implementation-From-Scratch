"""

### Random Word Set Oluşturma Adımları

1. **Kelimeleri Bulma**  
   Girilen metin içerisinde, kelimeleri bulmalıyım. Bunu `tokenizer.normalize_str` ve `tokenizer.pre_tokenize_str` kullanarak gerçekleştireceğim ki böylece tokenizer pipeline'nına bağlı kalalım. (sayı ve punctiation karakterlerini random olarak koymak istemediğimden random word set'ten atacağım)

2. **Tokenizasyon**  
   Bulunan kelimeler, tokenize edilmelidir.

3. **Statistiksel Analiz ve Temizleme**  
   Nadir kelimeler atılmalı. Daha sonra token_len'e göre gruplaştırma yapılmalı. Bu seferde gruplara göre stat bakılmalı: 3 tokenlık kaç kelime var vs gibi. Noktalama, sayılar, semboller de atılmalı!

4. **Son Temsil**  
   Son temsil aşağıdaki formatta olmalıdır:
   ```plaintext
   (token_size: 5, 5_tokenlık_kelime_listesi_tokenized_btw, word_count: 5644) 
   ```
   - `word_count` atılmamalı, rastgele indeks alırken kullanılacaktır.

5. **Dosya Kaydetme**  
   Temizlenmiş veri, bir JSON dosyası olarak kaydedilmelidir.



"""


# standard library
import os
import sys
import itertools
import multiprocessing as mp
from collections import Counter

# third party library
import tqdm
import pandas as pd
from tokenizers import normalizers, Regex, pre_tokenizers

# local library
from ..data import data_aux
from ..tokenizer.train_tokenizer import get_tokenizer
from ..config import get_random_word_set_py_config





root_dir = os.path.dirname(os.path.abspath(__file__))
random_word_set_save_path = root_dir + "/random_word_set/random_word_set.json"
tokenizer_path = root_dir + "/tokenizer/tr_wordpiece_tokenizer_cased.json" 



tokenizer = get_tokenizer(root_dir + "/tokenizer/tr_wordpiece_tokenizer_cased.json")


pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), 
                                               pre_tokenizers.Digits(individual_digits=True),
                                               pre_tokenizers.Punctuation()])


normalizer = normalizers.Sequence([normalizers.NFKC(),
                                           normalizers.Lowercase(),
                                           normalizers.Replace(Regex('[^\w\s]'),""),   # for numbers
                                           normalizers.Replace(Regex('\d+'),"") ])     # for punctiations



def get_random_word_set():
    if not os.path.exists(random_word_set_save_path):
        print(f"[INFO] {random_word_set_save_path} is not already exists. Try to execute random_word_set.py script to generate this file before calling this function...")
        sys.exit(0)

    random_word_set_df = pd.read_json(random_word_set_save_path, orient="records", lines=True, encoding="utf-8")
    random_word_set_dict = {}
    
    for group_name, group in random_word_set_df.groupby("token_len"):
        random_word_set_dict[f"token_group_{group_name}"] = group["token_ids"].to_list()
        random_word_set_dict[f"token_group_{group_name}_length"] = len(random_word_set_dict[f"token_group_{group_name}"])
    
    return random_word_set_dict



def normalize_line(line):
    return normalizer.normalize_str(line)

def pre_tokenize_line(line):
    pretokenized_tuple = pre_tokenizer.pre_tokenize_str(line)
    return [each_tuple[0] for each_tuple in pretokenized_tuple]

def tokenize_word(row):
   row = row[1]
   row["token_ids"] = tokenizer(row["word"])["input_ids"]
   row["token_len"] = len(row["token_ids"])
   return row




if __name__ == "__main__":
   
   cfg = get_random_word_set_py_config()


   LIMIT_FOR_TOKEN_GROUP = cfg.limit_for_token_group
   MAX_WORD_LIMIT_FOR_TOKEN_GROUP = cfg.max_word_limit_for_token_group
   MIN_FREQ_FOR_WORDS = cfg.min_freq_for_words

   RANDOM_SAMPLE = cfg.random_sample
   USE_NUMBER_OF_LINE = cfg.use_number_of_line


   if os.path.exists(random_word_set_save_path):
       print(f"[INFO] {random_word_set_save_path} is already exists Terminating...")
       exit(0)


   NUM_PROCESSES = (os.cpu_count() - 1) if os.cpu_count() > 1 else 1
   print(f"[INFO] Using {NUM_PROCESSES} processes...")


   merged_file_content = data_aux.get_merged_files()

   if USE_NUMBER_OF_LINE is None:
       merged_file_content = merged_file_content.splitlines()
   else:
       merged_file_content = merged_file_content.splitlines()[:USE_NUMBER_OF_LINE]

   len_merged_file_content = len(merged_file_content)

                                  

   # multiprocessing pool
   with mp.Pool(NUM_PROCESSES) as pool:
       merged_file_content = pool.imap(normalize_line, merged_file_content, chunksize= 2048)
       merged_file_content = list(tqdm.tqdm(merged_file_content, total=len_merged_file_content, desc="[INFO] Normalizing lines..."))    

   # multiprocessing pool
   with mp.Pool(NUM_PROCESSES) as pool:
       merged_file_content = pool.imap(pre_tokenize_line, merged_file_content, chunksize= 2048)
       merged_file_content = list(tqdm.tqdm(merged_file_content, total=len_merged_file_content, desc="[INFO] Pre-tokenizing lines..."))    
   

   # list[list[str]] -> list[str]
   merged_file_content = list(itertools.chain.from_iterable(merged_file_content))

   print(f"[INFO] Counting words...")
   frequency = Counter(merged_file_content)
   most_common = frequency.most_common(50) 

   frequency_df = pd.DataFrame(frequency.items(), columns=["word", "frequency"])

   # free memory
   del merged_file_content
   del frequency 


   # New columns for token_ids and token_len
   frequency_df["token_ids"]=None
   frequency_df["token_len"]=None

   
   # Create a multiprocessing pool
   with mp.Pool(NUM_PROCESSES) as pool:
       iterable_freq_df = pool.imap(tokenize_word, frequency_df.iterrows(), chunksize= 216)
       frequency_df = list(tqdm.tqdm(iterable_freq_df, total=len(frequency_df), desc="[INFO] Tokenization of words..."))  
       frequency_df = pd.DataFrame(frequency_df, columns=["word", "frequency", "token_ids", "token_len"])



   print(f"[INFO] Most common 50 words: {most_common}")
   print(f"[INFO] Total number of unique words: {len(frequency_df)}")
   print(f"[INFO] Total number of all words: {frequency_df['frequency'].sum()}")

   group_list = []
   
   for group_name, group_df in frequency_df.groupby("token_len"):
       
      # bundan büyük token grupları ile işimiz yok
      if int(group_name) > LIMIT_FOR_TOKEN_GROUP:
          continue

      
      # rasgele sampling yapmadan önce grupta belli bir frequency altındaki kelimeleri atmalıyız (çince arapça vs)
      group_df = group_df[group_df["frequency"] >= MIN_FREQ_FOR_WORDS]

      number_of_sample = len(group_df) if len(group_df) < MAX_WORD_LIMIT_FOR_TOKEN_GROUP else MAX_WORD_LIMIT_FOR_TOKEN_GROUP


      if RANDOM_SAMPLE:
          # her gruptan belirtilen sayıda rastgele sampling yapılacak
          group_list.append(group_df.sample(n=number_of_sample, random_state=13013))
      else:
          # en çok occur olan ilk "number_of_sample" tane kelimeyi al, rastgele yerine
          group_list.append(group_df.sort_values(by="frequency", ascending=False).iloc[:number_of_sample])
          

   del frequency_df


   print(f"[INFO] Saving {random_word_set_save_path} ... ")
   
   total_df = pd.concat(group_list)
   total_df.to_json(random_word_set_save_path, orient="records", lines=True, force_ascii=False)