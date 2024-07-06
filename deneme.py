from nltk.tokenize import sent_tokenize
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial
# import pandas as pd

import json
import os
import sys
import logging
import time
import random
import tqdm
import re
from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer

SAVE_PATH = "tr_wordpiece_tokenizer_cased.json"

def count_words(input_string):
    word_pattern = r'\b\w+\b'

    # Split the string into words
    words = re.findall(word_pattern, input_string)
    
    # Initialize an empty dictionary to store word counts
    word_count = {}
    
    # Count occurrences of each word
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    # Return the dictionary containing word counts
    return word_count


if __name__ == '__main__':

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=SAVE_PATH)

    # tokenizer = Tokenizer.from_file(SAVE_PATH)
    # print(type(tokenizer))
    # print(tokenizer.is_fast)

    deneme_str = "Tarra White (d. <---> II. <---> Murat devrinde Osmanlı topraklarına katılan Çal, Karahisar-ı Sahib sancağına bağlandığı 1849'a kadar Kütahya sancağına bağlı kaldı. <---> notNext"

    deneme_str = deneme_str.replace(" <---> ", " ").replace(" notNext", "").replace(" isNext", "")


    print(deneme_str)
    
    print("-" * 10)
    
    count_words_dict = count_words(deneme_str)
    unik_words = list(count_words_dict.keys()) # list[str, str, ...]
    
    print(unik_words)
    print(list(map(len, tokenizer(unik_words)["input_ids"])))
    






    

