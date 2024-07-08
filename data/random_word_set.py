""" creates unique_words.json, random_word_set.json, token_len_stat.png"""

from transformers import PreTrainedTokenizerFast
import pandas as pd

import json
import os
import tqdm
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_WORDS_SET_DIR = "random_words_set"
SAVE_PATH = "../tr_wordpiece_tokenizer_cased.json"


def count_words(input_string):
    word_pattern = r'\b[A-Za-z\']+\b'

    print("[INFO] all text gonna split into words...")
    # Split the string into words
    words = re.findall(word_pattern, input_string)
    
    pbar = tqdm.tqdm(words, desc="Counting words")

    print("[INFO] counting words...")
    counter = Counter()

    for word in pbar:
        counter[word] += 1

    return counter


if __name__ == '__main__':

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=SAVE_PATH)

 
    if not os.path.exists(RANDOM_WORDS_SET_DIR + "/unique_words.json"):

        from data import load_ab_string
        ab_strings = load_ab_string("ab_string.raw") # list[str, str, ...]

        print("[INFO] ab_string.raw is being processed for word count...")
        # just string
        all_text = "".join(ab_strings).replace(" <---> ", " ").replace(" notNext", "").replace(" isNext", "")

        counter_words = count_words(all_text)

        #print(f"[INFO] most common 5 word: {counter_words.most_common(10)} ")

        word_and_count_tuples = list(counter_words.items())

        df_word_count = pd.DataFrame(word_and_count_tuples, columns=["word", "count"])

        # [word, count, tokens, token_len]
        df_word_count["tokens"] = df_word_count["word"].map(lambda x: (tokenizer(x).tokens()))
        df_word_count["token_len"] = df_word_count["tokens"].map(lambda x: len(x))

        print("[INFO] Saving word count dataframe to unique_words.json...")

        # save dataframe
        df_word_count.to_json("unique_words.json", orient="records", lines=True)
    else:
        print("[INFO] unique_words.json is already exists. loading dataframe...")

        df_word_count = pd.read_json(RANDOM_WORDS_SET_DIR + "/unique_words.json", orient="records", lines=True)

    # group by token length and then plot the stat
    if not os.path.exists(RANDOM_WORDS_SET_DIR + "/token_len_stat.png"):

        print("[INFO] Plotting token length frequency...")

        token_len_stat = df_word_count.groupby("token_len")["count"].sum()

        # Plotting
        plt.figure(figsize=(8, 6))  # Optional: Set figure size
        plt.scatter(range(1, len(token_len_stat) + 1), token_len_stat.values, color='skyblue')
        # plt.bar(token_len_stat.values, range(1, len(token_len_stat) + 1), color='skyblue')
        plt.xlabel('token lengths')
        plt.ylabel('count-logscale')
        plt.yscale('log')
        plt.title('Token Length Frequency')
        plt.xticks(range(1, len(token_len_stat) + 1), token_len_stat.index, rotation=90)
        plt.tight_layout()
  
        # Save plot to a file or display in console
        plt.savefig(RANDOM_WORDS_SET_DIR + '/token_len_freq.png')  # Save plot as PNG file
        plt.show()  # Display plot in console
    

    # if token len of word is bigger than 5 (TRESH_TOKEN_LEN), 
    # this word will not be used by bert model training so we can remove it dataframe
    TRESH_TOKEN_LEN = 5

    if not os.path.exists(RANDOM_WORDS_SET_DIR + "/random_words_set.json"):

        print("[INFO] Random words set is being created...")

        df_word_count = df_word_count[df_word_count['token_len'] <= TRESH_TOKEN_LEN] 
        group_by_token_len = df_word_count.groupby("token_len")
        dict_to_save = {"lengths_of_groups": {}} 
                                                 
                                                 
        # {     "lengths_of_groups": {grup1:len1, grup2:len2, ...},
        #             "token_len_1": [[token1], [token2], ...],
        #             "token_len_2": [[token1], [token2], ...],  ...            }


        # group_df["tokens"] -> list[list[str], list[str]] ->  [[token1, token2], [token1], [token1, token2, token3]]
        for token_len, group_df in group_by_token_len:
            dict_to_save["lengths_of_groups"][f"group_{token_len}_len"] = len(group_df)
            dict_to_save[f"token_len_{token_len}"] = group_df["tokens"].to_list()

        json_data = json.dumps(dict_to_save, indent=4) 

        with open(RANDOM_WORDS_SET_DIR + "/random_words_set.json", 'w', encoding="utf-8") as json_file:
            json_file.write(json_data)


    # lets load random words set and check
    with open(RANDOM_WORDS_SET_DIR + "/random_words_set.json", 'r', encoding="utf-8") as json_file:
        random_words_dict = json.load(json_file)
        

    print(random_words_dict.keys())
    print("\n-------------------------------\n")
    print(random_words_dict["lengths_of_groups"].items())
    print("\n-------------------------------\n")
    print(random_words_dict["token_len_1"][0][:])
    print(type(random_words_dict["token_len_1"]))
    print(len(random_words_dict["token_len_1"]))
    print("\n-------------------------------\n")
    print(random_words_dict["token_len_2"][0][:])
    print(type(random_words_dict["token_len_2"]))
    print(len(random_words_dict["token_len_2"]))
    print("\n-------------------------------\n")
    print(random_words_dict["token_len_3"][0][:])
    print(type(random_words_dict["token_len_3"]))
    print(len(random_words_dict["token_len_3"]))


    
