import os
import sys

import pandas as pd


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random_word_set_save_path = root_dir + "/random_word_set/random_word_set.json"


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