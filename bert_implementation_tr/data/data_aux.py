
import os
from dataclasses import dataclass, asdict
from typing import Tuple
import numpy as np
import pandas as pd

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast




@dataclass
class ModelInput:
    x: np.ndarray
    y: np.ndarray
    attention_mask: np.ndarray
    segment_ids: np.ndarray



@dataclass
class VisualizeModelInput:
    model_input: ModelInput
    show_ids: bool = False
    show_attention_and_segment: bool = False

@dataclass
class VisualizeInputAB:
    ab: pd.Series


@dataclass
class FillInput:
    mask_word_array: np.ndarray 
    replace_word_array: np.ndarray 
    identity_word_array: np.ndarray 

@dataclass
class OneSampleStat:
    isNext: int = 0
    number_of_mask_token_count: int = 0
    number_of_replace_token_count: int = 0
    number_of_identity_token_count: int = 0
    number_of_word: int = 0
    number_of_mask_word: int = 0
    number_of_replace_word: int = 0
    number_of_identity_word: int = 0
    number_of_not_accepted_word: int = 0


@dataclass
class Stat:
    # for stat.txt
    block_size: int
    total_number_of_token: int = 0
    total_number_of_sample: int = 0
    total_isNext_count: int = 0
    total_mask_token_count: int = 0
    total_replace_token_count: int = 0
    total_identity_token_count: int = 0
    total_number_of_word: int = 0
    total_number_of_mask_word: int = 0
    total_number_of_replace_word: int = 0
    total_number_of_identity_word: int = 0
    total_number_of_not_accepted_word: int = 0


    def update_stat_with_another_stat(self, other: OneSampleStat) -> None:

        self.total_mask_token_count += other.number_of_mask_token_count
        self.total_replace_token_count += other.number_of_replace_token_count
        self.total_identity_token_count += other.number_of_identity_token_count
        self.total_number_of_mask_word += other.number_of_mask_word
        self.total_number_of_replace_word += other.number_of_replace_word
        self.total_number_of_identity_word += other.number_of_identity_word
        self.total_number_of_not_accepted_word += other.number_of_not_accepted_word
        self.total_number_of_token += self.block_size
        self.total_number_of_sample += 1
        self.total_isNext_count += other.isNext  
        self.total_number_of_word += other.number_of_word


    def save_stat(self, save_path: str) -> None:
        with open(save_path, "w", encoding="utf-8") as f:
            for key, value in asdict(self).items():
                f.write(f"{key}: {value:_}\n")
       
    @staticmethod
    def parse_line(line:str) -> Tuple[str, int]:
        key, value = line.strip().split(": ")
        return key, int(value.replace('_', ''))  

    @classmethod
    def from_file(cls, load_path: str) -> "Stat":
        data = {}
        with open(load_path, "r", encoding="utf-8") as f:
            for line in f:
                key, value = Stat.parse_line(line)
                data[key] = value
        return cls(**data)
    


def get_merged_files():

    raw_dir = os.path.join(os.path.dirname(__file__), "raw")

    files = os.listdir(raw_dir)

    print(f"[INFO] Files in dir: {files}...")

    merged_file_content = ""

    for raw_file in files:
        with open(os.path.join(raw_dir, raw_file), encoding="utf-8") as raw:
            merged_file_content += (raw.read() + "\n")

    return merged_file_content



# TODO: path işlerini vs ayarla
SAVE_PATH = "C:/Users/user/Desktop/Bert Implementation Tr/bert_implementation_tr/tr_wordpiece_tokenizer_cased.json"

def get_tokenizer(tokenizer_path=SAVE_PATH, fast=True):
    

    if not os.path.exists(tokenizer_path):
        print(f"[INFO] there is no tokenizer file to wrap with fast tokenizer in {tokenizer_path} Please train tokenizer first...")
        import sys
        sys.exit(0)
    
    if fast:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file = tokenizer_path, # You can load from the tokenizer file, alternatively
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            clean_up_tokenization_spaces=True   # default olarak ta True ancak future warning ilerde False olacağını belirtti.
                                                # ilerde problem olmaması için (ve tabiki future warning almamak için) açıkca True yaptık
        )
             
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer