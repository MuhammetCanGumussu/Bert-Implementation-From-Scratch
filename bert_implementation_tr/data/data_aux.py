
from dataclasses import dataclass, asdict
from typing import Tuple
import numpy as np
import pandas as pd
import ast


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