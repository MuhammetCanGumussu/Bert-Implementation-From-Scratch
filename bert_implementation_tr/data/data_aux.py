
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ModelInput:
    x: np.ndarray
    y: np.ndarray
    attention_mask: np.ndarray
    segment_ids: np.ndarray

@dataclass
class VisualizeModelInput:
    sample: ModelInput
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

class Stat:
    # for stat.txt
    total_token_count: int = 0
    total_number_sample: int = 0
    mask_token_count: int = 0
    replace_token_count: int = 0
    identity_token_count: int = 0

    def update_stat_with_one_sample(self, filled_word_array: FillInput, block_size: int):
        self.mask_token_count += len(filled_word_array.mask_word_array)
        self.replace_token_count += len(filled_word_array.replace_word_array)
        self.identity_token_count += len(filled_word_array.identity_word_array)
        self.total_token_count += block_size
        self.total_number_sample += 1
    
    def save_stat(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"Total token count: {self.total_token_count}\n")
            f.write(f"Total number sample: {self.total_number_sample}\n")
            f.write(f"Mask token count: {self.mask_token_count}\n")
            f.write(f"Replace token count: {self.replace_token_count}\n")
            f.write(f"Identity token count: {self.identity_token_count}\n")
    
    @classmethod
    def from_file(cls, load_path: str):
        with open(load_path, "r", encoding="utf-8") as f:
            # dict'e benzer txt yazdığımızdan, eval ile bu txt dosyası dict'e çevrilir (ve tabi unpack)
            return cls(**eval(f.read()))

    
            







