
import os
from dataclasses import dataclass, asdict
from typing import Tuple
import numpy as np
import pandas as pd
import torch

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast



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

# geçici silinecek (data.py'da labels'da cls tokeni full pad olarak verildiğinde buna ve alttaki olaya gerek kalmayacak)
TEMP_PAD_TOKEN_ID = get_tokenizer().convert_tokens_to_ids("[PAD]")

@dataclass
class ModelInput:
    input_ids: np.ndarray | torch.Tensor        # bakılacak: isimler model forward parametreleri ile uyumlu hale getirilecek
    labels: np.ndarray | torch.Tensor
    attention_mask: np.ndarray | torch.Tensor
    token_type_ids: np.ndarray | torch.Tensor
    next_sentence_label: np.ndarray | torch.Tensor = None
    
    @classmethod
    def from_numpy_to_ModelInput(cls, np_array: np.ndarray, block_size: int, device: torch.device | str) -> "ModelInput":
        return cls(
            # sıraya dikkat : x -> y -> segment_ids -> attention_mask
            # np array genişliği: x + y + segment_ids + attention_mask --> (BLOCK_SIZE * 4)
            input_ids = torch.tensor(np_array[:, :block_size], dtype=torch.long, device=device),
            labels = torch.tensor(np_array[:, block_size:2 * block_size], dtype=torch.long, device=device),
            token_type_ids = torch.tensor(np_array[:, 2 * block_size:3 * block_size], dtype=torch.long, device=device),
            attention_mask= torch.tensor(np_array[:, 3 * block_size:], dtype=torch.bool, device=device),
            next_sentence_label = torch.tensor(np_array[:, block_size], dtype=torch.long, device=device)
        )
    
    @staticmethod
    def from_numpy_to_tensors_dict(np_array: np.ndarray, block_size: int) -> dict:
        """
        ModelInput return edip daha sonra asdict ile dict'e çevirmek eğitim sırasında gereksiz maliyet olabilir (from_numpy_to_tensors).
        Bundan dolayı direkt dict return eden bu static method kullanılmalı (from_numpy_to_tensors_dict).
        """

        # geçici çözüm, data.py'da halledilecek
        # data.py'da : en sağa is next eklenecek ayrı olarak, y'de pad token yerine ignore idx kullanacak yani -100
        labels = torch.tensor(np_array[:, block_size:2 * block_size], dtype=torch.long)
        labels[:, 0] = TEMP_PAD_TOKEN_ID

        # "..." token'ı kaynaklı (vocab'ı 32001 yaptı) geçici çözüm:
        # 32000 id'sini gördüğümüz elemanları "." token'ı yapacağım
        # "..." -> "." 
        labels[labels == 32000] = 18
        input_ids = torch.tensor(np_array[:, :block_size], dtype=torch.long)
        input_ids[input_ids == 32000] = 18

        return dict(
            # sıraya dikkat : x -> y -> segment_ids -> attention_mask
            # np array genişliği: x + y + segment_ids + attention_mask --> (BLOCK_SIZE * 4)
            input_ids = input_ids,
            labels = labels,
            token_type_ids = torch.tensor(np_array[:, 2 * block_size:3 * block_size], dtype=torch.long),
            attention_mask= torch.tensor(np_array[:, 3 * block_size:], dtype=torch.bool),
            next_sentence_label = torch.tensor(np_array[:, block_size], dtype=torch.long)
        )


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






def _visualize_ab(sample: VisualizeInputAB):
        tokenizer = get_tokenizer(SAVE_PATH, fast=True)
        print(f"A: {sample.ab['A_token_ids']}")
        print(f"B: {sample.ab['B_token_ids']}")
        print(f"A_decoded: {tokenizer.decode(sample.ab['A_token_ids'])}")
        print(f"B_decoded: {tokenizer.decode(sample.ab['B_token_ids'])}")
        print(f"len_of_A: {len(sample.ab['A_token_ids'])}")
        print(f"len_of_B: {len(sample.ab['B_token_ids'])}")
        print(f"A_word_ids: {sample.ab['A_word_ids']}")
        print(f"B_word_ids: {sample.ab['B_word_ids']}")
        print(f"len_of_A_word_ids: {len(sample.ab['A_word_ids'])}")
        print(f"len_of_B_word_ids: {len(sample.ab['B_word_ids'])}")
        print(f"sum_of_AB_tokens: {len(sample.ab['B_word_ids']) + len(sample.ab['A_word_ids'])}")
        print(f"isNext: {sample.ab['isNext']}")
        
        print("---------------\n")



def _visualize_model_input(sample: VisualizeModelInput) -> None:

    show_attention_and_segment = sample.show_attention_and_segment
    show_ids = sample.show_ids
    sample = sample.model_input

    tokenizer = get_tokenizer(SAVE_PATH, fast=True)

    print(f"x_decoded: {tokenizer.decode(sample.input_ids)}")
    print(f"y_decoded: {tokenizer.decode(sample.labels)}\n")

    # sep_idx = np.where(sample.x == SEP_TOKEN_ID)[0][0]
    # print(f"A_x: {tokenizer.decode(sample.x[:sep_idx], skip_special_tokens=True)}")
    # print(f"B_x: {tokenizer.decode(sample.x[sep_idx:], skip_special_tokens=True)}\n")
    
    
    if show_attention_and_segment == True:
        print(f"attention_mask: {sample.attention_mask}")
        print(f"segment_ids: {sample.token_type_ids}\n")

    mask_of_filled = (sample.labels != tokenizer.convert_tokens_to_ids("[PAD]"))
    x_filled = sample.input_ids[mask_of_filled]

    # tokenların hizalı olabilmesi için pd.Dataframe kullanacağım
    print_df = pd.DataFrame({
        "X_FILLED": ["----------"] + [tokenizer.decode(token_id) for token_id in sample.input_ids[mask_of_filled]],
        "Y_FILLED": ["----------"] + [tokenizer.decode(token_id) for token_id in sample.labels[mask_of_filled]],
    })
    print(print_df.to_string(index=False))

    print(f"\nisNext: {sample.labels[0] == 1}")
    print(f"number_of_mask_token: {print_df['X_FILLED'].value_counts()['[MASK]']}")
    print(f"number_of_filled_token: {len(print_df) - 1}\n")

    print(f"\nlen_of_x: {len(sample.input_ids)}")
    print(f"len_of_y: {len(sample.labels)}\n")
    print(f"")

    if show_ids == True:
        print(f"x_ids: {sample.input_ids}")
        print(f"y_ids: {sample.labels}\n")
    
    print("---------------\n")



def visualize_sample(sample: VisualizeInputAB | VisualizeModelInput):
    """Be careful, this function works for just one sample"""
    if isinstance(sample, VisualizeModelInput):
        _visualize_model_input(sample)
    elif isinstance(sample, VisualizeInputAB):  
        _visualize_ab(sample)
    else:
        raise TypeError("sample must be VisualizeInputAB or VisualizeModelInput")



def get_last_shard_idx(shards_dir:str) -> int:
    """-1 return ederse dizinde hiç shard dosyası yok demektir"""
    files = os.listdir(shards_dir)
    
    last_shard_idx = -1
    for file in files:

        # eğer xy_shards dizininde isek, extra stat.txt dosyası olacak, atla
        if file == "stat.txt":
            continue

        # dizin ile aynı prefix'e sahip olmayan dosyaları atla (zaten olmaması gerek ancak her ihtimale karşı)
        if not file.startswith("xy_shard_"):
            continue

        last_shard_idx += 1
    return last_shard_idx 


def save_xy_shard(placeholder_array, shard_idx, block_size) -> None:
    np.save(f"xy_shards_{block_size}/xy_shard_{shard_idx}.npy", placeholder_array)
                        

def load_xy_shard(shard_idx, block_size=256) -> np.ndarray:
    if (shard_idx < 0) or (shard_idx > get_last_shard_idx(f"bert_implementation_tr/data/xy_shards_{block_size}")):
        raise IndexError(f"shard idx must be >= 0 and <= {get_last_shard_idx(f'bert_implementation_tr/data/xy_shards_{block_size}')}, shard_idx you gave was: {shard_idx}")
    # print(f"loading xy_shard_{shard_idx}.npy")
    return np.load(f"bert_implementation_tr/data/xy_shards_{block_size}/xy_shard_{shard_idx}.npy")





class DataLoaderCustom:
    def __init__(self, batch_size: int,
                 block_size: int,
                 device: str = "cpu",
                 verbose: bool = False,
                 split: str = "train") -> None:
        self.split = split
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size
        self.data_dir = f"bert_implementation_tr/data/xy_shards_{self.block_size}"

        assert split in ["train", "val"], f"unknown split: {self.split}"

        last_shard_idx = get_last_shard_idx(self.data_dir)
        assert last_shard_idx != -1 , f"no shards found in {self.data_dir}"


        self.last_shard_id = (last_shard_idx - 1) if self.split == "train" else last_shard_idx
        self.current_shard_id = 0 if self.split == "train" else last_shard_idx
        self.current_position_in_shard = 0

        # print(f"\nsplit: {self.split}")
        # print(f"data shards directory: {self.data_dir}\n")

        self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size), 
                                        block_size = self.block_size)

        
        self.stat = Stat.from_file(f"{self.data_dir}/stat.txt")
        isNext_ratio = self.stat.total_isNext_count / self.stat.total_number_of_sample
        self.class_weights = torch.tensor([isNext_ratio, 1 - isNext_ratio], dtype=torch.float32, device=self.device)
        
        # bakılacak bunu kaldırıcam galiba, illa bu statları istersem pretrain dosyasında kendim yaparım kontrolümde (train_loader.stat ile örn)
        if verbose and split == "train":
            # bu değerler vs approx'tur, son shard'ı val'a ayırdık

            # some stats
            print(f"block size: {self.block_size}")
            print(f"batch size: {self.batch_size}")
            print(f"total number of shards: {self.last_shard_id + 1}")
            print(f"total number of tokens: {self.stat.total_number_of_token}")
            print(f"total number of samples: {self.stat.total_number_of_sample}")
            print("-------------------------------------------------------------------------")
            print(f"1 batch: {self.block_size * self.batch_size} tokens")
            print(f"1 epoch: {self.stat.total_number_of_token // (self.block_size * self.batch_size)} batches")
            print(f"1 epoch: {self.stat.total_number_of_token} tokens")
            print(f"1 shard: ~{self.stat.total_number_of_token // (self.last_shard_id + 1)} tokens")
            print(f"1 shard: ~{(self.stat.total_number_of_token // (self.last_shard_id + 1)) // (self.block_size * self.batch_size)} batches")
            print("-------------------------------------------------------------------------\n")


    def reset(self):
        if self.split == "train":
            self.current_shard_id = 0
            self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size), 
                                        block_size = self.block_size)

        else:
            self.current_shard_id = self.last_shard_id  # same shard, last shard
            # no need to load the same shard again, because it's already loaded in the constructor

        # reset position
        self.current_position_in_shard = 0
        
        

    def next_batch(self):
        if (self.current_position_in_shard + self.batch_size) >= len(self.current_shard["input_ids"]):
            # bakılacak: circulasyona izin vermeyeceksek o zaman son kalanları (varsa) verelim (vermeyelim, 5 sample için kod iyice çirkinleşmesin)
            if self.split == "val":
                # bakılacak: düz resetlemeye ek olarak bununiçinde reset yapackasın (düz reset: her val sürecinde resetleme yapılır aynı sample'lar gözüksün)
                # (bu yeni ek reset'de istenilen max_eval sayısı çok büyük olursa durumunda val'da artık başka yeni sample kalmadığında val sürecinden çıkılacak, aslında bu reset değil evet)
                return None
            
            model_input_temp = {k: v[self.current_position_in_shard:].to(self.device) for k, v in self.current_shard.items()}
            temp_len = len(model_input_temp["input_ids"])
                
            # for making modulus work we need to add 1 (first shard id is 0, 0 % any_number == 0, because of this we added 1 to each operand)
            self.current_shard_id = 0 if (self.current_shard_id + 1) % (self.last_shard_id + 1) == 0 else (self.current_shard_id + 1)
            
            self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size), 
                                        block_size = self.block_size)
            
            model_input = {k: torch.cat([model_input_temp[k], v[:self.batch_size - temp_len].to(self.device)]) for k, v in self.current_shard.items()}
            model_input["class_weights"] = self.class_weights
            self.current_position_in_shard = self.batch_size - temp_len

            return model_input
        
        model_input = {k: v[self.current_position_in_shard : self.current_position_in_shard + self.batch_size].to(self.device) for k, v in self.current_shard.items()}
        model_input["class_weights"] = self.class_weights
        self.current_position_in_shard += self.batch_size

        return model_input
    

    def get_current_state(self) -> dict:
        return {
            "last_shard_id": self.current_shard_id,
            "last_position": self.current_position_in_shard
        }

    
    def load_state_dict(self, state_dict: dict):
        self.current_shard_id = state_dict["last_shard_id"]
        self.current_position_in_shard = state_dict["last_position"]
        self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size), 
                                        block_size = self.block_size)