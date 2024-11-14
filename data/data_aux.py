"""CAUTION: Some docstrings are generated by AI (Codeium), so they may not always make sense."""

import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass, asdict

from tokenizer.tokenizer_aux import get_tokenizer


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelInput:
    input_ids: np.ndarray | torch.Tensor        
    attention_mask: np.ndarray | torch.Tensor
    token_type_ids: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor = None
    next_sentence_label: np.ndarray | torch.Tensor = None
    
    @classmethod
    def from_numpy_to_ModelInput(cls, np_array: np.ndarray, block_size: int, device: torch.device | str) -> "ModelInput":
        """
        Converts a numpy array into a ModelInput object suitable for model input.

        Args:
            np_array (np.ndarray): A numpy array where each row contains concatenated sequences
                                  representing input_ids, labels, token_type_ids, attention_mask,
                                  and next_sentence_label.
            block_size (int): The size of each individual sequence block within the array.
            device (torch.device | str): The device where ModelInput tensors will be allocated.

        Returns:
            ModelInput: A ModelInput object containing the following PyTorch tensors:
                - 'input_ids': Tensor of input IDs.
                - 'labels': Tensor of labels.
                - 'token_type_ids': Tensor of token type IDs.
                - 'attention_mask': Tensor representing the attention mask.
                - 'next_sentence_label': Tensor for the next sentence prediction label.
        """
        return cls(
            # sıraya dikkat : x -> y -> segment_ids -> attention_mask
            # np array genişliği: x + y + segment_ids + attention_mask --> (BLOCK_SIZE * 4)
            input_ids = torch.tensor(np_array[:, :block_size], dtype=torch.long, device=device),
            labels = torch.tensor(np_array[:, block_size:2 * block_size], dtype=torch.long, device=device),
            token_type_ids = torch.tensor(np_array[:, 2 * block_size:3 * block_size], dtype=torch.long, device=device),
            attention_mask= torch.tensor(np_array[:, 3 * block_size:], dtype=torch.bool, device=device),
            next_sentence_label = torch.tensor(np_array[:, -1], dtype=torch.long, device=device)
        )
    
    @staticmethod
    def from_numpy_to_tensors_dict(np_array: np.ndarray, block_size: int) -> dict:
        """
        Converts a numpy array into a dictionary of PyTorch tensors suitable for model input.
    
        Args:
            np_array (np.ndarray): A numpy array where each row contains concatenated sequences
                                   representing input_ids, labels, token_type_ids, attention_mask,
                                   and next_sentence_label.
            block_size (int): The size of each individual sequence block within the array.
    
        Returns:
            dict: A dictionary containing the following keys and corresponding PyTorch tensors:
                - 'input_ids': Tensor of input IDs.
                - 'labels': Tensor of labels.
                - 'token_type_ids': Tensor of token type IDs.
                - 'attention_mask': Tensor representing the attention mask.
                - 'next_sentence_label': Tensor for the next sentence prediction label.
        """
        return dict(
            input_ids =  torch.tensor(np_array[:, :block_size], dtype=torch.long),
            labels = torch.tensor(np_array[:, block_size:2 * block_size], dtype=torch.long),
            token_type_ids = torch.tensor(np_array[:, 2 * block_size:3 * block_size], dtype=torch.long),
            attention_mask= torch.tensor(np_array[:, 3 * block_size:-1], dtype=torch.bool),
            next_sentence_label = torch.tensor(np_array[:, -1], dtype=torch.long)
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

     """
     Update the Stat object with values from OneSampleStat instance.
 
     This method updates various statistical counters by adding values from the
     provided OneSampleStat object to the corresponding attributes of 
     the current Stat object. It increments token, sample, and word counts, 
     including specific counts for masked, replaced, and identity tokens/words.
 
     Parameters
     ----------
     other : OneSampleStat
         An instance of OneSampleStat containing statistical data to be added
         to the current statistics.
     """
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


    def save_stat(self, save_path: str, cfg: Optional[dataclass]) -> None:
        """Save the statistics to a file.

        Parameters
        ----------
        save_path : str
            Path to the file to save the statistics to
        """
       
        with open(save_path, "w", encoding="utf-8") as f:
            for key, value in asdict(self).items():
                f.write(f"{key}: {value:_}\n")

            if cfg is not None:
                cfg = asdict(cfg)
                f.write("\nCONFIGS:\n")
                f.write("-----------------------------------------------------------------------\n")
                
                for k, v in cfg.items():
                    f.write(f"{k}: {v}\n")
       
    @staticmethod
    def parse_line(line:str) -> Tuple[str, int]:
        """
        Parse a line of a stat file into a tuple of key and value

        Parameters
        ----------
        line : str
            The line to parse

        Returns
        -------
        Tuple[str, int]
            A tuple of key and value
        """

        key, value = line.strip().split(": ")
        return key, int(value.replace('_', ''))  

    @classmethod
    def from_file(cls, load_path: str) -> "Stat":
        """
        Create a Stat instance from a file

        Parameters
        ----------
        load_path : str
            Path to the file containing the statistics

        Returns
        -------
        Stat
            A new Stat instance with the loaded statistics
        """
        data = {}
        with open(load_path, "r", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    break
                key, value = Stat.parse_line(line)
                data[key] = value
        return cls(**data)
    





def get_merged_files() -> str:

    """
    Merges all files in the raw directory into one string and returns it.

    The files are read in the order they are listed in the directory and are concatenated together with a newline character in between.

    Returns:
        str: The merged content of all the files in the raw directory.
    """
    raw_dir = root_dir + "/data/raw"

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory {raw_dir} does not exist.")

    files = os.listdir(raw_dir)

    if len(files) == 0:
        raise FileNotFoundError(f"Raw directory {raw_dir} is empty.")

    print(f"[INFO] Files in raw_dir: {files}...")

    merged_file_content = ""

    for raw_file in files:
        with open(os.path.join(raw_dir, raw_file), encoding="utf-8") as raw:
            merged_file_content += (raw.read() + "\n")

    return merged_file_content






def _visualize_ab(sample: VisualizeInputAB):
    """
    Visualizes a sample of VisualizeInputAB.

    Prints the decoded inputs A and B, the length of A and B, the word ids of A and B, the length of word ids of A and B, the sum of the length of A and B, and the isNext label.

    Args:
        sample (VisualizeInputAB): The sample to visualize.
    """
    tokenizer = get_tokenizer()
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

    """
    Visualizes a sample of ModelInput.

    Shows the decoded inputs and labels, attention mask and segment ids if show_attention_and_segment is True, and the ids of inputs and labels if show_ids is True.

    Args:
        sample (VisualizeModelInput): The sample to visualize.
    """
    show_attention_and_segment = sample.show_attention_and_segment
    show_ids = sample.show_ids
    sample = sample.model_input

    tokenizer = get_tokenizer()

    print(f"x_decoded: {tokenizer.decode(sample.input_ids)}")
    print(f"y_decoded: {tokenizer.decode(sample.labels)}\n")
    
    
    if show_attention_and_segment == True:
        print(f"attention_mask: {sample.attention_mask}")
        print(f"segment_ids: {sample.token_type_ids}\n")

    mask_of_filled = (sample.labels != tokenizer.convert_tokens_to_ids("[PAD]"))
    x_filled = sample.input_ids[mask_of_filled]


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
    
    """
    Visualizes the given sample.

    This function determines the type of the given sample and calls the appropriate
    visualization function.

    Args:
        sample (VisualizeInputAB | VisualizeModelInput): The sample to visualize.

    Raises:
        TypeError: If the given sample is not of type VisualizeInputAB or VisualizeModelInput.
    """
    if isinstance(sample, VisualizeModelInput):
        _visualize_model_input(sample)
    elif isinstance(sample, VisualizeInputAB):  
        _visualize_ab(sample)
    else:
        raise TypeError("sample must be VisualizeInputAB or VisualizeModelInput")



def get_last_shard_idx(shards_dir:str) -> int:
    
    """
    Determines the index of the last shard in the specified directory.

    This function iterates over the files in the given directory, skipping
    the "stat.txt" file, and increments the index for each file encountered.

    Args:
        shards_dir (str): The path to the directory containing shard files.

    Returns:
        int: The index of the last shard found in the directory, or -1 if
        no valid shard files were found.
    """
    files = os.listdir(shards_dir)
    
    last_shard_idx = -1
    for file in files:

        if file == "stat.txt":
            continue

        last_shard_idx += 1
    return last_shard_idx 



                        

def load_xy_shard(shard_idx, block_size=256, tokenizer_type="custom") -> np.ndarray:
    #DİR LER
    """
    Loads a shard of preprocessed data as a numpy array from a specified directory.

    Args:
        shard_idx (int): The index of the shard to load. Must be within the range of available shards.
        block_size (int, optional): The block size used for data preprocessing. Defaults to 256.
        tokenizer_type (str, optional): The type of tokenizer used for preprocessing. Defaults to "custom".

    Returns:
        np.ndarray: The loaded shard data as a numpy array.

    Raises:
        IndexError: If the `shard_idx` is out of the valid range of available shards.
    """
    if (shard_idx < 0) or (shard_idx > get_last_shard_idx(root_dir + f"/data/xy_shards_{tokenizer_type}_{block_size}")):
        raise IndexError(f"shard idx must be >= 0 and <= {get_last_shard_idx(root_dir + f'/data/xy_shards_{tokenizer_type}_{block_size}')}, shard_idx you gave was: {shard_idx}")
    if not os.path.exists(root_dir + f"/data/xy_shards_{tokenizer_type}_{block_size}/xy_shard_{shard_idx}.npy"):
        raise FileNotFoundError(f"xy_shard_{shard_idx}.npy not found in {root_dir + f'/data/xy_shards_{tokenizer_type}_{block_size}/xy_shard_{shard_idx}.npy'}... Please prepare data properly...")
    return np.load(root_dir + f"/data/xy_shards_{tokenizer_type}_{block_size}/xy_shard_{shard_idx}.npy")





class DataLoaderCustom:
    def __init__(self, batch_size: int,
                 block_size: int,
                 device: str = "cpu",
                 verbose: bool = True,
                 tokenizer_type: str = "custom",
                 split: str = "train") -> None:
        """
        Creates a DataLoaderCustom instance

        Args:
            batch_size (int): the batch size to use
            block_size (int): the block size to use
            device (str, optional): the device to use. Defaults to "cpu".
            verbose (bool, optional): whether to print some information. Defaults to True.
            tokenizer_type (str, optional): the tokenizer type to use. Defaults to "custom".
            split (str, optional): the split to use. Defaults to "train".
        """
        self.split = split
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size
        self.tokenizer_type = tokenizer_type
        self.data_dir = root_dir + f"/data/xy_shards_{self.tokenizer_type}_{self.block_size}"

        assert split in ["train", "val"], f"unknown split: {self.split}"

        last_shard_idx = get_last_shard_idx(self.data_dir)
        assert last_shard_idx != -1 , f"no shards found in {self.data_dir}"


        self.last_shard_id = (last_shard_idx - 1) if self.split == "train" else last_shard_idx
        self.current_shard_id = 0 if self.split == "train" else last_shard_idx
        self.current_position_in_shard = 0


        self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size, tokenizer_type=self.tokenizer_type), 
                                        block_size = self.block_size)

        
        self.stat = Stat.from_file(f"{self.data_dir}/stat.txt")
        isNext_ratio = self.stat.total_isNext_count / self.stat.total_number_of_sample
        self.class_weights = torch.tensor([isNext_ratio, 1 - isNext_ratio], dtype=torch.float32, device=self.device)
        
        if verbose and split == "train":
            print("class weights 0:notNext, 1:isNext -> ", self.class_weights)



        

    def reset(self):
        """
        Resets the DataLoaderCustom to its initial state.

        If the split is "train", it resets the current shard id to 0 and loads the first shard.
        If the split is "val", nothing is changed actually (self.current_shard_id and self.last_shard_id are the same)

        Resets the current position in shard to 0.

        This method is useful when you want to iterate over the data multiple times.
        """

        if self.split == "train":
            self.current_shard_id = 0
            self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size, tokenizer_type=self.tokenizer_type), 
                                        block_size = self.block_size)

        else:
            self.current_shard_id = self.last_shard_id  # same shard, last shard
            # no need to load the same shard again, because it's already loaded in the constructor

        # reset position
        self.current_position_in_shard = 0
        
        

    def next_batch(self):
        """
        Returns the next batch of data. If the current shard is exhausted, 
        load the next shard and return the remaining samples from the current shard
        plus the first samples from the next shard. If the split is "val", 
        return None when the current shard is exhausted.

        Returns:
            dict: A dictionary of model inputs, including "input_ids", "attention_mask", 
            "labels", "next_sentence_label", "token_type_ids" and "class_weights". If the split is "val" and the current shard 
            is exhausted, returns None.
        """
        if (self.current_position_in_shard + self.batch_size) >= len(self.current_shard["input_ids"]):
            if self.split == "val":
                return None
            
            model_input_temp = {k: v[self.current_position_in_shard:].to(self.device) for k, v in self.current_shard.items()}
            temp_len = len(model_input_temp["input_ids"])
                
            # for making modulus work we need to add 1 (first shard id is 0, 0 % any_number == 0)
            self.current_shard_id = 0 if (self.current_shard_id + 1) % (self.last_shard_id + 1) == 0 else (self.current_shard_id + 1)
            
            self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size, tokenizer_type=self.tokenizer_type), 
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
        """
        Returns the current state of the data loader as a dictionary.

        The state consists of two values: "last_shard_id" and "last_position". The
        "last_shard_id" is the index of the last shard that was used to fetch a batch.
        The "last_position" is the position of the last fetched batch in the shard.

        Returns:
            dict: A dictionary containing the state of the data loader.
        """

        return {
            "last_shard_id": self.current_shard_id,
            "last_position": self.current_position_in_shard
        }

    
    def load_state_dict(self, state_dict: dict):
        """
        Loads the state of the data loader from a dictionary.

        Args:
            state_dict (dict): A dictionary containing the state of the data loader.
                It should have two keys: "last_shard_id" and "last_position".
        """
        
        self.current_shard_id = state_dict["last_shard_id"]
        self.current_position_in_shard = state_dict["last_position"]
        self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size, tokenizer_type=self.tokenizer_type), 
                                        block_size = self.block_size)