import os
from dataclasses import dataclass
from typing import List, Optional, Type

import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast

from data.data_aux import DataLoaderCustom




MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    dataloader: DataLoaderCustom,
                    step: int, 
                    best_val_loss: float,
                    postfix: int | str,
                    pretrain_config: dict | Type[dataclass],
                    mlflow_run_id: Optional[str] = None
                    ) -> None:
    """
    Save checkpoint dictionary
    """
    temp_dict = {
        'last_step': step,
        'best_val_loss': best_val_loss, 
        'model_config': model.config,
        'pretrain_config': pretrain_config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'last_dataloader_state': dataloader.get_current_state(),
        'mlflow_run_id': mlflow_run_id
    }
    
    torch.save(temp_dict, MODEL_DIR + f"/model_ckpts/BertForPretraining_{postfix}.pt")



def load_checkpoint(postfix: int | str) -> dict:
    """
    Return ckeckpoint dictionary
    """
    return torch.load(MODEL_DIR + f"/model_ckpts/BertForPretraining_{postfix}.pt")

    

def get_last_ckpt_idx():
    """
    Retrieves the index of the last checkpoint from the model checkpoints directory.

    This function checks the "model/model_ckpts" directory for checkpoint files, 
    and determines the highest numerical index among them, ignoring any files 
    with the "best" postfix. If no checkpoint files are found, an error is raised.

    Returns:
        int: The index of the last checkpoint.

    Raises:
        FileNotFoundError: If the "model/model_ckpts" directory does not exist.
        FileNotFoundError: If no checkpoint files are found in the directory.
    """
    
    if not os.path.exists("model/model_ckpts"):
        raise FileNotFoundError("model/model_ckpts dir is not exists, please train first")

    ckpt_files = os.listdir("model/model_ckpts")

    last_ckpt_idx = 0

    if len(ckpt_files) == 0:
        raise FileNotFoundError("model/model_ckpts dir is empty, please train first")

    for post_fix in list(map(lambda x: x.split("_")[-1].split(".")[0], ckpt_files)):

        if post_fix == "best":
            continue

        last_ckpt_idx = int(post_fix) if int(post_fix) > last_ckpt_idx else last_ckpt_idx

    return last_ckpt_idx



class FillMaskPipeline():
    """"default top-50 multinomial sampling if strategy is multionomial, else top-50 greedy sampling"""
    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, strategy: str = "multinomial"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = self.model.bert.config.max_position_embeddings
        self.strategy = strategy

    def __call__(self, text: List[str], do_print: bool = True) -> None:
        
        if self.strategy not in ["multinomial", "greedy"]:
            raise ValueError(f"unknown strategy: {self.strategy}")


        text_with_special_tokens = []
        for t in text:
            if "[MASK]" not in t:
                raise ValueError(f"mask token not found in text: {t}")
            text_with_special_tokens.append("[CLS] " + t + " [SEP]")


        encoding = self.tokenizer(text_with_special_tokens, padding="longest", return_tensors="pt").to(self.model.bert.embeddings.word_embeddings.weight.device)
        encoding["attention_mask"] = encoding["attention_mask"].to(torch.bool)
        if encoding["input_ids"].size(1) > self.max_length:
            raise ValueError(f"text too long: {encoding['input_ids'].size(1)} > {self.max_length}")
        
        mask_row_idxs, mask_col_idxs = torch.where(encoding["input_ids"] == self.tokenizer.mask_token_id)

        with torch.no_grad():
            self.model.eval()
            # B, T, V
            model_prediction_logits = self.model(**encoding).prediction_logits
            # B, V
            model_mask_logits = model_prediction_logits[mask_row_idxs, mask_col_idxs, :]    
            # B, V
            model_mask_probs = F.softmax(model_mask_logits, dim=-1)
            if self.strategy == "multinomial":
                # B, 50
                topk_probs, topk_indices = torch.topk(model_mask_probs, 50, dim=-1)
            else:
                # top 5 probs will automatically be selected in multinomial below (kind of cheaty but it works as greedy)
                # B, 5
                topk_probs, topk_indices = torch.topk(model_mask_probs, 5, dim=-1)
            # B, 5
            sampled_ids = torch.multinomial(topk_probs, 5) 
            # B, 5
            sampled_token_ids = torch.gather(topk_indices, -1, sampled_ids)
            sampled_token_probs = torch.gather(topk_probs, -1, sampled_ids)
            txt = ""
            for b_idx in range(topk_probs.size(0)):
                d = {"token_str":[], "score":[]}
                for i in range(5):
                    d["score"].append(format(sampled_token_probs[b_idx][i].item(), '.4f'))
                    d["token_str"].append(self.tokenizer.convert_ids_to_tokens(sampled_token_ids[b_idx][i].item()))
                txt += f"Text: {text[b_idx]} ----> Top 5 Predictions: {d}\n"
            print(txt) if do_print else None
            return txt




 
class IsNextPipeline():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = self.model.bert.config.max_position_embeddings

    def __call__(self, text: List[List[str]], do_print: bool = True) -> None:
        text_with_special_tokens = []
        for l_str in text:
            textA = l_str[0]
            textB = l_str[1]
            text_with_special_tokens.append("[CLS] " + textA + " [SEP] " + textB + " [SEP]")
        
            
        encoding = self.tokenizer(text_with_special_tokens, padding="longest", return_tensors="pt").to(self.model.bert.embeddings.word_embeddings.weight.device)
        encoding["attention_mask"] = encoding["attention_mask"].to(torch.bool)

        if encoding["input_ids"].size(1) > self.max_length:
            raise ValueError(f"text too long: {encoding['input_ids'].size(1)} > {self.max_length}")
        
        token_type_ids = torch.ones_like(encoding["input_ids"])
        first_sep_indices = torch.argmax(((encoding["input_ids"] == self.tokenizer.sep_token_id)).to(torch.int), dim=1)

        for i in range(encoding["input_ids"].size(0)):
            token_type_ids[i, :first_sep_indices[i]] = 0
        
        encoding["token_type_ids"] = token_type_ids

        with torch.no_grad():
            self.model.eval()

            # B, 2
            model_seq_logits = self.model(**encoding).seq_relationship_logits
            # B, 2
            nsp_probs = F.softmax(model_seq_logits, dim=-1) 
            txt = ""
            for b_idx in range(model_seq_logits.size(0)):
                txt += f"Text: {text[b_idx]} Predictions ----> { f'isNext: {nsp_probs[b_idx][0].item():.3f}, notNext: {nsp_probs[b_idx][1].item():.3f}' }\n" 
            print(txt) if do_print else None
            return txt

