"""some docstrings are generated by ai (codeium)"""
import os
import argparse
from dataclasses import dataclass, field, fields
from typing import Tuple, List, Type

import torch

from model.model_aux import load_checkpoint, get_last_ckpt_idx


def is_bool(value):

    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    raise argparse.ArgumentTypeError(f"'{value}' is not a valid boolean value.")


def compare_cfgs_if_changed(cfg: Type[dataclass], overridden_cfg: Type[dataclass], dont_look_fields: List[str] = [], error_msg:str = None):

    
    if not isinstance(dont_look_fields, list):
        raise TypeError("dont_look_fields must be a list of strings")
    
    if type(cfg) == dict:
        cls = type(overridden_cfg)
        cfg = cls(**cfg)
    
    if type(cfg) != type(overridden_cfg):
        raise TypeError(f"cfg and overridden_cfg must be of same type, type of cfg {type(cfg)},  type of overridden_cfg {type(overridden_cfg)}")

    for field in fields(cfg):
        if getattr(cfg, field.name) != getattr(overridden_cfg, field.name) and field.name not in dont_look_fields:
            raise argparse.ArgumentError(f"{error_msg}. You gave at least one argument for config : {field.name}: {getattr(cfg, field.name)} -> {getattr(overridden_cfg, field.name)}...")


def _parse_args(cfg_classes: Tuple[Type], verbose_changes=True, verbose_all=True) -> Tuple[Type[dataclass]]:


    parser = argparse.ArgumentParser(description='PreTrainBertConfig and BertConfig arguments...')

    # add all config fields to parser
    for cfg_class in cfg_classes:
        for field in cfg_class.__dataclass_fields__.values():
            parser.add_argument('--' + field.name, type=field.type if field.type is not bool else is_bool, help=field.metadata["description"])

    args = parser.parse_args()
    
    # create default configs (they will be overridden by given user args)
    overridden_cfgs = [cls() for cls in cfg_classes]

    if verbose_changes or verbose_all:
        print("\nCONFIGS:")
        print("---------------------------------------")

    # let's override configs by given args (for not given arg use default value of config field)
    for cfg in overridden_cfgs:
        for field in cfg.__dataclass_fields__.values():
            setattr(cfg, field.name, getattr(cfg, field.name) if getattr(args, field.name) is None else getattr(args, field.name))
            if verbose_changes and getattr(cfg, field.name) != field.default:
                print(f"changed {field.name}: {field.default} -----------> {field.name}: {getattr(cfg, field.name)}")
            elif verbose_all:
                print(f"default {field.name}: {getattr(cfg, field.name)}")

    if verbose_changes or verbose_all:
        print("---------------------------------------\n")
    
    return overridden_cfgs




@dataclass
class TrainTokenizerConfig:
    vocab_size: int = field(default=32_000, metadata={"description": "Vocab size of token embeddings"})
    limit_alphabet: int = field(default=200, metadata={"description": "Limit for alphabet"})
    min_frequency: int = field(default=2, metadata={"description": "Min frequency"})
    cased: bool = field(default=True, metadata={"description": "Cased or uncased"})



def get_train_tokenizer_py_config(verbose_changes=True, verbose_all=True) -> TrainTokenizerConfig:

    
    overridden_cfg = _parse_args(cfg_classes=(TrainTokenizerConfig,),
                                 verbose_changes=verbose_changes,
                                 verbose_all=verbose_all)[0]
    
    # let's check all number fields are positive
    for field in fields(overridden_cfg):
        if field.type is int and getattr(overridden_cfg, field.name) < 0:
            raise argparse.ArgumentTypeError(f"{field.name} must be positive")
        
    return overridden_cfg


@dataclass
class DataConfig:
    block_size: int = field(default=256, metadata={"description": "Block size"})
    num_of_docs_per_shard: int = field(default=4_000, metadata={"description": "Number of docs per shard (for doc_shards and ab_shards creation)"})
    num_tokens_per_shard: int = field(default=10_000_000, metadata={"description": "Number of tokens per shard (for xy_shards creation)"})
    overlap: int = field(default=128, metadata={"description": "Overlap, how much overlap between windows when ab samples are generated (suggestion: make half of the block size)"})
    edge_buffer: int = field(default=10, metadata={"description": "prevents the use of a and b seperators near window ends by this many tokens (makes a and b more similar length-wise) (for ab_shards creation)"})
    seed: int = field(default=13013, metadata={"description": "Seed, for reproducibility"})
    rate_of_untouched_words: float = field(default=0.85, metadata={"description": "Rate of untouched words, these are not gonna replaced by [mask, identity, replaced] tokens"})
    mask_ratio: float = field(default=0.80, metadata={"description": "Mask ratio"})
    replace_ratio: float = field(default=0.10, metadata={"description": "Replace ratio"})
    identity_ratio: float = field(default=0.10, metadata={"description": "Identity ratio"})
    tokenizer_type: str = field(default="custom", metadata={"description": "Tokenizer type, Data has been prepared/tokenized with. choices: [custom, hf]"})




def get_prepare_data_py_config(verbose_changes=True, verbose_all=True) -> DataConfig:

    
    overridden_cfg = _parse_args(cfg_classes=(DataConfig,),
                                 verbose_changes=verbose_changes,
                                 verbose_all=verbose_all)[0]
    
    # let's check all number fields are positive
    for field in fields(overridden_cfg):
        if field.type is int and getattr(overridden_cfg, field.name) < 0:
            raise argparse.ArgumentTypeError(f"{field.name} must be positive")
    
    
    if overridden_cfg.mask_ratio + overridden_cfg.replace_ratio + overridden_cfg.identity_ratio != 1:
        raise argparse.ArgumentTypeError("mask_ratio + replace_ratio + identity_ratio must be 1")
    if overridden_cfg.rate_of_untouched_words > 1 or overridden_cfg.rate_of_untouched_words < 0:
        raise argparse.ArgumentTypeError("rate_of_untouched_words must be between 0 and 1")
    if overridden_cfg.tokenizer_type not in ["custom", "hf"]:
        raise argparse.ArgumentTypeError("tokenizer_type must be 'custom' or 'hf'")
    
    return overridden_cfg




@dataclass
class RandomWordSetConfig:
    limit_for_token_group: int = field(default=5, metadata={"description": "Limit for token group"})
    max_word_limit_for_token_group: int = field(default=5_000, metadata={"description": "Max word limit for token group"})
    min_freq_for_words: int = field(default=150, metadata={"description": "Min freq for words"})
    random_sample: bool = field(default=False, metadata={"description": "Randomly sample words (not directly most frequent ones)"})
    use_number_of_line: int = field(default=None, metadata={"description": "Use specified number of lines for training (if None then use all lines)"})
    tokenizer_type: str = field(default="custom", metadata={"description": "Tokenizer type, random_word_set.json prepared/tokenized with. choices: [custom, hf]"})



def get_random_word_set_py_config(verbose_changes=True, verbose_all=True) -> RandomWordSetConfig:

    
    overridden_cfg = _parse_args(cfg_classes=(RandomWordSetConfig,),
                                 verbose_changes=verbose_changes,
                                 verbose_all=verbose_all)[0]
    

    
    # let's check all number fields are positive
    for field in fields(overridden_cfg):
        if field.type is int and getattr(overridden_cfg, field.name) is not None and getattr(overridden_cfg, field.name) < 0:
            raise argparse.ArgumentTypeError(f"{field.name} must be positive")
        
    if overridden_cfg.tokenizer_type not in ["custom", "hf"]:
        raise argparse.ArgumentTypeError("tokenizer_type must be 'custom' or 'hf'")
        
    return overridden_cfg





@dataclass
class BertConfig:
    vocab_size: int = field(default=32000, metadata={"description": "Vocabulary size of token embeddings"})
    hidden_size: int = field(default=768, metadata={"description": "Hidden size of feed-forward layers"})
    num_hidden_layers: int = field(default=12, metadata={"description": "Number of layers in encoder"})
    num_attention_heads : int = field(default=12, metadata={"description": "Number of attention heads"})
    hidden_act: str = field(default="gelu", metadata={"description": "Activation function type choices: [gelu, relu]"})
    intermediate_size: int = field(default=3072, metadata={"description": "Size of the intermediate layer in feed-forward"}) 
    hidden_dropout_prob: float = field(default=0.1, metadata={"description": "Dropout probability for hidden layers"})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"description": "Dropout probability for attention"})
    max_position_embeddings: int = field(default=512, metadata={"description": "Maximum number of position embeddings (max block size of the model)"})
    initializer_range: float = field(default=0.02, metadata={"description": "Standard deviation for linear and embedding initialization"})
    layer_norm_eps: float = field(default=1e-12, metadata={"description": "Layer norm epsilon"})
    type_vocab_size: int = field(default=2, metadata={"description": "Type vocabulary size (for token type embeddings)"})
    classifier_dropout: float = field(default=0.1, metadata={"description": "Dropout probability for classifier"})

    

@dataclass
class PreTrainBertConfig:
    do_train_custom: bool = field(default=True, metadata={"description": "Do train with custom model or with BERTurk hf model weights."})
    do_eval_from_best_ckpt: bool = field(default=False, metadata={"description": "Just evaluate the model from_best_ckpt but not training."})
    do_eval_from_huggingface: bool = field(default=False, metadata={"description": "Just evaluate the model from_huggingface but not training."})
    resume: bool = field(default=False, metadata={"description": "Resume training from the last step"})
    stage1_ratio: float = field(default=0.9, metadata={"description": "Ratio of stage1 (e.g. 0.9 means use block_size_s1 and train_batch_size_s1 until the end of %90 training then switch to stage2)"})
    block_size_s1: int = field(default=256, metadata={"description": "Block size for stage1"})
    block_size_s2: int = field(default=512, metadata={"description": "Block size for stage2"})
    train_batch_size_s1: int = field(default=32, metadata={"description": "Training batch size for stage1"})
    train_batch_size_s2: int = field(default=16, metadata={"description": "Training batch size for stage2"})
    val_block_size: int = field(default=512, metadata={"description": "Validation block size"})
    val_batch_size: int = field(default=8, metadata={"description": "Validation batch size"})
    grad_accum_steps: int = field(default=5, metadata={"description": "Gradient accumulation steps (micro steps)"})
    max_learning_rate: float = field(default=1e-4, metadata={"description": "Maximum learning rate"})
    min_learning_rate: float = field(default=1e-4 * 0.01, metadata={"description": "Minimum learning rate"})
    lr_scheduler: str = field(default="cosine", metadata={"description": "Learning rate scheduler choices: [linear, cosine]"})
    num_train_steps: int = field(default=1000, metadata={"description": "Number of training steps"}) 
    num_warmup_steps: int = field(default=100, metadata={"description": "Number of warmup steps"})
    save_checkpoints_steps: int = field(default=50, metadata={"description": "Save checkpoints steps"})
    val_check_steps: int = field(default=50, metadata={"description": "Validation check steps"})
    device: str = field(default="cuda", metadata={"description": "Device choices: [cpu, cuda, mps]"})
    max_eval_steps: int = field(default=50, metadata={"description": "Maximum evaluation steps (if validation set is small then max_eval_steps is gonna be treated as validation set size)"})
    weight_decay: float = field(default=0.01, metadata={"description": "Weight decay (L2 regularization)"})
    max_ckpt: int = field(default=5, metadata={"description": "Maximum number of last checkpoints to keep"})
    seed: int = field(default=13013, metadata={"description": "Random seed"})
    generate_samples: bool = field(default=True, metadata={"description": "Try model on predefined samples"})
    mlflow_tracking: bool = field(default=True, metadata={"description": "MLflow tracking"})      
    tokenizer_type: str = field(default="custom", metadata={"description": "Tokenizer type, Data has been prepared/tokenized with. choices: [custom, hf]"})



def get_pretrain_bert_py_configs(verbose_changes=True, verbose_all=True) -> Tuple[BertConfig, PreTrainBertConfig]:


    # let's parse command line arguments
    overridden_bert_cfg, overridden_pretrain_cfg = _parse_args(cfg_classes=(BertConfig, PreTrainBertConfig),
                                                               verbose_changes=verbose_changes,
                                                               verbose_all=verbose_all)
    
    default_pretrain_cfg = PreTrainBertConfig()
    default_bert_cfg = BertConfig()


    # control over given args
    if overridden_bert_cfg.hidden_act not in ["gelu", "relu"]:
        raise argparse.ArgumentError("hidden_act should be gelu or relu")
    if overridden_pretrain_cfg.min_learning_rate > overridden_pretrain_cfg.max_learning_rate:
        raise argparse.ArgumentError("min_learning_rate should be less than max_learning_rate")
    if overridden_pretrain_cfg.device not in ["cpu", "cuda", "mps"]:
        raise argparse.ArgumentError("device should be cpu, cuda or mps")
    if overridden_pretrain_cfg.device == "cuda" and not torch.cuda.is_available():
        raise argparse.ArgumentError("cuda is not available")
    if overridden_pretrain_cfg.device == "mps" and not torch.backends.mps.is_available():
        raise argparse.ArgumentError("mps is not available")
    if overridden_pretrain_cfg.lr_scheduler not in ["linear", "cosine"]:
        raise argparse.ArgumentError("lr_scheduler should be linear or cosine")
    if overridden_pretrain_cfg.tokenizer_type not in ["custom", "hf"]:
        raise argparse.ArgumentError("tokenizer_type should be custom or hf")

    # eval, scratch, resume


    #  IF DO JUST EVAL
    # --------------------------------------------------------------------------------------------------------------------------

    if overridden_pretrain_cfg.do_eval_from_best_ckpt and overridden_pretrain_cfg.do_eval_from_huggingface:
        raise argparse.ArgumentError("do_eval_from_best_ckpt and do_eval_from_huggingface cannot be True at the same time")
    

    # if do eval from best ckpt
    if overridden_pretrain_cfg.do_eval_from_best_ckpt:
        if overridden_pretrain_cfg.generate_samples == False:
            raise argparse.ArgumentError("generate_samples should be True if do_eval_from_best_ckpt is True")
        
        best_ckpt = load_checkpoint(postfix="best")
        ckpt_pretrain_cfg = best_ckpt["pretrain_config"]
        ckpt_bert_cfg = best_ckpt["bert_config"]
        

        dont_look_fields = ["do_eval_from_best_ckpt", "seed", "max_eval_steps", "device", "val_block_size", "val_batch_size"]

        compare_cfgs_if_changed(cfg=ckpt_pretrain_cfg, overridden_cfg=overridden_pretrain_cfg, dont_look_fields=dont_look_fields,
                                error_msg=f"for do_eval_from_best_ckpt you should not give any pretrain cfg args rather than : {dont_look_fields}.")

        compare_cfgs_if_changed(cfg=ckpt_bert_cfg, overridden_cfg=overridden_bert_cfg,
                                error_msg=f"for do_eval_from_best_ckpt you should not give bert cfg args, it will use best ckpt model cfg...")
        
        return overridden_bert_cfg, overridden_pretrain_cfg
        

    # if do eval from huggingface
    if overridden_pretrain_cfg.do_eval_from_huggingface:
        if overridden_pretrain_cfg.generate_samples == False:
            raise argparse.ArgumentError("generate_samples should be True if do_eval_from_best_ckpt or do_eval_from_huggingface is True")
        
        dont_look_fields = ["do_eval_from_huggingface", "seed", "max_eval_steps", "device", "val_block_size", "val_batch_size"]

        compare_cfgs_if_changed(cfg=default_pretrain_cfg, overridden_cfg=overridden_pretrain_cfg, dont_look_fields=dont_look_fields,
                                error_msg=f"for do_eval_from_huggingface you should not give any pretrain cfg args rather than : {dont_look_fields}.")

        compare_cfgs_if_changed(cfg=default_bert_cfg, overridden_cfg=overridden_bert_cfg,
                                error_msg=f"for do_eval_from_huggingface you should not give bert cfg args, it will use default model cfg...")
        
        return overridden_bert_cfg, overridden_pretrain_cfg
    # --------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------


    #  IF RESUME
    # --------------------------------------------------------------------------------------------------------------------------
    if overridden_pretrain_cfg.resume:
        last_ckpt_idx = get_last_ckpt_idx()
        ckpt_dict = load_checkpoint(last_ckpt_idx)
        ckpt_bert_cfg = ckpt_dict["model_config"]
        ckpt_pretrain_cfg = ckpt_dict["pretrain_config"]

        compare_cfgs_if_changed(cfg=ckpt_bert_cfg, overridden_cfg=overridden_bert_cfg, dont_look_fields=["resume"],
                                error_msg="for resuming training you should not give any different bert cfg args (it should use its saved bert cfg)...")
        
        # we can change some pretrain cfg args when resuming
        all_fields = set(overridden_pretrain_cfg.__dataclass_fields__.keys())
        look_fields = set(["max_ckpt", "seed", "do_train_custom", "do_eval_from_best_ckpt", "do_eval_from_huggingface", "tokenizer_type"]) 

        # all_fields - dont_change_fields -> returns fields that we can change by args (so we should not look at them when comparing)
        dont_look_fields_for_pretrain = list(all_fields - look_fields)
        compare_cfgs_if_changed(cfg=ckpt_pretrain_cfg, overridden_cfg=overridden_pretrain_cfg, dont_look_fields=dont_look_fields_for_pretrain,
                                error_msg=f"for resuming training you can only give/change some pretrain cfg args: {dont_look_fields_for_pretrain}...")
        
        return overridden_bert_cfg, overridden_pretrain_cfg
    # --------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------    


    # IF DO TRAIN WITH HF MODEL WEIGHTS
    # --------------------------------------------------------------------------------------------------------------------------
    if overridden_pretrain_cfg.do_train_custom == False:
        # you cannot change bert_config if do train with hf model weights (default config is fixed for hf model weights (BERTurk)) 
        compare_cfgs_if_changed(cfg=default_bert_cfg, overridden_cfg=overridden_bert_cfg,
                                error_msg="for training with hf model weights (BERTurk) bert_config should be fixed (default).")
        # tokenizer_type should be hf if do train with hf model weights
        if overridden_pretrain_cfg.tokenizer_type != "hf":
            raise argparse.ArgumentError("tokenizer_type should be 'hf' if do_train_custom is False")
        
        return overridden_bert_cfg, overridden_pretrain_cfg
    # --------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------- 
    

    # IF DO TRAIN WITH CUSTOM MODEL
    if overridden_pretrain_cfg.tokenizer_type != "custom":
        raise argparse.ArgumentError("tokenizer_type should be 'custom' if do_train_custom is True")
    return overridden_bert_cfg, overridden_pretrain_cfg
    
    

    