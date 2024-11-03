
import argparse
from dataclasses import dataclass, field
from typing import Tuple

import torch





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
    # bakılacak: şimdilik 0 değerini verdim (pad_token id) ancak bunu -100 yapacaksın
    # bakılacak: ASLINDA modelin configinde ignore index olmaması gerekiyor, data config'de olmalı, model ignore index'ini (loss için) data config'den almalı!
    ignore_index: int = field(default=0, metadata={"description": "Pad token id"})   



@dataclass
class PreTrainBertConfig:
    do_train_custom: bool = field(default=True, metadata={"description": "Do train with custom model or with BERTurk hf model weights."})
    do_eval_from_best_ckpt: bool = field(default=False, metadata={"description": "Just evaluate the model from_best_ckpt but not training."})
    do_eval_from_huggingface: bool = field(default=False, metadata={"description": "Just evaluate the model from_huggingface but not training."})
    resume: bool = field(default=False, metadata={"description": "Resume training from the last step"})
    stage1_ratio: float = field(default=0.9, metadata={"description": "Ratio of stage1 (e.g. 0.9 means use block_size_s1 and train_batch_size_s1 until the end of %90 training then switch to stage2)"})
    block_size_s1: int = field(default=512, metadata={"description": "Block size for stage1"})
    block_size_s2: int = field(default=512, metadata={"description": "Block size for stage2"})
    train_batch_size_s1: int = field(default=32, metadata={"description": "Training batch size for stage1"})
    train_batch_size_s2: int = field(default=32, metadata={"description": "Training batch size for stage2"})
    val_block_size: int = field(default=512, metadata={"description": "Validation block size"})
    val_batch_size: int = field(default=8, metadata={"description": "Validation batch size"})
    grad_accum_steps: int = field(default=1, metadata={"description": "Gradient accumulation steps (micro steps)"})
    max_learning_rate: float = field(default=1e-4, metadata={"description": "Maximum learning rate"})
    min_learning_rate: float = field(default=1e-4 * 0.01, metadata={"description": "Minimum learning rate"})
    lr_scheduler: str = field(default="cosine", metadata={"description": "Learning rate scheduler", "choices": ["linear", "cosine"]})
    num_train_steps: int = field(default=1000, metadata={"description": "Number of training steps"}) 
    num_warmup_steps: int = field(default=100, metadata={"description": "Number of warmup steps"})
    save_checkpoints_steps: int = field(default=50, metadata={"description": "Save checkpoints steps"})
    val_check_steps: int = field(default=50, metadata={"description": "Validation check steps"})
    device: str = field(default="cuda", metadata={"description": "Device", "choices": ["cpu", "cuda", "mps"]})
    max_eval_steps: int = field(default=50, metadata={"description": "Maximum evaluation steps (if validation set is small then max_eval_steps is gonna be treated as validation set size)"})
    weight_decay: float = field(default=0.01, metadata={"description": "Weight decay (L2 regularization)"})
    max_ckpt: int = field(default=5, metadata={"description": "Maximum number of last checkpoints to keep"})
    seed: int = field(default=13013, metadata={"description": "Random seed"})
    generate_samples: bool = field(default=True, metadata={"description": "Try model on predefined samples"})
    mlflow_tracking: bool = field(default=True, metadata={"description": "MLflow tracking"})      


def is_bool(value):
    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    raise argparse.ArgumentTypeError(f"'{value}' is not a valid boolean value.")



def get_data_py_config():
    raise NotImplementedError("Not implemented yet")




def get_pretrain_bert_py_configs() -> Tuple[BertConfig, PreTrainBertConfig]:
    parser = argparse.ArgumentParser(description='PreTrainBertConfig and BertConfig arguments...')
    
    for cfg_class in [BertConfig, PreTrainBertConfig]:
        for field in cfg_class.__dataclass_fields__.values():
            parser.add_argument('--' + field.name, type=field.type if field.type is not bool else is_bool, help=field.metadata["description"])

    args = parser.parse_args()
    
    bert_cfg = BertConfig()
    pretrain_cfg = PreTrainBertConfig()

    # let's override defaultconfigs by given args (for not given arg use default value of config field)
    for cfg in [bert_cfg, pretrain_cfg]:
        for field in cfg.__dataclass_fields__.values():
            print("old:", field.name, " ", getattr(cfg, field.name))
            # if arg is None, use default cfg value
            setattr(cfg, field.name, getattr(cfg, field.name) if getattr(args, field.name) is None else getattr(args, field.name))
            
            print("new:", field.name, " ", getattr(cfg, field.name))


    # you cannot change bert_config if do train with hf model weights (default config is fixed for hf model weights (BERTurk))
    if pretrain_cfg.do_train_custom == False:
        for field in bert_cfg.__dataclass_fields__.values():
            if getattr(bert_cfg, field.name) != field.default:
                raise argparse.ArgumentTypeError(f"for training with hf model weights (BERTurk) bert_config is fixed. You gave at least one argument for model config : {field.name}: {field.default} -> {getattr(bert_cfg, field.name)}...")


    # some controls (i am not gonna control every possibility or edge case)
    if bert_cfg.hidden_act not in ["gelu", "relu"]:
        raise argparse.ArgumentTypeError("hidden_act should be gelu or relu")
    if pretrain_cfg.min_learning_rate > pretrain_cfg.max_learning_rate:
        raise argparse.ArgumentTypeError("min_learning_rate should be less than max_learning_rate")
    if pretrain_cfg.do_eval_from_best_ckpt and pretrain_cfg.do_eval_from_huggingface:
        raise argparse.ArgumentTypeError("do_eval_from_best_ckpt and do_eval_from_huggingface cannot be True at the same time")
    if pretrain_cfg.device not in ["cpu", "cuda", "mps"]:
        raise argparse.ArgumentTypeError("device should be cpu, cuda or mps")
    if pretrain_cfg.device == "cuda" and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("cuda is not available")
    if pretrain_cfg.device == "mps" and not torch.backends.mps.is_available():
        raise argparse.ArgumentTypeError("mps is not available")
    if pretrain_cfg.lr_scheduler not in ["linear", "cosine"]:
        raise argparse.ArgumentTypeError("lr_scheduler should be linear or cosine")

    
    


    return bert_cfg, pretrain_cfg