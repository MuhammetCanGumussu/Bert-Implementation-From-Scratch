
import argparse
from dataclasses import dataclass, field
from typing import Tuple


from bert_implementation_tr.data.data_aux import PAD_TOKEN_ID





@dataclass
class BertConfig:
    vocab_size: int = field(default=32000, metadata={"description": "Vocabulary size of token embeddings"})
    hidden_size: int = field(default=768, metadata={"description": "Hidden size of feed-forward layers"})
    num_hidden_layers: int = field(default=12, metadata={"description": "Number of layers in encoder"})
    num_attention_heads : int = field(default=12, metadata={"description": "Number of attention heads"})
    hidden_act: str = field(default="gelu", metadata={"description": "Activation function type", "choices": ["gelu", "relu"]})
    intermediate_size: int = field(default=hidden_size * 4, metadata={"description": "Size of the intermediate layer in feed-forward"}) 
    hidden_dropout_prob: float = field(default=0.1, metadata={"description": "Dropout probability for hidden layers"})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"description": "Dropout probability for attention"})
    max_position_embeddings: int = field(default=512, metadata={"description": "Maximum number of position embeddings (max block size of the model)"})
    initializer_range: float = field(default=0.02, metadata={"description": "Standard deviation for linear and embedding initialization"})
    layer_norm_eps: float = field(default=1e-12, metadata={"description": "Layer norm epsilon"})
    type_vocab_size: int = field(default=2, metadata={"description": "Type vocabulary size (for token type embeddings)"})
    classifier_dropout: float = field(default=0.1, metadata={"description": "Dropout probability for classifier"})
    pad_token_id: int = field(default=PAD_TOKEN_ID, metadata={"description": "Pad token id"})



@dataclass
class PreTrainBertConfig:
    do_eval: bool = field(default=False, metadata={"description": "Just evaluate the model (from_best_ckpt or from_huggingface) but not training"})
    from_best_ckpt: bool = field(default=False, metadata={"description": "Do evaluation from the best ckpt"})
    from_huggingface: bool = field(default=False, metadata={"description": "Do evaluation from huggingface pretrained model"})
    resume: bool = field(default=False, metadata={"description": "Resume training from the last step"})
    block_size: int = field(default=512, metadata={"description": "Block size"})
    train_batch_size: int = field(default=32, metadata={"description": "Training batch size"})
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
    generate_samples: bool = field(default=False, metadata={"description": "Try model on some samples"})
    mlflow_tracking: bool = field(default=False, metadata={"description": "MLflow tracking"})      


def get_data_py_config():
    raise NotImplementedError("Not implemented yet")


def get_pretrain_bert_py_configs() -> Tuple[BertConfig, PreTrainBertConfig]:
    parser = argparse.ArgumentParser(description='PreTrainBertConfig and BertConfig arguments...')
    
    for cfg_class in [BertConfig, PreTrainBertConfig]:
        for field in cfg_class.__dataclass_fields__.values():
            parser.add_argument('--' + field.name, type=field.type, help=field.metadata["description"])

    args = parser.parse_args()

    bert_cfg = BertConfig()
    pretrain_cfg = PreTrainBertConfig() 

    # let's override configs by given args (for not given arg use default value of config field)
    for cfg in [bert_cfg, pretrain_cfg]:
        for field in cfg.__dataclass_fields__.values():
            # if arg is None, use default cfg value
            setattr(cfg, field.name, getattr(cfg, field.name) if getattr(args, field.name) is None else getattr(args, field.name))


    # control
    # örn pad token id dışında hiç bir değer negative olamaz
    # dropout probability dışında hiç bir değer 0 da olamaz




    return bert_cfg, pretrain_cfg

