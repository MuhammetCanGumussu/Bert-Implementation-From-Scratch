"""CAUTION: Some docstrings are generated by AI (Codeium), so they may not always make sense."""


import argparse
from dataclasses import dataclass, field, fields, asdict
from typing import Tuple, List, Type, Any

import torch

from model.model_aux import load_checkpoint, get_last_ckpt_idx


def is_bool(value):
    """Converts given value to a boolean.
    
    Args:
        value: String value to be converted to boolean.
        
    Returns:
        bool: True if value is 'true' or '1', False if value is 'false' or '0'.
        
    Raises:
        argparse.ArgumentTypeError: If value is not a valid boolean string.
    """
    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    raise argparse.ArgumentTypeError(f"'{value}' is not a valid boolean value.")


def compare_cfgs_if_changed(cfg: Type[dataclass], overridden_cfg: Type[dataclass], dont_look_fields: List[str] = [], error_msg:str = None):
    """Compares two dataclass objects and raises an ArgumentTypeError if any of their fields are different.

    Args:
        cfg: The dataclass object to compare with overridden_cfg.
        overridden_cfg: The dataclass object to compare with cfg.
        dont_look_fields: A list of strings of field names that should not be compared.
        error_msg: A string to be prepended to the error message if an ArgumentTypeError is raised.
    """
    if not isinstance(dont_look_fields, list):
        raise TypeError("dont_look_fields must be a list of strings")
    
    if type(cfg) == dict:
        cls = type(overridden_cfg)
        cfg = cls(**cfg)
    
    if type(cfg) != type(overridden_cfg):
        raise TypeError(f"cfg and overridden_cfg must be of same type, type of cfg {type(cfg)},  type of overridden_cfg {type(overridden_cfg)}")

    for field in fields(cfg):
        if getattr(cfg, field.name) != getattr(overridden_cfg, field.name) and field.name not in dont_look_fields:
            raise argparse.ArgumentTypeError(f"{error_msg}. You gave at least one argument for config : {field.name}: {getattr(cfg, field.name)} -> {getattr(overridden_cfg, field.name)}...")


def _parse_args(cfg_classes: Tuple[Type], verbose_changes=True, verbose_all=True) -> Tuple[Type[dataclass]]:
    """Parses command line arguments for given dataclass configurations.

    Args:
        cfg_classes: A tuple of dataclass configurations.
        verbose_changes: A boolean indicating whether to print changes in configurations.
        verbose_all: A boolean indicating whether to print all configurations.

    Returns:
        A tuple of overridden configurations.
    """
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
    """
    Returns TrainTokenizerConfig object with overridden values by given args.

    Args:
        verbose_changes (bool): If True, print changed config values.
        verbose_all (bool): If True, print all config values.

    Returns:
        TrainTokenizerConfig

    Raises:
        argparse.ArgumentTypeError: If any int field is negative.
    """
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
    """
    Returns a DataConfig object with overridden values based on command line arguments.

    Args:
        verbose_changes (bool): If True, print changed configuration values.
        verbose_all (bool): If True, print all configuration values.

    Returns:
        DataConfig: The configuration object with values overridden by command line arguments.

    Raises:
        argparse.ArgumentTypeError: If any int field is negative, the sum of mask_ratio, 
                                    replace_ratio, and identity_ratio is not 1, 
                                    rate_of_untouched_words is not between 0 and 1, 
                                    or tokenizer_type is not 'custom' or 'hf'.
    """
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
    """
    Returns RandomWordSetConfig object with overridden values by given args.

    Args:
        verbose_changes (bool): If True, print changed config values.
        verbose_all (bool): If True, print all config values.

    Returns:
        RandomWordSetConfig

    Raises:
        argparse.ArgumentTypeError: If any int field is negative.
    """
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
    """This is from scratch scenario default config!"""
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
    grad_accum_steps: int = field(default=1, metadata={"description": "Gradient accumulation steps (micro steps)"})
    max_learning_rate: float = field(default=1e-2, metadata={"description": "Maximum learning rate"})
    min_learning_rate: float = field(default=1e-2 * 0.001, metadata={"description": "Minimum learning rate"})
    lr_scheduler: str = field(default="cosine", metadata={"description": "Learning rate scheduler choices: [linear, cosine]"})
    num_train_steps: int = field(default=10_000, metadata={"description": "Number of training steps"}) 
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



def _parse_argsv2(cfg_classes: Tuple[Type]) -> List[argparse.Namespace]:
    """
    Parse command line arguments for given dataclass configurations.

    :param cfg_classes: Tuple of dataclass configurations
    :return: List of argparse.Namespace objects, each containing parsed arguments relative to corresponding dataclass configuration
    """
    parser = argparse.ArgumentParser(description=f'{cfg_classes} arguments...')

    for cfg_class in cfg_classes:
        for field in cfg_class.__dataclass_fields__.values():
            parser.add_argument('--' + field.name, type=field.type if field.type is not bool else is_bool, help=field.metadata["description"])

    all_args = parser.parse_args()

    list_of_args = []

    # let's seperate args relative to each config class
    for cfg_class in cfg_classes:
        arg = argparse.Namespace(**{k:None for k in asdict(cfg_class()).keys()})
        list_of_args.append(arg)
        for key in vars(arg).keys():
            setattr(arg, key, getattr(all_args, key))

    return list_of_args



def _control_if_user_give_extra_args(can_be_given_keys:List[str], args:argparse.Namespace):
    """
    Checks if user give extra arguments which are not in can_be_given_keys
    
    Args:
        can_be_given_keys (List[str]): List of keys that can be given by user
        args (argparse.Namespace): Parsed arguments
    Raises:
        argparse.ArgumentTypeError: If user give extra arguments which are not in can_be_given_keys
    """
    for key in set(args.__dict__.keys()) - set(can_be_given_keys):
        if args.__dict__[key] is not None:
            raise argparse.ArgumentTypeError(f"For this scenario you should not give {key} arg... (the args you can give are: {can_be_given_keys if len(can_be_given_keys) > 0 else 'None'})")
    


def _assign_dict_to_args(dict_to_assign:dict[str, Any], args:argparse.Namespace, fixed_scenario_args:dict[str, Any]):
    """assigns dict_to_assign to 'args' for none values. If user gives different values for fixed_scenario_args then raise error"""
    if type(args) != argparse.Namespace:
        raise TypeError("args must be argparse.Namespace")
    if type(dict_to_assign) != dict:
        raise TypeError("dict_to_assign must be dict")
    if type(fixed_scenario_args) != dict:
        raise TypeError("fixed_scenario_args must be dict")

    for k, v in vars(args).items():
        if k in fixed_scenario_args and v is None:
            setattr(args, k, fixed_scenario_args[k])
        elif k in fixed_scenario_args and v != fixed_scenario_args[k]:
            raise argparse.ArgumentTypeError(f"you should not give {k} arg to this value {v}. The value should be {fixed_scenario_args[k]} (fixed value)")
        elif k in dict_to_assign and v is None:
            setattr(args, k, dict_to_assign[k])
    


def _control_consistency(pretrain_args:argparse.Namespace, model_args:argparse.Namespace):
    """general controls for some strings, device and lr max-min range"""
    # lets take not none args and check if they are consistent
    if model_args.hidden_act is not None and model_args.hidden_act not in ["gelu", "relu"]:
        raise argparse.ArgumentTypeError("hidden_act should be gelu or relu")
    if pretrain_args.min_learning_rate is not None and pretrain_args.max_learning_rate is not None:
        if pretrain_args.min_learning_rate > pretrain_args.max_learning_rate:
            raise argparse.ArgumentTypeError("min_learning_rate should be less than max_learning_rate")
    if pretrain_args.device is not None:
        if pretrain_args.device not in ["cpu", "cuda", "mps"]:
            raise argparse.ArgumentTypeError("device should be cpu, cuda or mps")
        if pretrain_args.device == "cuda" and not torch.cuda.is_available():
            raise argparse.ArgumentTypeError("cuda is not available")
        if pretrain_args.device == "mps" and not torch.backends.mps.is_available():
            raise argparse.ArgumentTypeError("mps is not available")
    if pretrain_args.lr_scheduler is not None and pretrain_args.lr_scheduler not in ["linear", "cosine"]:
        raise argparse.ArgumentTypeError("lr_scheduler should be linear or cosine")
    if pretrain_args.tokenizer_type is not None and pretrain_args.tokenizer_type not in ["custom", "hf"]:
        raise argparse.ArgumentTypeError("tokenizer_type should be custom or hf")



def _control_args(user_can_give_pretrain:List[str],
                  user_can_give_model:List[str],
                  default_values_pretrain:dict[str, Any],
                  default_values_model:dict[str, Any],
                  fixed_scenario_args_pretrain:dict[str, Any],
                  fixed_scenario_args_model:dict[str, Any],
                  pretrain_args:argparse.Namespace,
                  model_args:argparse.Namespace,
                  verbose:bool=True):

    # control if user give extra args
    _control_if_user_give_extra_args(user_can_give_pretrain, pretrain_args) 
    # assign default values for user_can_give args that are None, for some args fixed values should be assigned
    _assign_dict_to_args(default_values_pretrain, pretrain_args, fixed_scenario_args_pretrain)


    # control if user give extra args
    _control_if_user_give_extra_args(user_can_give_model, model_args)
    # assign default values for user_can_give args that are None, for some args fixed values should be assigned
    _assign_dict_to_args(default_values_model, model_args, fixed_scenario_args_model)


    # control consistency
    _control_consistency(model_args=model_args, pretrain_args=pretrain_args)


    if verbose:
        print("model args:","\n-----------------------\n", vars(model_args))
        print("\npretrain args:","\n-----------------------\n", vars(pretrain_args))
    




def get_pretrain_bert_py_configs(verbose:bool=True) -> Tuple[BertConfig, dict[str, Any]]:
    """returns BertConfig dataclass and pretrain args dict"""
    
    print("Be careful! There are three scenarios: DO_EVAL, RESUME, DO_TRAIN_CUSTOM (from scratch).")
    print("The scenario precedences are: [do_eval_from_(...) >>> resume >>> do_train_custom]\n")

    model_args, pretrain_args = _parse_argsv2((BertConfig, PreTrainBertConfig))

    
    
    if pretrain_args.do_eval_from_best_ckpt and pretrain_args.do_eval_from_huggingface:
        raise argparse.ArgumentTypeError("do_eval_from_best_ckpt and do_eval_from_huggingface cannot be True at the same time")

    if pretrain_args.do_eval_from_best_ckpt:
        print("'do eval from best ckpt' scenario inferenced...\n")

        ckpt = load_checkpoint(postfix="best")
        ckpt_model_config = asdict(ckpt["model_config"])
        ckpt_pretrain_config = ckpt["pretrain_config"]

        # user can give different values for these args, if user gives None then default values will be used
        user_can_give = ["do_eval_from_best_ckpt", "val_block_size", "val_batch_size", "device", "max_eval_steps", "seed", "generate_samples", "tokenizer_type"]
        
        # default values for user_can_give (we will use these values if user gives None)
        default_values = {k:v for k, v in ckpt_pretrain_config.items() if k in user_can_give}


        fixed_scenario_args = {
            "do_eval_from_best_ckpt": True,
            "generate_samples": True,
            "tokenizer_type": ckpt_pretrain_config["tokenizer_type"]
        }

        assert set(fixed_scenario_args.keys()) & set(user_can_give), "fix args cannot be given by user! (there should not be any intersection between fix args and user_can_give)"

        
        _control_args(user_can_give_pretrain=user_can_give,
                      user_can_give_model=[],
                      default_values_pretrain=default_values,
                      default_values_model=ckpt_model_config,
                      fixed_scenario_args_pretrain=fixed_scenario_args,
                      fixed_scenario_args_model={},
                      pretrain_args=pretrain_args,
                      model_args=model_args,
                      verbose=verbose)


        return BertConfig(**vars(model_args)), vars(pretrain_args)
    


    if pretrain_args.do_eval_from_huggingface:
        print("'do eval from huggingface' scenario inferenced...\n")

        # user can give different values for these args, if user gives None then default values will be used
        user_can_give = ["do_eval_from_huggingface", "val_block_size", "val_batch_size", "device", "max_eval_steps", "seed"]

        # default values for user_can_give (we will use these values if user gives None)
        default_values = {field.name:field.default for field in fields(PreTrainBertConfig) if field.name in user_can_give}


        fixed_scenario_args = {
            "do_eval_from_huggingface": True,
            "generate_samples": True,
            "tokenizer_type": "hf"
        }

        assert set(fixed_scenario_args.keys()) & set(user_can_give), "fix args cannot be given by user! (there should not be any intersection between fix args and user_can_give)"

        _control_args(user_can_give_pretrain=user_can_give,
                      user_can_give_model=[],
                      default_values_pretrain=default_values,
                      default_values_model=asdict(BertConfig()),
                      fixed_scenario_args_pretrain=fixed_scenario_args,
                      fixed_scenario_args_model={},
                      pretrain_args=pretrain_args,
                      model_args=model_args,
                      verbose=verbose)


        return BertConfig(**vars(model_args)), vars(pretrain_args)
    

    if pretrain_args.resume:
        print("'resume' scenario inferenced...\n")

        ckpt = load_checkpoint(postfix=get_last_ckpt_idx())
        ckpt_model_config = asdict(ckpt["model_config"]) 
        ckpt_pretrain_config = ckpt["pretrain_config"]

        user_cannot_give = ["do_train_custom", "do_eval_from_best_ckpt", "do_eval_from_huggingface", "max_ckpt", "tokenizer_type"]
        user_can_give = list(set(asdict(PreTrainBertConfig()).keys()) - set(user_cannot_give)) + ["resume"]

        default_values = {k:v for k, v in ckpt_pretrain_config.items() if k in user_can_give}

        fixed_scenario_args = {
            "resume": True,
            "tokenizer_type": ckpt_pretrain_config["tokenizer_type"],
            "max_ckpt": ckpt_pretrain_config["max_ckpt"]
        }

        assert set(fixed_scenario_args.keys()) & set(user_can_give), "fix args cannot be given by user! (there should not be any intersection between fix args and user_can_give)"

        _control_args(user_can_give_pretrain=user_can_give,
                      user_can_give_model=[],
                      default_values_pretrain=default_values,
                      default_values_model=ckpt_model_config,
                      fixed_scenario_args_pretrain=fixed_scenario_args,
                      fixed_scenario_args_model={},
                      pretrain_args=pretrain_args,
                      model_args=model_args,
                      verbose=verbose)


        return BertConfig(**vars(model_args)), vars(pretrain_args)
    

    if pretrain_args.do_train_custom == False:
        print("'do train from hf' scenario inferenced...\n")

        user_cannot_give = ["do_eval_from_best_ckpt", "do_eval_from_huggingface", "tokenizer_type", "resume"]

        user_can_give = list(set(asdict(PreTrainBertConfig()).keys()) - set(user_cannot_give)) + ["do_train_custom"]

        default_values = {k:v for k, v in asdict(PreTrainBertConfig()).items() if k in user_can_give}

        fixed_scenario_args = {
            "do_train_custom": False,
            "tokenizer_type": "hf"
        }

        assert set(fixed_scenario_args.keys()) & set(user_can_give), "fix args cannot be given by user! (there should not be any intersection between fix args and user_can_give)"

        _control_args(user_can_give_pretrain=user_can_give,
                      user_can_give_model=[],
                      default_values_pretrain=default_values,
                      default_values_model=asdict(BertConfig()),
                      fixed_scenario_args_pretrain=fixed_scenario_args,
                      fixed_scenario_args_model={},
                      pretrain_args=pretrain_args,
                      model_args=model_args,
                      verbose=verbose)


        return BertConfig(**vars(model_args)), vars(pretrain_args)
    
    else:
        print("'do train custom' scenario inferenced...\n")

        user_cannot_give = ["do_eval_from_best_ckpt", "do_eval_from_huggingface", "tokenizer_type", "resume"]

        user_can_give = list(set(asdict(PreTrainBertConfig()).keys()) - set(user_cannot_give)) + ["do_train_custom"]

        default_values = {k:v for k, v in asdict(PreTrainBertConfig()).items() if k in user_can_give}

        fixed_scenario_args = {
            "do_train_custom": True,
            "tokenizer_type": "custom"
        }

        assert set(fixed_scenario_args.keys()) & set(user_can_give), "fix args cannot be given by user! (there should not be any intersection between fix args and user_can_give)"

        _control_args(user_can_give_pretrain=user_can_give,
                      user_can_give_model=asdict(BertConfig()).keys(),
                      default_values_pretrain=default_values,
                      default_values_model=asdict(BertConfig()),
                      fixed_scenario_args_pretrain=fixed_scenario_args,
                      fixed_scenario_args_model={},
                      pretrain_args=pretrain_args,
                      model_args=model_args,
                      verbose=verbose)


        return BertConfig(**vars(model_args)), vars(pretrain_args)
    
        
        
