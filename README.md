🇬🇧 [English](./README.md) &nbsp; | &nbsp; 🇹🇷 [Türkçe](./README-TR.md)


Project Description
------------
This project is a system for building a ``BERT`` model from scratch and training it on pretraining tasks to create a baseline model for downstream tasks. The ``BERT`` model can be thought of as the encoder part of the Transformer architecture. Unlike decoders (such as GPT models), each token's context is not only derived from the tokens to its left but also from the tokens to its right.

The ``BERT`` model is trained in a self-supervised manner. By corrupting (or masking) the input signal in a specific way or form, the model is asked to predict the corrupted part of the signal correctly. For this, there are two distinct tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP), which can be considered as a form of multi-task learning.

Samples have the following format: ``[CLS] Seq_A [SEP] Seq_B [SEP]``. For the Next Sentence Prediction (NSP) task, sequences A and B are randomly selected with a 50% probability, where they could be either consecutive or random. For the Masked Language Modeling (MLM) task, certain words in the sample (both in Seq_A and Seq_B) are randomly selected with a certain probability, and each word's tokens are modified in one of three possible ways: (*): Mask, Replace, Identity. In the Mask scenario, the word tokens are replaced with [MASK] tokens, in the Replace scenario, the word tokens are replaced with tokens of a randomly chosen word of the same token length, and in the Identity scenario, the word tokens remain unchanged.

(*) : In the original [BERT paper](https://arxiv.org/abs/1810.04805), the Mask and Replace scenarios are performed on a single token, whereas in this project, the ``Whole Word Masking`` method is applied.




Prepare Dataset
------------
A system (``prepare_data.py``) has been developed to work with text files where documents are separated by blank lines. The system first takes the files from the raw folder and creates document shards. Then, it moves a window over all these documents in a manner similar to a convolution operation to generate AB samples. After creating the AB shards, XY shards and a statistics file are generated to prepare the data for model training.

This system can work with the default configurations/parameters as well as user-specified parameters/configurations.

**Note:** The user must specify the correct tokenizer type ("``custom``" or "``hf``") for the model to be used in the dataset creation stage. Details about the tokenizer training for both "``custom``" and "``hf``" can be found in the tokenizer training section.

**Important:** Since the window will often extend beyond the end of the document, instead of filling the remaining space with a [PAD] token, a random B_seq is chosen (making the sample "notNext"). However, this results in an imbalance, with more "notNext" samples than "isNext" samples. In extreme cases, where the overlap and/or block size is increased, this imbalance becomes much more pronounced, with "notNext" samples being up to 9 times more frequent than "isNext" samples. This is known as class imbalance, which is detrimental to stable training and model performance. To mitigate this issue, a weighted loss function was used.




Model
------------
The implementation of the model architecture was based on Huggingface's [modeling_bert.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py). Depending on the configurations specified by the user, the model can be created with different depths, shapes, features, etc. Models initialized from scratch with random parameters/weights can be referred to as "``custom``". In such cases, the user can modify the configurations as desired.

When it is not "``custom``", the pre-trained parameters of the Huggingface [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased) model are transferred to a model that has been "``custom``" created (randomly initialized, with default configs). This process can be referred to as "``hf``". For this transfer to work, the default model configuration must be used (default configurations are fixed for the default ``BERT`` model), meaning the user should not input any model configurations. To make the parameter transfer easier, the keys for the model parameters were kept with the same names as in the original BERTurk model.




Custom Tokenizer Training
------------
Using the HuggingFace [Tokenizers](https://huggingface.co/docs/tokenizers/index) library, the tokenizer is trained with the data from the "raw" folder based on the parameters/configurations specified by the user (details about the configurations are below) (``train_tokenizer.py``). The tokenizer pipeline is similar to the classic BERT tokenizer (e.g., with the same model: WordPiece), but some components have been modified with the assumption that these changes could improve the tokenizer and model's performance. For example, the normalizer component does not include the strip accent feature. In the pretokenizer, not only the WhitespaceSplit component but also digits and punctuation components have been added. However, it has not been tested whether these changes truly improve the model's performance. If different components (such as normalizer, pretokenizer, model, etc.) need to be used or modified in the tokenizer pipeline, changes can be made directly in [train_tokenizer.py](./tokenizer/train_tokenizer.py). The tokenizer can be trained in either cased or uncased mode (default is cased). Finally, the trained tokenizer is saved (e.g., ``tr_wordpiece_tokenizer_cased.json``).

The appropriate tokenizer for the model to be used should be present in the folder. If the model is "``custom``", the tokenizer in this folder will be used. If the model is not "``custom``" but "``hf``", this means the "HuggingFace BERTurk weights" will be used, and in this case, the system will automatically fetch the BERTurk tokenizer (using ``AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")``).




Creating Random Word Set
------------
As mentioned earlier under the "``replace``" scenario, to replace a word with a random word, the number of tokens in the word must first be known. Then, a random word with the same number of tokens should be selected, and its tokens can be returned. To enable this process in a smooth and easy manner, a system (``prepare_random_word_set.py``) should generate the ``random_word_set.json`` file using the parameters/configurations specified by the user (details about the configurations are below).

**Note**: The user must specify the correct tokenizer type ("``custom``" or "``hf``") for the model to be used. Additionally, when using the *use_number_of_line* parameter with a large number, the operation may take a long time. The *use_number_of_line* parameter specifies how many lines of the dataset will be used while creating the random word set.




Pretrain (MLM, NSP)
------------
This system (``pretrain_bert.py``) trains a custom model from scratch or uses HF weights (pre-trained, not random) on pretraining tasks with the configurations/parameters specified by the user. With the checkpoint system, training can continue from the last saved state. During training, the model is run on predefined text samples, and the outputs are saved to "generated_samples.txt". Additionally, metrics or variables like *train_loss*, *val_loss*, *grad_norm*, etc., are tracked/logged. The user can also enable tracking with mlflow by adjusting configurations. 

To make the training/optimization more efficient and stable, several systems and algorithms are used: *lr_scheduler* and *lr_warmup* (defined inside the get_lr function), *amp* (automatic mixed precision), *grad_accum*, *clip_norm*, *adamW fused* (weight decay is applied only on 2D parameter tensors).

Evaluation can be done on an already trained model without training on the pretraining tasks by using the "evaluation" mode ("*do_eval_from_best_ckpt*" or "*samples_for_nsp_generation*" settings). If you want to test the model on different texts (either during training or just during evaluation), changes can be made to the "*samples_for_mlm_generation*" and "*samples_for_nsp_generation*" list objects inside *pretrain_bert.py* (while keeping the formats intact).

In this system, the overall structure was inspired by Andrej Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py) project.

**Note**: "resume" continues from the last checkpoint, whereas when creating a model from scratch with "custom" or "hf", all previous checkpoints and log files (*log.txt* and *generated_samples.txt*) are reset.




Installation
----------
To ensure the project works correctly, the necessary packages should be installed (it is recommended to create a virtual environment first):

```sh
pip install -r requirements.txt
```




Configurations
----------
Parameters/configurations that can be used in ``train_tokenizer.py``:

```python
@dataclass
class TrainTokenizerConfig:
    vocab_size: int = field(default=32_000, metadata={"description": "Vocab size of token embeddings"})
    limit_alphabet: int = field(default=200, metadata={"description": "Limit for alphabet"})
    min_frequency: int = field(default=2, metadata={"description": "Min frequency"})
    cased: bool = field(default=True, metadata={"description": "Cased or uncased"})
```
```sh
python train_tokenizer.py --vocab_size=32000 --cased=True ...
```




Parameters/configurations that can be used in ``prepare_data.py``:

```python
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
```
```sh
python prepare_data.py --block_size=512 --seed=1881 --tokenizer_type=hf ...
```




Parameters/configurations that can be used in ``prepare_random_word_set.py``:

```python
@dataclass
class RandomWordSetConfig:
    limit_for_token_group: int = field(default=5, metadata={"description": "Limit for token group"})
    max_word_limit_for_token_group: int = field(default=5_000, metadata={"description": "Max word limit for token group"})
    min_freq_for_words: int = field(default=150, metadata={"description": "Min freq for words"})
    random_sample: bool = field(default=False, metadata={"description": "Randomly sample words (not directly most frequent ones)"})
    use_number_of_line: int = field(default=None, metadata={"description": "Use specified number of lines for training (if None then use all lines) use small number, otherwise it will take a long time!"})
    tokenizer_type: str = field(default="custom", metadata={"description": "Tokenizer type, random_word_set.json prepared/tokenized with. choices: [custom, hf]"})
```
```sh
python prepare_random_word_set.py --block_size=512 --seed=1881 --tokenizer_type=hf --use_number_of_line=10000 ...
```




Parameters/configurations that can be used in ``pretrain_bert.py`` (BertConfig and PreTrainBertConfig):

```python
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
```



**Note**: The dataset must already be prepared using **prepare_data.py** with the specified "*block_size*" or "*tokenizer_type*" parameters.


**Scenario for creating and training a custom model from scratch**: The user can provide any parameter other than the following parameters ``["do_eval_from_best_ckpt", "do_eval_from_huggingface", "tokenizer_type", "resume"]``:
```sh
python pretrain_bert.py --block_size_s1=128 --block_size_s2=512 \
                        --grad_accum_steps=5 --tokenizer_type=custom \
                        --train_batch_size_s1=64 --train_batch_size_s2=16 \
                        --vocab_size=15000 --num_hidden_layers=8 --hidden_size=512 ...
```




**Scenario for continuing training from the last saved model**: The user can provide any ``PreTrainBertConfig`` parameter other than the following ``["do_train_custom", "do_eval_from_best_ckpt", "do_eval_from_huggingface", "max_ckpt", "tokenizer_type"]``. The provided parameters override the pretrain_cfg of the latest checkpoint. However, no ``BertConfig`` parameters can be provided; the model configuration is directly taken from the last checkpoint and used:

```sh
python pretrain_bert.py --resume=True ...
```



**do_eval_from_best_ckpt scenario**: The user can only provide these parameters ``["do_eval_from_best_ckpt", "val_block_size", "val_batch_size", "device", "max_eval_steps", "seed", "generate_samples", "tokenizer_type"]`` (**Note**: some parameters are fixed, so different values cannot be provided, e.g., ``--tokenizer_type=custom`` must be used):

```sh
python pretrain_bert.py --do_eval_from_best_ckpt=True --val_block_size=512 ...
```



**do_eval_from_huggingface scenario**: The user can only provide these parameters `["do_eval_from_huggingface", "val_block_size", "val_batch_size", "device", "max_eval_steps", "seed"]` (**Note**: some parameters are fixed, so different values cannot be provided, e.g., ``--tokenizer_type=hf`` must be used):

```sh
python pretrain_bert.py --do_eval_from_huggingface=True --val_batch_size=32 ...
```




TODO
----------

* For AMP, bfloat16 precision is specified, but on GPUs that do not support AMP, float32 will be used automatically. For such hardware, float16 should be the default choice, and of course, gradscaler should be used for this case.

* A separate and cleaner interface type can be designed to directly use the model (currently, this is possible by manually modifying example samples in pretrain_bert.py).

* Make the train-val split ratio parameterized/configurable. (Currently, the system only separates the last shard for validation.)

* Besides custom model creation, only BERTurk HF weights are available. Other HF weights should also be usable (for example, the classic Google BERT weights should be usable, so "hf_google_bert", "hf_berturk", etc., instead of just "hf").

* To eliminate class imbalance, the system should calculate in advance the number of documents in the dataset and the number of samples to be extracted, and adjust the randomB ratio (default was 0.5) accordingly. This way, the sample count for `isNext` and `notNext` can be kept fixed at a 0.5 ratio.

* DDP (Data Distributed Parallel) can be implemented.

* Deploy the trained model (to HuggingFace).

* Fine-tune a pre-trained model on different tasks (sentiment analysis, NER, QA, POS, etc.) and evaluate the pretraining process better (benchmark tasks, datasets, etc., can be used).

* Compile (there were some issues compiling the model on Windows as the relevant current library doesn't have a direct distribution for Windows. If the problems can be resolved (e.g., using WSL, running on Linux, or if a wheel is released that allows the latest version of the library to work properly on Windows), compilation can be attempted, which would significantly speed up the training).





----------

**Author:** *Muhammet Can Gümüşsu*

🔗 [LinkedIn Profilim](https://www.linkedin.com/in/muhammet-can￾g%C3%BCm%C3%BC%C5%9Fsu-876041174/)






