
import os

from transformers import PreTrainedTokenizerFast


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_tokenizer_file_path():
    import glob
    tokenizer_file = glob.glob(root_dir + f"/tokenizer/tr_wordpiece_tokenizer_*.json")
    if len(tokenizer_file) == 0:
        raise FileNotFoundError(f"there is no tokenizer file in {root_dir + f'/tokenizer'}")
    if len(tokenizer_file) > 1:
        raise ValueError(f"there is more than one tokenizer file in {root_dir + f'/tokenizer'}, please keep only one tokenizer file that you want to use...")
    return tokenizer_file[0]


def get_tokenizer(custom = True):
    """
        tokenizer_postfix can be: [custom_cased, custom_uncased, hf]
    """
    if custom == False:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file = get_tokenizer_file_path(), # You can load from the tokenizer file, alternatively
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        clean_up_tokenization_spaces=True   # default olarak ta True ancak future warning ilerde False olacağını belirtti.
                                            # ilerde problem olmaması için (ve tabiki future warning almamak için) açıkca True yaptık
    )
    return tokenizer