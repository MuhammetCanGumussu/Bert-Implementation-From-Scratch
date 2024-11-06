"""
Prepare data for 'do_eval: True' in pretrain_bert.py.

Why do we need this?

We need this to evaluate either a custom model or a Hugging Face (HF) model. In both cases, we might need to use different tokenizers.
In data.py, we prepare data using a custom tokenizer, which means we cannot use it for HF model evaluation.
For this reason, this script will prepare data using the specified tokenizer (specifically the HF tokenizer).
ab_shards_{block_size} will be used...

Be careful this script will be executed from pretrain_bert.py by "os.system('python prepare_do_eval tokenizer_name block_size')"

"""

import os
import sys
import argparse




def main():
    tokenizer_name = sys.argv[1]
    assert tokenizer_name in ["from_hf", "custom"], f"tokenizer_name must be 'from_hf' or 'custom' but got {tokenizer_name}"

    block_size = int(sys.argv[2])
    if not os.path.exists(f"ab_shards_{block_size}"):
        raise FileNotFoundError(f"ab_shards_{block_size} is not exists")
    
    # do tokenization in mp fashion




if __name__ == "__main__":
    main()