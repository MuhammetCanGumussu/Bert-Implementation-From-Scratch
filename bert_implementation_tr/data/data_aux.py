"""
ab_raw.txt
ab_dataframe [A, B, isNext]
ab_dataframe [A, B, isNext, A_tokenized, B_tokenized, A_word_ids, B_word_ids, A_token_len, B_token_len]
total_token_num

ab_block1_df [A, B, isNext, A_tokenized, B_tokenized, A_word_ids, B_word_ids, A_token_len, B_token_len]
ab_block2_df [A, B, isNext, A_tokenized, B_tokenized, A_word_ids, B_word_ids, A_token_len, B_token_len]
total_token_num, excluded_token_num


xy_block1_df [x, att_mask, token_type_ids, y] : shards (#token_per_shard must be defined, numpy, dtype important)
xy_block2_df [x, att_mask, token_type_ids, y] : shards

visualization tool

"""
import pandas as pd

BLOCK_1_SIZE = None
BLOCK_2_SIZE = None


def visualize_xy(): # default random
    pass


def tokenize_ab(ab_dataframe):
    pass 


def split_ab(ab_dataframe):
    pass


def create_xy_shards(ab_block1_df, ab_block2_df, random_words_dict):
    pass



def convert_ab2xy(ab_string_path, random_words_dict):
    ab_dataframe = pd.read_csv(ab_string_path, sep='[SEP]', names=["A", "B", "isNext"])
    ab_dataframe, total_num_tokens = tokenize_ab(ab_dataframe) # [A, B, isNext, A_tokenized, B_tokenized, A_word_ids, B_word_ids, A_token_len, B_token_len]
    ab_block1_df, ab_block2_df, block1_num_tokens, block2_num_tokens = split_ab(ab_dataframe)
    del ab_dataframe

    excluded_num_tokens = total_num_tokens - (block1_num_tokens + block2_num_tokens)
    print(f"total_num_tokens: {total_num_tokens}, block1_num_tokens: {block1_num_tokens}, block2_num_tokens: {block2_num_tokens}, excluded_num_tokens: {excluded_num_tokens}")

    create_xy_shards(ab_block1_df, ab_block2_df, random_words_dict)
    pass
