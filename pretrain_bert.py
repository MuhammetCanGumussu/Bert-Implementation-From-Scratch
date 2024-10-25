"""pretraining (mlm, nsp) script for BERT model (BertForPreTraining)"""

from dataclasses import dataclass
import os
import sys
import time

import torch
import torch.nn as nn
from torch.nn import functional as F



from model import BertConfig, BertForPreTraining
from bert_implementation_tr.data.data_aux import (
    ModelInput,
    Stat,
    load_xy_shard,
    get_last_shard_idx
)

import torch._dynamo

torch._dynamo.config.suppress_errors = True


@dataclass
class PreTrainBertConfig:
    block_size: int = 256
    do_train: bool = False
    do_eval: bool = False
    train_batch_size: int = 32
    eval_batch_size: int = 8
    learning_rate: float = 1e-4
    num_train_steps: int = 100000
    num_warmup_steps: int = 10000
    save_checkpoints_steps: int = 1000
    iterations_per_loop: int = 1000
    max_eval_steps: int = 100


class DataLoaderCustom:
    def __init__(self, batch_size: int, block_size: int, device: str = "cpu"):
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size
        self.data_dir = f"bert_implementation_tr/data/xy_shards_{self.block_size}"
        self.num_shards = get_last_shard_idx(self.data_dir)
        self.stat = Stat.from_file(f"{self.data_dir}/stat.txt")

        assert self.num_shards != -1 , f"no shards found in {self.data_dir}"

        # some stats
        print(f"block size: {self.block_size}")
        print(f"total number of shards: {self.num_shards}")
        print(f"total number of tokens: {self.stat.total_number_of_token}")
        print(f"total number of samples: {self.stat.total_number_of_sample}")
        print("-------------------------------------------------------------------------")
        print(f"1 batch: {self.block_size * self.batch_size} tokens")
        print(f"1 epoch: {self.stat.total_number_of_token // (self.block_size * self.batch_size)} batches")
        print(f"1 epoch: {self.stat.total_number_of_token} tokens")
        print(f"1 shard: ~{self.stat.total_number_of_token // self.num_shards} tokens")
        print(f"1 shard: ~{(self.stat.total_number_of_token // self.num_shards) // (self.block_size * self.batch_size)} batches")
        print("-------------------------------------------------------------------------")

        self.reset()

    def reset(self):
        self.current_shard_id = 0
        self.current_shard = ModelInput.from_numpy_to_tensors_dict(
                                        np_array = load_xy_shard(self.current_shard_id, block_size=self.block_size), 
                                        block_size = self.block_size)
        self.current_position_in_shard = 0

    def next_batch(self):
        if (self.current_position_in_shard + self.batch_size) >= len(self.current_shard["input_ids"]):
            model_input = {k: v[self.current_position_in_shard:].to(self.device) for k, v in self.current_shard.items()}
            self.current_shard_id = 0 if self.current_shard_id % self.num_shards == 0 else (self.current_shard_id + 1)
            self.current_shard = load_xy_shard(self.current_shard_id, block_size=self.block_size)
            self.current_shard = ModelInput.from_numpy_to_tensors_dict(self.current_shard, block_size=self.block_size)
            self.current_position_in_shard = 0
            return model_input
        
        model_input = {k: v[self.current_position_in_shard : self.current_position_in_shard + self.batch_size].to(self.device) for k, v in self.current_shard.items()}
        self.current_position_in_shard += self.batch_size

        return model_input




# auto-detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"[INFO] using device: {device}")

# override device
#device = "cpu"

# reproducibility
torch.manual_seed(13013)
if torch.cuda.is_available():
    torch.cuda.manual_seed(13013)

pretrain_config = PreTrainBertConfig()
B, T = pretrain_config.train_batch_size, pretrain_config.block_size

# create data loader
train_loader = DataLoaderCustom(batch_size=B, block_size=T, device=device)

# tanım: gpu'daki operasyonlara/kernel'lara mm operasyon precisionunu ayarlıyor
# yaklaşık x2 gain/hızlı oldu 3060rtx'de. Dikkat! actv ve parametreler hala float32
# "high" parametresi ile mm operasyonları tf32'a dönüşüyor (ismine aldanma
# precision bit azalıyor) (operasyon esnasında tf32)
torch.set_float32_matmul_precision("high")

# create model (randomly initialized)
cfg = BertConfig()
cfg.num_hidden_layers = 2
model = BertForPreTraining(cfg)

# move model to device
model.to(device)
# model = torch.compile(model)
# model = torch.compile(model, backend="cudagraphs")
#torch.compile(m, backend="cudagraphs")

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = pretrain_config.learning_rate)


for i in range(300):
    t0 = time.time()

    model_input_batch = train_loader.next_batch()

    optimizer.zero_grad()

    # A100 (ampere architecture) ve sonrası (3060rtx de bu klasmanda) bfloat16 kullanabiliyor
    # bu sayede gradscale kullanmaya gerek yok (float16'da gradscale gerekli)
    # artık actv dtype'ları bfloat16 (yarı yarıya bir düşüş (tabi actv için paramlar klasik float32))
    # torch.set_float32_matmul_precision("high") etkisi kalmadı bu arada
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        model_output = model(**model_input_batch)
        # mport code; code.interact(local=locals())

    model_output.loss.backward()

    optimizer.step()

    # wait for all kernels (gpu operations/processes) to finish
    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_second = (train_loader.batch_size * train_loader.block_size) / (t1 - t0)
    print(f"step {i}, total loss: {model_output.loss.item():.6}, mlm loss: {model_output.mlm_loss.item():.6}, nsp loss: {model_output.nsp_loss.item():.6}, dt: {dt:.2f}ms, tok/sec: {tokens_per_second:.2f} tokens/sec")





