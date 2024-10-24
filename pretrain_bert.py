"""pretraining (mlm, nsp) script for BERT model (BertForPreTraining)"""



import torch
import torch.nn as nn
from torch.nn import functional as F

from bert_implementation_tr.data.data_aux import ModelInput, load_xy_shard
from model import BertConfig, BertForPreTraining


# auto-detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"[INFO] using device: {device}")

# create model (randomly initialized)
model = BertForPreTraining(BertConfig())
# move model to device
model.to(device)

# get data
np_shard_0 = load_xy_shard(0, block_size=256)
np_batch = np_shard_0[0]   

model_input_batch = ModelInput.from_numpy_to_tensors_dict(np_batch, block_size=256, device=device)

model_output = model(**model_input_batch)
print(model_output.loss)

import sys;


