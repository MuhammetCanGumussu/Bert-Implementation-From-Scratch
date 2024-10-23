"""pretraining (mlm, nsp) script for BERT model (BertForPreTraining)"""



import torch
import torch.nn as nn
from torch.nn import functional as F

from model import BertConfig, BertForPreTraining


# auto-detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")


model = BertForPreTraining(BertConfig())