"""pretraining (mlm, nsp) script for BERT model (BertForPreTraining)"""

import os
import sys
import time
import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as F



from model import BertConfig, BertForPreTraining
from bert_implementation_tr.data.data_aux import (
    ModelInput,
    Stat,
    DataloaderCustomState,
    DataLoaderCustom,
    load_xy_shard,
    get_last_shard_idx
)



@dataclass
class PreTrainBertConfig:
    do_train: bool = False
    do_eval: bool = False
    resume: bool = False
    block_size: int = 256
    train_batch_size: int = 32
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1 # total batch size (we are simulating this by grad accum) -> 8x32 = 256 
    learning_rate: float = 1e-4
    num_train_steps: int = 100_000
    num_warmup_steps: int = 10_000
    save_checkpoints_steps: int = 1000
    max_eval_steps: int = 100
    weight_decay: float = 0.01
    max_ckpt_count: int = 10
    


# yaklaşık 40 epoch eğitim yapılmış (toplam: 128,000,000,000 token)
# 1 epoch o zaman 3,200,000,000 token'a denk gelir
# bendeki toplam token sayısı (yani 1 epoch) -> 350,000,000
# bu durumda original bert paper veri seti benimkinden ~10x daha büyük
# original paper'da normal bir adım için block_size'ı 512 belirlemiş (yani hesaplamalarını ona göre yapmış, bende öyle yapacağım (vazgeçtim en alta bak))
# buna göre original bert paperdaki bazı hyperparametreleri kendi veri setim ile uygun hale getirmek için 10'a böl.
# 1,000,000 max_steps -> 100,000 max_steps olacak 
# 10,000 warmup steps -> 1,000 warmup steps olacak
# 1000 save checkpoints steps -> 100 save checkpoints steps olacak (son 10 ya da 20 ckpt kaydedilsin, bunu araştırayım) (early stop vs belki yapabilirim bilmiyorum kind of, keza reducelronplato bakabilirm)
# ckpt mekanizması geliştirmeli (save-load) (her val aşaması sonunda galiba idk)
# pretrain resume olayına bakmalı
# min_lr = max_lr * 0.1
# b_size'ı 256'ya simüle etmelisin, etmezsen yaklaşık 850mil token görülür yani ~2 epoch olmuş olur eğitim
# böylece kendi veri setimdede 40 epoch uygulamış olacağım (aslında yaklaşık olarak 18 epoch olacak : (256 B * 256 T * 100,000 adım) / 350,000,000 tokens)
# eğitim esnasında plotlar vs anlık görülebilmeli (idk how to)
# %90, %10 olayını da yapacağım
# weighted loss yapabilirim idk (yapmalı gibiyim sanırsam stat öyle diyordu! özellikle bs 512 old'da, mismatch olmaması için dataloader'ın weight'leri bulması daha iyi olaiblir (bs512'nin w leri bs256'ya göre çok daha farklı))
# train-val set ayarlamaları, kaç adımda val, val bs vsvs ayarlanacak
# tüm eğitim config ile ayarlansın (magic number gözükmesin)
# eğitim sürerken tahmini kalan süreyi de basmalı
# eğitim verbose/print olaylarını ayarla (başta gerekli bilgiler, eğitim boyuncaki bilgiler vs)
# eğitim ilerledikçe printlerin konsol dışına çıkma ihtimali vs var nabaruk ne ederük (belki yüzdesel hale getirebiliriz)
# keza %10 girince yeni dataloader olusturulacak bu da constructure aşamasında print yapıyor kalabilir kalmayabilri idk bak ayarla
# 1 ckpt fazladan olsun (val'da en başarılı olan model/ckpt)
# commit yap sonra tüm bu yorumları vs sil temiz gözüksün
# net fonksiyonel hale geçirmeli herşeyi
# device pretrain'de param olamaz (dev aşamasında daha iyi debug için arada kendin zorla geçirebilirsin)
# neden xyshard512'de notNext bu kadar fazla? (bunu kanıtladım, github'ta bundan belirtmem gerek) [şimdi baktım da 256bs'de de baya aynı durum weighted loss şart olur]
# son shardların sample sayısına bakmalı
# BUNU CV.TXT'E DE YAZ ÖNEMLİ: bakılacak: data.py'da doc shuffle yapılmalı mı yaptım mı vs? (dikkat önemli konu bir tür class imbalance aslında, aynı anda 3 kitap okuma analojisi)


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


# tanım: gpu'daki operasyonlara/kernel'lara mm operasyon precisionunu ayarlıyor
# yaklaşık x2 gain/hızlı oldu 3060rtx'de. Dikkat! actv ve parametreler hala float32
# "high" parametresi ile mm operasyonları tf32'a dönüşüyor (ismine aldanma
# precision bit azalıyor) (operasyon esnasında tf32)
torch.set_float32_matmul_precision("high")

# create training config
pretrain_config = PreTrainBertConfig()

# create model (randomly initialized)
default_cfg = BertConfig()
model = BertForPreTraining(default_cfg)
model.to(device)

# create optimizer
optimizer = model.configure_optimizers(weight_decay=pretrain_config["weight_decay"], learning_rate=pretrain_config["learning_rate"], device=device)
train_loader = None # DataLoaderCustom(batch_size=B, block_size=T, device=device, verbose=False) # allta her türlü oluşturulacak verbose'lu şekilde ondan dolayı burda None yapalım


last_ckpt_idx = 0
last_step = 0
ckpt_files = os.listdir("bert_implementation_tr/model_ckpts")
if pretrain_config["resume"] == True:
    if len(ckpt_files) == 0:
        raise ValueError("no ckpt found to resume")
    
    for idx, post_fix in enumerate(list(map(lambda x: x.split("_")[-1], ckpt_files))):

        if post_fix == "best":
            continue

        last_ckpt_idx = int(post_fix) if int(post_fix) > last_ckpt_idx else last_ckpt_idx

    last_step, data_loader_state = model.load_checkpoint(idx=last_ckpt_idx, optimizer=optimizer)
    train_loader = DataLoaderCustom.from_state(data_loader_state, device=device)
    # bakılacak, daha güzel yap
    print(f"resume from {last_ckpt_idx} ckpt")
elif pretrain_config["resume"] == False:
    if len(ckpt_files) > 0:
        while True:
            response = input("Ckpt files are already exist. They will be overwritten to train from scratch, are you sure? (Y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("deleting all ckpt files ...")
                # delete all ckpt files
                for file in ckpt_files:
                    os.remove(file)
                # bakılacak, daha güzel info verilecek
                print(f"from scratch")
                train_loader = DataLoaderCustom(batch_size=pretrain_config["train_batch_size"], block_size=pretrain_config["block_size"], device=device)
                break 
            elif response in ['n', 'no']:
                # bakılacak, çoğu şeyi fonksiyonal hale getirdikten sonra:
                # sys exit yapmak yerine resume için sorgu alınabilir
                sys.exit(0)
            else:
                print("invalid input...")
else:
    raise ValueError("invalid resume arg...")



B, T = pretrain_config["train_batch_size"], pretrain_config["block_size"]
grad_accum_steps = pretrain_config["gradient_accumulation_steps"]

total_batch_size = B * grad_accum_steps
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


max_lr = pretrain_config["learning_rate"]
min_lr = max_lr * 0.1
warmup_steps = 10 # pretrain_config["num_warmup_steps"]
max_steps = 50 # pretrain_config["num_train_steps"]
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters (which is max_steps in our case), return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)



    

# training loop
for step in range(last_step, max_steps):  
    t0 = time.time()


    # step >= (max_steps * 0.9)
    # eğitimin son %10'lık dilimi: 512 block_size
    # için hazırlık, init
    if step == (int(max_steps * 0.9)):
        # bakılacak burada bir tür print yapılamlı info verilmeli (artık bunu tqdm'den mi anlarız yoksa saf print ile mi idk)
        print("hello, this is the end of 90% of training, now starting 10% of training B 512")
        train_loader = DataLoaderCustom(batch_size=B//2, block_size=T*2, device=device)

    
    loss_accum = 0.0
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        model_input_batch = train_loader.next_batch()

        # A100 (ampere architecture) ve sonrası (3060rtx de bu klasmanda) bfloat16 kullanabiliyor
        # bu sayede gradscale kullanmaya gerek yok (float16'da gradscale gerekli)
        # artık actv dtype'ları bfloat16 (yarı yarıya bir düşüş (tabi actv için paramlar klasik float32))
        # torch.set_float32_matmul_precision("high") etkisi kalmadı bu arada
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            model_output = model(**model_input_batch)

        model_output.loss /= grad_accum_steps
        # loss accum yapmazsak sadece en sonki micro step'in loss'unu basmış oluruz
        # ancak tüm adımların ortalama loss'una ihtiyacımız var
        loss_accum += model_output.loss.detach()
        model_output.loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    # bakılacak step +1 olması son durumdaki ckpt'yi kaydetmemeye neden olabilir mi?
    if (step + 1) % pretrain_config.save_checkpoints_steps == 0:
        model.save_checkpoint(step=step, postfix="BAKILACAK", optimizer=optimizer, dataloader=train_loader)
        # BAKILACAK: val loss'a vs göre ayrıca en iyi modeli kaydet

    # wait for all kernels (gpu operations/processes) to finish
    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = B * T * grad_accum_steps
    tokens_per_second = tokens_processed / dt
    print(f"step {step:5d} | total loss: {loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | mlm loss: {model_output.mlm_loss.item():.6f} | nsp loss: {model_output.nsp_loss.item():.6f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f} tokens/sec")


    
    
    




