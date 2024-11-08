"""pretraining (mlm, nsp) script for BERT model (BertForPreTraining)"""

import os
import sys
import time
import math
import warnings
from dataclasses import asdict

import mlflow
import torch

from model.model import BertForPreTraining
from model.model_aux import FillMaskPipeline, IsNextPipeline, load_checkpoint, save_checkpoint, get_last_ckpt_idx
from data.data_aux import DataLoaderCustom, get_tokenizer
from config import get_pretrain_bert_py_configs



warnings.filterwarnings("ignore")






# @dataclass
# class PreTrainBertConfig:
#     do_eval: bool = False                   # just do eval the model (from_best_ckpt or from_huggingface) but not training (sadece block512 için çalışacak bu arada)
#     from_best_ckpt: bool = False            # cannot be used with do_train
#     from_huggingface: bool = False          # cannot be used with do_train
#     resume: bool = False                    # resume training from the last step
#     block_size: int = 256
#     train_batch_size: int = 32              
#     val_batch_size: int = 8
#     grad_accum_steps: int = 1               # total batch size (we are simulating this by grad accum) -> 8x32 = 256 
#     max_learning_rate: float = 1e-4
#     min_learning_rate: float = max_learning_rate * 0.01
#     lr_scheduler: str = "cosine"
#     num_train_steps: int = 100_000
#     num_warmup_steps: int = 10_000
#     save_checkpoints_steps: int = 1000      # her val'da ckpt işini halledeceğim
#     val_check_interval: int = 1000          # 1000 adımda bir validation yap
#     device: str = "cpu"                     # cpu or cuda (or mps) bakılacak: şimdilik bunun bir etkisi olmayacak, script device'ı otomatik kendi belirleyecek
#     max_eval_steps: int = 100               # istesem'de 100'ü aşamam gibi çünkü: 256 block_size'da son shard'ta toplam 3872 tane sample var, max eval 122'de bu aşılır, block 512'de bu 8256 sample var
#     weight_decay: float = 0.01
#     max_ckpt: int = 10
#     seed: int = 13013
#     generate_samples: bool = True   


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
# en son adım da da validation yapılmalı (last_step flag)
# her val'da reset'lemeyi unutma
# edge case: max_eval step kontrol edilmeli neden? -> modeli validate ederken val split üzerinde aynı example'ları görmememiz gerekiyor (circular sistemin olmaması gerek ya da başta hesap yap ve max_eval'ı ona göre kabul et ya da treshle idk)
# --devamı--> çünkü modeli validate ederken aynı sample'ları görmemizin hiç bir anlamı yok (gereksiz işlem kaybı dışında) [bence circula izin vermesin, eval adım ismi "max_eval_steps" buna izin verir mantıken]
# loss history tutulabilir, 1M adım bile olsa en fazla ramde 3mb yer kaplıyor
# her valdan sonra modeli spesifik bir (ya da daha fazla) example üzerinde çalıştır, ve printle
# losslar, loglar vs bir txt dosyasına appendlenebilir!
# model eğitiminde reproductibilite kontrolü yap, seed vs ile (aynı değerler, losslar vs oluyormu)
# tüm yorumlara bak tr-eng farketmeksizin
# console'daki warning'lerden kurtulmaya çalış (gerekirse geçici kapat)
# seed olayıda ckpt'de olsa mı acaba?
# last step olaylarında sıkıntı var düzenlenecek (son adım ckpt yapılabilmeli, hali hazırda eğitimi bitmiş bir modelde pretrain dediğimizde de ckpt yapıyor (en sonu override ediyor) bu problem çözülmeli)
# has attrib ile gelu yerine farklı actv kullanabilmeli kullanıcı
# do_eval'ın tam anlamıyla çalışabilmesi için data.py'ı güncellemem gerek (tüm adımları hf tokenizer ile de yapabilmeli,
# -> shard count limit belirleyebilmeli(gerçi buna gerek olmadan data.py üzerinde prepare data for do_eval parametresi yapabiliriz))
# her validation'da best ckpt yapılmamalı (fazla vakit yiyor, her ckpt yapıldığı zaman yapılmalı bence)
# git repo işlerine geldiğinde göreceksin: xy_shards256 klasörünü (stat dosyası) track etmişim zamanında, track edilen dosyaları gitignore etsende hala track edilmeye devam edilir unutma!
# -> * (asterisk) kullandığın zaman '!' sembolü çalışmaya başladı!  (ÖNEMLİ)
# mlflow autolog'u dene
# takip edilmemesi gereken dosyaları takipten çıkarmayı da unutma!
# cli config comb vs işleri (device kontrolü de yapılmalı)
# run_id olayı yaptım resume ile scratch case lerini dene (resume olunca mlflow'da o run'ın devam etmesini istiyorum, scratch'de apayrı yeni bir run açılsın)
# unutma repoda da belirt! : hf modeli sadece evaluation için var (ki onun tensorlarini aldık sadece), alıp hf modelini eğitme yada resume etme ne ise bunu yapmadım (ama yapması çok kolay dur la belki yapabilirim: tokenizer/embedding olayına bi de generation olayına dikkat edecem)



# tanım: gpu'daki operasyonlara/kernel'lara mm operasyon precisionunu ayarlıyor
# yaklaşık x2 gain/hızlı oldu 3060rtx'de. Dikkat! actv ve parametreler hala float32
# "high" parametresi ile mm operasyonları tf32'a dönüşüyor (ismine aldanma
# precision bit azalıyor) (operasyon esnasında tf32)
torch.set_float32_matmul_precision("high")

# create training config (BAKILACAK: şimdilik bu ikisi default configler!)
model_cfg, pretrain_config = get_pretrain_bert_py_configs(verbose_all=True, verbose_changes=True)

# dataclass objects are not subscriptable, so convert to dict (i want subscription beacause it's more readable than property access in vscode)
model_cfg = asdict(model_cfg)
pretrain_config = asdict(pretrain_config)

if pretrain_config["generate_samples"]:
    samples_for_mlm_generation = ["Merhaba [MASK] efendi nasıl gidiyor?",
                                  "ışınlanma teknolojisini [MASK] Can  bulmuştur. Muhammet Can bu buluş ile büyük alkış topladı"]

    samples_for_nsp_generation = [["Çocuk sahibi çift sayısında inanılmaz bir artış var.", "Uzaya ilk Bekiroğ Reis çıkmıştır"],
                                  ["Bu nsp olayı bir garip oldu. Sanki kablolar ters bağlandı.", "Bir şekilde kabloları ayarlamak gerekli"],
                                  ["çoğunuz yaş itibariyle tanımaz ama istanbul'un en iyi belediye başkanı justinianus'tur.", "Samsun Ayvacık belediyesi ne yapmak nereye varmak istemektedir!"] ]



device = pretrain_config["device"]
print(f"using device: {device}")


torch.manual_seed(pretrain_config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(pretrain_config["seed"])



def do_just_evaluation_on_pretrain_tasks(model: BertForPreTraining, val_loader: DataLoaderCustom):

    tokenizer = get_tokenizer(tokenizer_postfix="hf" if pretrain_config["from_huggingface"] else pretrain_config["tokenizer_type"])
    model.eval()
    val_loader.reset()
    fill_mask_pipeline = FillMaskPipeline(model, tokenizer, strategy="greedy")
    nsp_pipeline = IsNextPipeline(model, tokenizer)
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(pretrain_config["max_eval_steps"]):
            model_input_batch = val_loader.next_batch()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                model_output = model(**model_input_batch)

            val_loss_accum += model_output.loss.detach().item() / pretrain_config["max_eval_steps"]

        # bakılacak modularleştirilebilir
        # bakılacak, sampling/geration/pipeline aşamaları, her task için (mlm, nsp) ayrı ayrı 5 example ve sonuçlarını bas
        print("\nMLM GENERATION EXAMPLES:")
        print("---------------------------------------")
        fill_mask_pipeline(samples_for_mlm_generation)
        print("NSP GENERATION EXAMPLES:")
        print("---------------------------------------")
        nsp_pipeline(samples_for_nsp_generation)
        print("---------------------------------------")

        print(f"validation loss: {val_loss_accum}\n")



# just do evaluation and exit (değiştirdim)
if pretrain_config["do_eval_from_huggingface"] == True or pretrain_config["do_eval_from_best_ckpt"] == True:
    val_loader = DataLoaderCustom(batch_size=pretrain_config["val_batch_size"], block_size=pretrain_config["val_block_size"],
                                  split="val", tokenizer_type=pretrain_config["tokenizer_type"], device=device)
    if pretrain_config["do_eval_from_huggingface"] == True:
        model = BertForPreTraining.from_pretrained()
    elif pretrain_config["do_eval_from_best_ckpt"] == True:
        ckpt_dict = load_checkpoint(postfix="best")
        model = BertForPreTraining(ckpt_dict["model_config"])
        model.load_state_dict(ckpt_dict["model_state_dict"])
    model.to(device)
    model.eval()
    do_just_evaluation_on_pretrain_tasks(model, val_loader)
    # bakılacak printle bişiler bitti falan de
    print("done, exiting...")
    sys.exit(0)



if not os.path.exists("model/model_ckpts"):
    os.mkdir("model/model_ckpts")

ckpt_files = os.listdir("model/model_ckpts")

best_val_loss = math.inf
last_ckpt_idx = 0
last_step = 0

# resume pretraining from last checkpoint
if pretrain_config["resume"] == True:

    last_ckpt_idx = get_last_ckpt_idx()    
    ckpt_dict = load_checkpoint(postfix=last_ckpt_idx)
    # bakılacak, daha güzel yap
    print(f"\nresume from ckpt: {last_ckpt_idx}\n")
    last_step = ckpt_dict["last_step"]
    best_val_loss = ckpt_dict["best_val_loss"]
    mlflow_run_id = ckpt_dict["mlflow_run_id"]
    model_config = ckpt_dict["model_config"]
    model = BertForPreTraining(model_config)
    model.load_state_dict(ckpt_dict["model_state_dict"])
    model.to(device)

    # create and load optimizer state
    optimizer = model.configure_optimizers(weight_decay=pretrain_config["weight_decay"], learning_rate=pretrain_config["max_learning_rate"], device=device)
    optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
    # val set baştan sonra 512 block size olarak kalacak
    val_loader = DataLoaderCustom(batch_size=pretrain_config["val_batch_size"], block_size=pretrain_config["val_block_size"],
                                  split="val", tokenizer_type=pretrain_config["tokenizer_type"], device=device)

    # create and load data loader state
    if last_step > pretrain_config["num_train_steps"] * pretrain_config["stage1_ratio"]:
        train_loader = DataLoaderCustom(batch_size=pretrain_config["train_batch_size_s2"], block_size=pretrain_config["block_size_s2"],
                                        split="train", tokenizer_type=pretrain_config["tokenizer_type"], device=device)
    else:
        train_loader = DataLoaderCustom(batch_size=pretrain_config["train_batch_size_s1"], block_size=pretrain_config["block_size_s1"],
                                        split="train", tokenizer_type=pretrain_config["tokenizer_type"], device=device)
    # val position state is not important, we are resetting it anyway
    train_loader.load_state_dict(ckpt_dict["last_dataloader_state"])



# train from scratch
elif (pretrain_config["resume"]) == False:
    if len(ckpt_files) > 0:
        while True:
            response = input("Ckpt files are already exist. They will be overwritten by 'train from scratch', are you sure? (Y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("deleting all ckpt files ...")
                # delete all ckpt files
                for file in ckpt_files:
                    os.remove("model/model_ckpts/" + file)

                # bakılacak, daha güzel info verilecek
                print("Training starts with {}...".format("randomly initialized" if pretrain_config["do_train_custom"] else "hf model weights"))
                break 
            elif response in ['n', 'no']:
                # bakılacak, çoğu şeyi fonksiyonal hale getirdikten sonra:
                # sys exit yapmak yerine resume için sorgu alınabilir
                print("exiting...")
                sys.exit(0)
            else:
                print("invalid input...")

    # create model, optimizer, dataloaders "from scratch" 
    model = BertForPreTraining(model_cfg) if pretrain_config["do_train_custom"] else BertForPreTraining.from_pretrained()
    model.to(device)
    optimizer = model.configure_optimizers(weight_decay=pretrain_config["weight_decay"], learning_rate=pretrain_config["max_learning_rate"], device=device)
    train_loader = DataLoaderCustom(batch_size=pretrain_config["train_batch_size_s1"], block_size=pretrain_config["block_size_s1"],
                                    split="train", tokenizer_type=pretrain_config["tokenizer_type"], device=device)
    # val set baştan sonra 512 block size olarak kalacak
    val_loader = DataLoaderCustom(batch_size=pretrain_config["val_batch_size"], block_size=pretrain_config["val_block_size"],
                                  split="val", tokenizer_type=pretrain_config["tokenizer_type"], device=device)


if pretrain_config["generate_samples"]:
    tokenizer = get_tokenizer(tokenizer_postfix="hf" if pretrain_config["do_train_custom"] else pretrain_config["tokenizer_type"])
    fill_mask_pipeline = FillMaskPipeline(model, tokenizer, strategy="greedy")
    nsp_pipeline = IsNextPipeline(model, tokenizer)



if pretrain_config["mlflow_tracking"]:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("bert_implementation_tr")
    # every run should have the same model training whether from scratch or resume! (if mlflow tracking is enabled)
    # if resume, we need to use last ckpt run id to continue tracking consistently (we kept it in ckpt files) 
    if (pretrain_config["resume"]):
        mlflow.start_run(run_id=mlflow_run_id)
    else:
        # if scratch, we need to create a new run and keep its run id
        mlflow_run_id = mlflow.start_run(run_name="bert_pretraining").info.run_id
    mlflow.log_params(pretrain_config)
    mlflow.log_params(asdict(model.config))
else:
    # we need this whether we use mlflow or not to keep things working
    mlflow_run_id = None    




max_lr = pretrain_config["max_learning_rate"]
min_lr = pretrain_config["min_learning_rate"]
warmup_steps = pretrain_config["num_warmup_steps"]
max_steps = pretrain_config["num_train_steps"]
schedular_type = pretrain_config["lr_scheduler"]

def get_lr(it):
    if schedular_type != "cosine":
        raise NotImplementedError(f"schedular type {schedular_type} not implemented yet...")
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





# bakılacak, mlflow kullanacağım için bunlara gerek yok bence
# val_lo = 0.0
# train_loss_history = 0.0
grad_accum_steps = pretrain_config["grad_accum_steps"]

print("seed: ", pretrain_config["seed"])
print("train batch size: ", train_loader.batch_size)
print("trainblock size: ", train_loader.block_size)
print("val batch size: ", val_loader.batch_size)
print("val block size: ", val_loader.block_size)




# 1 additional loop for validation
max_steps += 1
train_loss_accum = 0.0
val_loss_accum = 0.0

# training loop
for step in range(last_step, max_steps):  
    t0 = time.time()
    last_step_flag = (step == (max_steps - 1))


    # validation loop
    # ilk adımda da validation yapacağına dikkat! (scratch modelin (randomly initialized) ilk adımdaki performansını inceleyebiliriz)
    # ayrıca en son adımda da validation yapılacağına dikkat

    #if step > last_step and step % pretrain_config["val_check_interval"] == 0 or last_step_flag:
    if step % pretrain_config["val_check_steps"] == 0 or last_step_flag:
      
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(pretrain_config["max_eval_steps"]):
                model_input_batch = val_loader.next_batch()
                # val shard' komple gezildiği durumda bir sonraki aşamaya geç (val'da cycle olayı yok, gereksiz)
                if model_input_batch is None:
                    break
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    model_output = model(**model_input_batch)

                val_loss_accum += model_output.loss.detach().item() / pretrain_config["max_eval_steps"]
        
        if step > last_step and pretrain_config["mlflow_tracking"]:
            mlflow.log_metric("val_loss", val_loss_accum, step)
        
        
        

        

        # if step > last_step and val_loss_accum < best_val_loss:
        # save ckpt if val loss improved
        if step > last_step and val_loss_accum < best_val_loss:
            # update bast_val_loss
            best_val_loss = val_loss_accum

            #save_checkpoint(model, optimizer, train_loader, step, best_val_loss, train_loss_history, val_loss_history, "best")
            save_checkpoint(model, optimizer, train_loader, step, best_val_loss, "best", pretrain_config, mlflow_run_id=mlflow_run_id)
            # bakılacak, düzenlenebilir ya da komple kaldırılabilir
            print("best ckpt is updated...")

        # bakılacak modularleştirilebilir
        if pretrain_config["generate_samples"]:
            txt = f"\n[Step:{step}] MLM GENERATION EXAMPLES:\n"
            txt += "---------------------------------------\n"
            txt += fill_mask_pipeline(samples_for_mlm_generation, do_print=False) + "\n"
            txt += f"[Step:{step}] NSP GENERATION EXAMPLES:\n"
            txt += "---------------------------------------\n"
            txt += nsp_pipeline(samples_for_nsp_generation, do_print=False) + "\n"
            txt += "---------------------------------------\n"
            print(txt)
            with open("generated_samples.txt", "a", encoding="utf-8") as f:
                f.write(txt)

            mlflow.log_artifact("generated_samples.txt") if pretrain_config["mlflow_tracking"] else None

        print("val loss: ", val_loss_accum)

        # return model to train mode after validation
        model.train()
        
        



    # save ckpt every save_checkpoints_steps
    if step > last_step and step % pretrain_config["save_checkpoints_steps"] == 0 or last_step_flag:
        if (last_ckpt_idx + 1 > pretrain_config["max_ckpt"]):
            # we will remove the oldest checkpoint
            print(f"override old ckpt: {last_ckpt_idx - pretrain_config['max_ckpt']} with current ckpt: {last_ckpt_idx}")
            os.remove(f"model/model_ckpts/BertForPretraining_{last_ckpt_idx - pretrain_config['max_ckpt']}.pt")
        # bakılacak, mlflow ile track edeceğim için bunlara gerek yok bence
        # save_checkpoint(model, optimizer, train_loader, step, best_val_loss, train_loss_history, val_loss_history, last_ckpt_idx + 1)
        save_checkpoint(model, optimizer, train_loader, step, best_val_loss, last_ckpt_idx, pretrain_config, mlflow_run_id=mlflow_run_id)
        # bakılacak, düzenlenebilir ya da komple kaldırılabilir
        print(f"ckpt:{last_ckpt_idx} saved...")
        # next ckpt idx
        last_ckpt_idx += 1


    # son adımda validation yaptık, ckpt işlerini hallettik, çıkıyoruz (tekrar eğitim döngüsüne girmeye gerek yok)
    if last_step_flag:
        print("training done...")
        mlflow.end_run() if pretrain_config["mlflow_tracking"] else None
        break
    

    # step >= (max_steps * 0.9)
    # eğitimin son %10'lık dilimi: 512 block_size
    # için hazırlık, init
    if step == (int(max_steps * pretrain_config["stage1_ratio"])):
        # bakılacak burada bir tür print yapılamlı info verilmeli (artık bunu tqdm'den mi anlarız yoksa saf print ile mi idk)
        print(f"Switching to stage 2, block_size: {pretrain_config['block_size_s2']}, batch_size: {pretrain_config['train_batch_size_s2']}")
        train_loader = DataLoaderCustom(batch_size=pretrain_config["train_batch_size_s2"], block_size=pretrain_config["block_size_s2"],
                                        split="train", tokenizer_type=pretrain_config["tokenizer_type"], device=device)

    

    train_loss_accum = 0.0
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
        train_loss_accum += model_output.loss.detach()
        model_output.loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()


    # wait for all kernels (gpu operations/processes) to finish
    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.batch_size * train_loader.block_size * pretrain_config["grad_accum_steps"]
    tokens_per_second = tokens_processed / dt

    if pretrain_config["mlflow_tracking"]:
        mlflow.log_metric("total_train_loss", train_loss_accum, step)
        mlflow.log_metric("lr", lr, step)
        mlflow.log_metric("grad_norm", norm, step)
        mlflow.log_metric("mlm_loss", model_output.mlm_loss.item(), step)
        mlflow.log_metric("nsp_loss", model_output.nsp_loss.item(), step)
        mlflow.log_metric("tokens_per_second", tokens_per_second, step)

    print_txt = f"step {step:5d} | total loss: {train_loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | mlm loss: {model_output.mlm_loss.item():.6f} | nsp loss: {model_output.nsp_loss.item():.6f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f} tokens/sec"
    
    # log to file whether or not mlflow is enabled
    with open("log.txt", "a", encoding="utf-8") as f:
                f.write(print_txt)

    print(print_txt)



    
    
    




