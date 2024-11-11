"""pretraining (mlm, nsp) script for BERT model (BertForPreTraining)"""

# MLFLOW ile resume edememe
# config sistemi güncellenmeli (daha kavramsal, readable olmalı, hatalar giderilmeli)
# test edilmeyen bir sürü case var (hf weigh ile eğitim, eval denenmedi)

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




# tanım: gpu'daki operasyonlara/kernel'lara mm operasyon precisionunu ayarlıyor
# yaklaşık x2 gain/hızlı oldu 3060rtx'de. Dikkat! actv ve parametreler hala float32
# "high" parametresi ile mm operasyonları tf32'a dönüşüyor (ismine aldanma
# precision bit azalıyor) (operasyon esnasında tf32)
torch.set_float32_matmul_precision("high")

# create training config (BAKILACAK: şimdilik bu ikisi default configler!)
model_cfg, pretrain_config = get_pretrain_bert_py_configs(verbose=True)





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
    # bakılacak: config'in doğru tokenizer vermesi lazım
    tokenizer = get_tokenizer(custom=True if pretrain_config["tokenizer_type"] == "custom" else False)
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
    print(f"model is being evaluated...\n")
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




# train from scratch (from randomly initialized or huggingface weights)
else: 
    if len(ckpt_files) > 0:
        while True:
            response = input("Ckpt files are already exist. They will be overwritten by 'train from scratch', are you sure? (Y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("deleting all ckpt files ...")
                # delete all ckpt files
                for file in ckpt_files:
                    os.remove("model/model_ckpts/" + file)

                print("reset log file ...")
                with open("log.txt", "w", encoding="utf-8") as f:
                    pass

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
    tokenizer = get_tokenizer(custom=True if pretrain_config["tokenizer_type"] == "custom" else False)
    fill_mask_pipeline = FillMaskPipeline(model, tokenizer, strategy="greedy")
    nsp_pipeline = IsNextPipeline(model, tokenizer)



if pretrain_config["mlflow_tracking"]:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("bert_implementation_tr")
    # every run should have the same model training whether from scratch or resume! (if mlflow tracking is enabled)
    # if resume, we need to use last ckpt run id to continue tracking consistently (we kept it in ckpt files) 
    if pretrain_config["resume"]:
        mlflow_run_id = mlflow.start_run(run_name=f"bert_pretraining_resume_{mlflow_run_id}").info.run_id
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
        raise NotImplementedError(f"schedular type {schedular_type} not implemented yet, only 'cosine' is supported for now...")
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
print("train batch size (micro batch size): ", train_loader.batch_size)
print("macro batch size: ", train_loader.batch_size * grad_accum_steps)
print("train block size: ", train_loader.block_size)
print("val batch size: ", val_loader.batch_size)
print("val block size: ", val_loader.block_size)




# 1 additional loop for validation
max_steps += 1
train_loss_accum = 0.0
val_loss_accum = 0.0

sys.exit(0)
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
    remaining_s = (max_steps - step) * tokens_processed / tokens_per_second 
    remaining_m = (remaining_s % 3600) // 60.0
    remaining_h = remaining_s // 3600.0
    
    if pretrain_config["mlflow_tracking"]:
        mlflow.log_metric("total_train_loss", train_loss_accum, step)
        mlflow.log_metric("lr", lr, step)
        mlflow.log_metric("grad_norm", norm, step)
        mlflow.log_metric("mlm_loss", model_output.mlm_loss.item(), step)
        mlflow.log_metric("nsp_loss", model_output.nsp_loss.item(), step)
        mlflow.log_metric("tokens_per_second", tokens_per_second, step)

    print_txt = f"step {step:5d} | total loss: {train_loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | mlm loss: {model_output.mlm_loss.item():.6f} | nsp loss: {model_output.nsp_loss.item():.6f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f} tokens/sec | remaining: {remaining_h:.0f}h:{remaining_m:.0f}m"
    
    # log to file whether or not mlflow is enabled
    with open("log.txt", "a", encoding="utf-8") as f:
                f.write(print_txt + "\n")
    #   toplam adım sayısı / birim adımda geçen süre -> kalan süre
    print(print_txt)



    
    
    




