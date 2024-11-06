"""Model implementation is heavily inspired by HuggingFace BERT model: "https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/bert/modeling_bert.py#L529"""

import os
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast

from ..data.data_aux import DataLoaderCustom
from ..tokenizer.train_tokenizer import get_tokenizer
from config import BertConfig





class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        # bakılacak
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        # bakılacak
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states



class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        # bakılacak
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states




class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # bakılacak
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        

    
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:        
        # (B, num_heads, T, head_size)
        q = self.query(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = self.key(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = self.value(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        B, T = attention_mask.shape
        # (B, 1, 1, T) : this will be broadcasted to (B, num_heads, T, T) in scaled_dot_product_attention
        attention_mask = attention_mask.view(B, 1, 1, T)
        dropout_p = self.attention_probs_dropout_prob if self.training else 0.0

        y = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=dropout_p, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(hidden_states.size(0), hidden_states.size(1), self.all_head_size) 
        
        return y



class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        # bakılacak
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask) -> torch.Tensor:

        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output



class BertLayer(nn.Module):
    """sa, ln, residual_add, ffn, ln, residual_add"""
    def __init__(self, config: BertConfig):
        super().__init__()

        # bakılacak
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        self_attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(self_attention_output)
        layer_output = self.output(intermediate_output, self_attention_output)

        return layer_output



class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        # bakılacak
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, token_type_ids, position_ids) -> torch.Tensor:
        

        # bakılacak: geçici assertion
        assert input_ids.max().item() < self.word_embeddings.num_embeddings, f"[ERROR]: input_id_max:{input_ids.max().item()} , num_embeddings: {self.word_embeddings.num_embeddings}"

        input_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        # bakılacak
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states



class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        # bakılacak
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output




class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        # bakılacak
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states



class BertLMPredictionHead(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        # bakılacak
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # bakılacak: bunu ekstradan koydum, config'de classfier dropout var idi, ancak hf imp'de bunu görmedim (ya da kaçırdım)
        self.dropout = nn.Dropout(config.classifier_dropout)

        # decoder.bias is False means it is None
        # parameter sharing (now self.decoder.bias reference(pointer logic) self.bias tensor (same memory/object))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states



class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)


    def forward(self, input_ids, attention_mask, token_type_ids, position_ids) -> Tuple[torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        last_hidden_state = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(last_hidden_state)

        return last_hidden_state, pooled_output


class BertPreTrainingHeads(nn.Module):
    """MLM and NSP heads"""
    def __init__(self, config: BertConfig):
        super().__init__()

        # MLM HEAD
        self.predictions = BertLMPredictionHead(config)
        # NSP HEAD
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    
    def forward(self, last_hidden_state, pooled_output) -> Tuple[torch.Tensor]:
        prediction_scores = self.predictions(last_hidden_state)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score



@dataclass
class BertForPreTrainingOutput():
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    nsp_loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None


PAD_TOKEN_ID = get_tokenizer().convert_tokens_to_ids("[PAD]")


class BertForPreTraining(nn.Module):
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        # bakılacak
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # weight sharing scheme
        self.bert.embeddings.word_embeddings.weight = self.cls.predictions.decoder.weight

        # initialize weights
        self.apply(self._init_weights)

    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            position_ids: torch.Tensor = None,
            labels: Optional[torch.Tensor] = None,
            next_sentence_label: Optional[torch.Tensor] = None,
            class_weights: Optional[torch.Tensor] = None
    ) -> BertForPreTrainingOutput:
        
        last_hidden_state, pooled_output = self.bert(input_ids, attention_mask, token_type_ids, position_ids)
        prediction_logits, seq_relationship_logits = self.cls(last_hidden_state, pooled_output)
            
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct_mlm = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID)
            loss_fct_nsp = nn.CrossEntropyLoss(ignore_index = -100, weight=class_weights)

            masked_lm_loss = loss_fct_mlm(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct_nsp(seq_relationship_logits.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss


            return BertForPreTrainingOutput(
            loss=total_loss,
            mlm_loss=masked_lm_loss,
            nsp_loss=next_sentence_loss,
            prediction_logits=prediction_logits,
            seq_relationship_logits=seq_relationship_logits,
            )


        return BertForPreTrainingOutput(
            prediction_logits=prediction_logits,
            seq_relationship_logits=seq_relationship_logits,
        )



        
            
    # bakılacak: save % load checkpoint mekanizması yapılacak

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, weight_decay: float, learning_rate: float, device: str) -> torch.optim.Optimizer:
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=use_fused)
        return optimizer
    

    @classmethod
    def from_pretrained(cls):
       """Loads pretrained BerTurk model weights from huggingface"""

       # create a from-scratch initialized BertForPretraining model
       config = BertConfig()
       model = cls(config)
       sd = model.state_dict()
       sd_keys = sd.keys()

       from transformers import BertForPreTraining
       print("loading weights from pretrained bert: %s" % "dbmdz/bert-base-turkish-cased")

       # init a huggingface/transformers model
       model_hf = BertForPreTraining.from_pretrained("dbmdz/bert-base-turkish-cased")
       sd_hf = model_hf.state_dict()
       sd_keys_hf = sd_hf.keys()
       
       assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

       # port to sd from sd_hf
       for k in sd_keys_hf:
        # vanilla copy over the other parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k])

       return model
       
    
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def save_checkpoint(model: BertForPreTraining,
                    optimizer: torch.optim.Optimizer,
                    dataloader: DataLoaderCustom,
                    step: int, 
                    best_val_loss: float,
                    # train_loss: float,    # bakılacak, mlflow ile track edeceğim için bunlara gerek yok bence
                    # val_loss: float,
                    postfix: int | str,
                    mlflow_run_id: Optional[str] = None
                    ) -> None:
    """
    Save checkpoint dictionary
    """
    temp_dict = {
        'last_step': step,
        'best_val_loss': best_val_loss,
        #'train_loss': train_loss,     # bakılacak, mlflow ile track edeceğim için bunlara gerek yok bence
        #'val_loss': val_loss,         # 
        'model_config': model.config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'last_dataloader_state': dataloader.get_current_state(),
        'mlflow_run_id': mlflow_run_id
    }
    
    torch.save(temp_dict, MODEL_DIR + f"/model_ckpts/BertForPretraining_{postfix}.pt")



def load_checkpoint(postfix: int | str) -> dict:
    """
    Return ckeckpoint dictionary
    """
    return torch.load(MODEL_DIR + f"/model_ckpts/BertForPretraining_{postfix}.pt")

    


class FillMaskPipeline():
    """"default top-50 multinomial sampling if strategy is multionomial, else top-50 greedy sampling"""
    def __init__(self, model: BertForPreTraining, tokenizer: PreTrainedTokenizerFast, strategy: str = "multinomial"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = self.model.bert.config.max_position_embeddings
        self.strategy = strategy

    def __call__(self, text: List[str], do_print: bool = True) -> None:
        
        if self.strategy not in ["multinomial", "greedy"]:
            raise ValueError(f"unknown strategy: {self.strategy}")


        text_with_special_tokens = []
        for t in text:
            if "[MASK]" not in t:
                raise ValueError(f"mask token not found in text: {t}")
            text_with_special_tokens.append("[CLS] " + t + " [SEP]")


        encoding = self.tokenizer(text_with_special_tokens, padding="longest", return_tensors="pt").to(self.model.bert.embeddings.word_embeddings.weight.device)
        encoding["attention_mask"] = encoding["attention_mask"].to(torch.bool)
        if encoding["input_ids"].size(1) > self.max_length:
            raise ValueError(f"text too long: {encoding['input_ids'].size(1)} > {self.max_length}")
        
        mask_row_idxs, mask_col_idxs = torch.where(encoding["input_ids"] == self.tokenizer.mask_token_id)

        with torch.no_grad():
            self.model.eval()
            # B, T, V
            model_prediction_logits = self.model(**encoding).prediction_logits
            # B, V
            model_mask_logits = model_prediction_logits[mask_row_idxs, mask_col_idxs, :]    
            # B, V
            model_mask_probs = F.softmax(model_mask_logits, dim=-1)
            if self.strategy == "multinomial":
                # B, 50
                topk_probs, topk_indices = torch.topk(model_mask_probs, 50, dim=-1)
            else:
                # top 5 probs will automatically be selected in multinomial below (kind of cheaty but it works as greedy)
                # B, 5
                topk_probs, topk_indices = torch.topk(model_mask_probs, 5, dim=-1)
            # B, 5
            sampled_ids = torch.multinomial(topk_probs, 5) 
            # B, 5
            sampled_token_ids = torch.gather(topk_indices, -1, sampled_ids)
            sampled_token_probs = torch.gather(topk_probs, -1, sampled_ids)
            txt = ""
            for b_idx in range(topk_probs.size(0)):
                d = {"token_str":[], "score":[]}
                for i in range(5):
                    d["score"].append(format(sampled_token_probs[b_idx][i].item(), '.4f'))
                    d["token_str"].append(self.tokenizer.convert_ids_to_tokens(sampled_token_ids[b_idx][i].item()))
                txt += f"Text: {text[b_idx]} ----> Top 5 Predictions: {d}\n"
            print(txt) if do_print else None
            return txt




 
class IsNextPipeline():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = self.model.bert.config.max_position_embeddings

    def __call__(self, text: List[List[str]], do_print: bool = True) -> None:
        text_with_special_tokens = []
        for l_str in text:
            textA = l_str[0]
            textB = l_str[1]
            text_with_special_tokens.append("[CLS] " + textA + " [SEP] " + textB + " [SEP]")
        
            
        encoding = self.tokenizer(text_with_special_tokens, padding="longest", return_tensors="pt").to(self.model.bert.embeddings.word_embeddings.weight.device)
        encoding["attention_mask"] = encoding["attention_mask"].to(torch.bool)

        if encoding["input_ids"].size(1) > self.max_length:
            raise ValueError(f"text too long: {encoding['input_ids'].size(1)} > {self.max_length}")
        
        token_type_ids = torch.ones_like(encoding["input_ids"])
        first_sep_indices = torch.argmax(((encoding["input_ids"] == self.tokenizer.sep_token_id)).to(torch.int), dim=1)

        for i in range(encoding["input_ids"].size(0)):
            token_type_ids[i, :first_sep_indices[i]] = 0
        
        encoding["token_type_ids"] = token_type_ids

        with torch.no_grad():
            self.model.eval()

            # B, 2
            model_seq_logits = self.model(**encoding).seq_relationship_logits
            # B, 2
            nsp_probs = F.softmax(model_seq_logits, dim=-1) 
            txt = ""
            for b_idx in range(model_seq_logits.size(0)):
                txt += f"Text: {text[b_idx]} Predictions ----> { f'isNext: {nsp_probs[b_idx][0].item():.3f}, notNext: {nsp_probs[b_idx][1].item():.3f}' }\n" 
            print(txt) if do_print else None
            return txt


