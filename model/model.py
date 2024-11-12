"""Model implementation is heavily inspired by HuggingFace BERT model: "https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/bert/modeling_bert.py#L529"""


import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer.tokenizer_aux import get_tokenizer
from config import BertConfig





class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

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

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states



class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

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
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, token_type_ids, position_ids) -> torch.Tensor:
        
        assert position_ids[-1] <= self.config.max_position_embeddings - 1, f"max block size exceeded, {position_ids[-1]} > {self.config.max_position_embeddings - 1}"

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

        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
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
            loss_fct_nsp = nn.CrossEntropyLoss(ignore_index = -100, weight=class_weights)   # -100 is default ignore_index

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
       
    

