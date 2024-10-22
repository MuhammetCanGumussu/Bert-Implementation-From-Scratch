"""Heavily inspired by HuggingFace BERT model: "https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/bert/modeling_bert.py#L529"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F


def deneme():
    return 0

@dataclass
class BertConfig:
    vocab_size = 32000
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    hidden_act = "gelu"
    intermediate_size = 3072
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    max_position_embeddings = 512
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    type_vocab_size = 2
    classifier_dropout = None
    # bakılacak
    pad_token_id = deneme()



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
        q = self.query(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = self.key(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = self.value(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.attention_head_size).transpose(1, 2)

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

        # decoder.bias is False means it is None
        # parameter sharing (now self.decoder.bias reference(pointer logic) self.bias tensor (same memory/object))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
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
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None


class BertForPreTraining(nn.Module):
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        # bakılacak
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # weight sharing scheme
        self.bert.embeddings.word_embeddings.weight = self.cls.predictions.decoder.weight

        # self._init_weights()

    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            position_ids: torch.Tensor = None,
            labels: Optional[torch.Tensor] = None,
            next_sentence_label: Optional[torch.Tensor] = None 
    ) -> BertForPreTrainingOutput:
        
        last_hidden_state, pooled_output = self.bert(input_ids, attention_mask, token_type_ids, position_ids)
        prediction_logits, seq_relationship_logits = self.cls(last_hidden_state, pooled_output)
            
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            # bakılacak: pad id'lerden loss alınıyor mu alınmıyor mu?
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=self.config.pad_token_id)
            # bakılacak: focal loss (ce'de weight parametresi) kullanılabilir (notNext, isNext'ten daha fazla old için)
            next_sentence_loss = loss_fct(seq_relationship_logits.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss


        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_logits,
            seq_relationship_logits=seq_relationship_logits,
        )
            
    # bakılacak: save % load checkpoint mekanizması yapılacak

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

   



from transformers import PreTrainedTokenizer
class FillMaskPipeline():
    def __init__(self, model: BertForPreTraining, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: Union[str, List[str]], max_length: Optional[int] = 100, sampling_strategy: Optional[str] = "greedy", num_sample: Optional[int] = 5) -> None:
        """
        if greedy is False, then top-k sampling is used (select top-k then apply multinomial)
        
        Args:
            sampling_strategy (str, optional): --- "greedy" selects the top probability (deterministic), "top-k" selects the top k probabilities then multinomial sampling (stochastic)
            num_sample (int, optional): --- print k top probabilities, whether the strategy is greedy or top-k sampling 
        """
        if sampling_strategy != "greedy" or sampling_strategy != "top-k":
            raise ValueError(f"sampling strategy not supported: {sampling_strategy}")

        if isinstance(text, str):
            text = "[CLS] " + text + " [SEP]"
        else:
            for i, t in enumerate(text):
                text[i] = "[CLS] " + t + " [SEP]"
        
        encoding = self.tokenizer(text, padding="longest", return_tensors="pt").to(self.model.bert.embeddings.word_embeddings.weight.device) # :D
        
        if encoding["input_ids"].size(1) > max_length:
            raise ValueError(f"text too long: {encoding['input_ids'].size(1)} > {max_length}")
        

        mask_idx = torch.where(encoding["input_ids"] == self.tokenizer.mask_token_id)[1]

        with torch.no_grad():
            self.model.eval()

            # B, T, V
            model_output = self.model(**encoding).prediction_logits
            mask_logits = model_output[:, mask_idx, :]  # B, V
            mask_probs = F.softmax(mask_logits, dim=-1) # B, V

            if sampling_strategy == "greedy":
                topk_probs, topk_indices = torch.topk(mask_probs, num_sample, dim=-1)
                for b_idx in range(mask_probs.size(0)):
                    tokens_and_scores = dict()
                    for token_prob, token_id in zip(topk_probs[b_idx].tolist(), topk_indices[b_idx].tolist()):
                        tokens_and_scores["token"] = self.tokenizer.convert_ids_to_tokens(token_id)
                        tokens_and_scores["score"] = token_prob 
                    print(f"Text: {text[b_idx]}, Top {num_sample} Predictions: {tokens_and_scores}")

            else: # sampling_strategy == "top-k":  

                topk_probs, topk_indices = torch.topk(mask_probs, 50, dim=-1)   # topk_probs: (B, 50), topk_indices: (B, 50)
                ix = torch.multinomial(topk_probs, num_sample) # (B, num_sample)
                sampled_token_ids = torch.gather(topk_indices, -1, ix) # (B, 50) --> (B, num_sample)
                sampled_token_probs = torch.gather(topk_probs, -1, ix) # (B, 50) --> (B, num_sample)

                for b_idx in range(mask_probs.size(0)):
                    tokens_and_scores = dict()
                    for token_prob, token_id in zip(sampled_token_probs[b_idx].tolist(), sampled_token_ids[b_idx].tolist()):
                        tokens_and_scores["token"] = self.tokenizer.convert_ids_to_tokens(token_id)
                        tokens_and_scores["score"] = token_prob 
                    print(f"Text: {text[b_idx]}, Top {num_sample} Predictions-> {tokens_and_scores}")



class IsNextPipeline():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: List[str, str] | List[List[str, str]], max_length: Optional[int] = 100) -> None:
        # A'nın uzunluğu max uzunluğu aşarsa hata verecek
        if isinstance(text[0], str):
            text = "[CLS] " + text[0] + " [SEP]" + text[1] + " [SEP]"
        else:
            for i, t in enumerate(text):
                text[i] = "[CLS] " + t[0] + " [SEP]" + t[1] + " [SEP]"

        encoding = self.tokenizer(text, padding="longest", return_tensors="pt").to(self.model.bert.embeddings.word_embeddings.weight.device)

        if encoding["input_ids"].size(1) > max_length:
            raise ValueError(f"text too long: {encoding['input_ids'].size(1)} > {max_length}")
        
        # encoder.token_type_ids is all 0 due to lack of "postprocess" component in tokenizer (i will fix it later)
        # because of this reason we need to create token_type_ids by ourselves
        token_type_ids = torch.ones_like(encoding["input_ids"])
        first_sep_indices = torch.argmax((encoding["input_ids"] == self.tokenizer.sep_token_id), dim=1)

        for i in torch.arange(token_type_ids.size(0)):
            token_type_ids[i, :first_sep_indices[i]] = 0

        encoding["token_type_ids"] = token_type_ids

        with torch.no_grad():
            self.model.eval()

            # B, 2
            nsp_logits = self.model(**encoding).seq_relationship_logits
            nsp_probs = F.softmax(nsp_logits, dim=-1) # B, V

            for b_idx in range(nsp_logits.size(0)):
                print(f"Text: {text[b_idx]}, Predictions-> { f'isNext: {nsp_probs[b_idx][1].item()}, notNext: {nsp_probs[b_idx][0].item()}' }")










