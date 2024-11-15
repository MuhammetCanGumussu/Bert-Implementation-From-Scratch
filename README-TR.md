
🇬🇧 [English](./README.md) &nbsp; | &nbsp; 🇹🇷 [Türkçe](./README-TR.md)


Proje Tanımı
------------
Bu proje sıfırdan ``BERT`` modeli oluşturup pretrain taskler üzerinde eğitim yaparak daha sonraki hedef taskler (downstream tasks) için baseline bir model oluşturma
sistemidir. ``BERT`` modeli Transformers mimarisinin sadece encoder kısmı olarak düşünülebilir. Decoder'ın aksine (GPT gibi modeller) her token context'i sadece solundaki tokenlardan değil ayrıca sağ tarafındaki tokenlardan da beslenerek oluşur. 


``BERT`` modeli self supervised şeklinde eğitilir. Girdi sinyalini belirli şekil veya formda bozarak (veya kapatarak) modelden sinyalin bu kısmı doğru şekilde tahmin etmesi istenir. Bunun için mask language modeling ve next sentence prediction şekline iki farklı task mevcuttur (multi task learning denebilir).


Sample'lar şu formata sahiptir: ``[CLS] Seq_A [SEP] Seq_B [SEP]``. Next sentence prediction (NSP) task'i için veri setinde 0.5 olasılık ile A ve B sequenceler ardışık veya rasgele olacak şekilde çıkartılır. Mask language modeling task'i için ise sample üzerinde (A ve B sequence) belirli olasılıklar ile rasgele kelimeler seçilir, ve her kelimenin tokenları 3 farklı duruma göre değiştirilir(*): Mask, Replace, Identity. Mask durumunda kelime tokenları [MASK] tokenları ile doldurulur, replace durumunda kelime tokenları aynı token sayısına sahip rasgele seçilen bir kelimenin tokenları ile doldurulur, identity durumunda ise kelime tokenları aynı kelimenin tokenları ile doldurulur.


(*) : Orijinal [BERT paper](https://arxiv.org/abs/1810.04805)'inda mask, replace durumları tek bir token üzerinden yapılırken, Bu projede ``Whole Word Masking`` yöntemi uygulanmıştır.




Prepare Dataset
------------
Dokümanların boş satırlar ile ayrıştığı txt dosyaları ile de çalışabilecek bir sistem  (``prepare_data.py``) geliştirildi. Sistem önce raw klasöründeki dosyaları alır doc shards oluşturur. Daha sonra tüm bu doc'lar üzerinde conv operasyonuna benzer bir şekilde bir pencere hareket ettirerek ab sample'ları oluşturur. Ab shard'ları oluşumundan sonra xy shardları ve stat dosyası oluşturarak veriyi model eğitimine hazır hale getirir.


Bu sistem default konfigurasyonlar/parametreler çalışabileceği gibi kullanıcı tarafından da belirtilen parametreler/konfigurasyonlar ile de çalışabilmektedir.


**Dikkat:** Kullanıcı, veri seti oluşturma aşamasında kullanılacak model ile uygun tokenizer tipini ("``custom``" veya "``hf``") doğru belirtmelidir. "``custom``" ile "``hf``" hakkında detaylara tokenizer eğitimi ile alakalı başlıkta erişilebilir.


**Not:** Pencere çoğu durumda dokümanın sonunda dışarıda kalacağından dolayı boş kalan kısımı [PAD] token'i ile doldurmak yerine rasgele B_seq almak tercih edildi (sample notNext oldu yani). Ancak bu durumda sample'lar arasında notNext lehine olacak şekilde en az doküman sayısı kadar fazlalık oluştu. Bu durum overlap ve veya block size arttıkça çok daha dengesiz bir hal aldı öyle ki notNext'li sample sayısı isNext'li sample sayısının 9 katı kadar olduğu ekstrem durumlar da görüldü. Bu duruma class imbalance denir. Sağlıklı, stabil bir eğitim ve model performansı için kötü bir durumdur. Bunu dengeleyebilmek için weighted loss kullanıldı




Model
------------
Model mimarisinin implementasyonu huggingface'in [modeling_bert.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py )'ına bakılarak yapıldı. Kullanıcı tarafından belirtilen konfigurasyonlara göre farklı derinliklerde, şekillerde, özelliklerde vs. model oluşturulabilir. Sıfırdan, parametreleri/ağırlıkları random şekilde initialized edilerek oluşturulan modellere "``custom``" denenebilir. Bu tarz durumlarda kullanıcı istediği konfigürasyonları değiştirebilir.


"``custom``" olmadığı durumlarda huggingface'deki halihazırda eğitilmiş [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased) modelin parametrelerini "``custom``" oluşturulan (randomly initialized, default configs) bir modele yükleme/transfer etme işlemi yapılır, buna da "``hf``" denenebilir. Bu işlemin yapılabilmesi için model konfigurasyonu default olarak kullanılmalıdır (default konfigurasyonlar default ``BERT`` modeline fix'tir) yani kullanıcı farklı bir ayar girmemelidir. Bu "``hf``" modelinin parametre transferinin kolay olabilmesi için model parametrelerinin keyleri aynı isimde bırakıldı.




Custom Tokenizer Eğitimi
------------
HuggingFace [Tokenizers](https://huggingface.co/docs/tokenizers/index) kütüphanesi kullanılarak kullanıcı tarafından belirtilen parametreler/konfigurasyonlar (konfigurasyonlar ile ilgili detaylar altta) ile "raw" klasöründeki veriler ile tokenizer eğitilir (``train_tokenizer.py``). Tokenizer pipeline'ı klasik Bert tokenizer'ına benzer olup (örn aynı model'e sahip: WordPiece), bazı komponentler ile tokenizer'ın ve modelin daha başarılı olabileceği varsayımı ile değiştirilmiştir. Örneğin normalizer kısmında strip accent komponenti konulmamıştır. Pretokenizer'da sadece WhitespaceSplit değil digits ve punctuation komponentleri de eklenmiştir. Ancak bu değişikliklerin gerçektende model için daha iyi olup olmadığı test edilmemiştir. Eğer tokenizer pipeline'ında farklı komponentler (normalizer, pretokenizer, model vs.) kullanmak veya değiştirmek istenirse direkt [train_tokenizer.py](./tokenizer/train_tokenizer.py) üzerinden değişiklik yapılabilir. Kullanıcı isteğine göre cased veya uncased şekilde tokenizer eğitilebilir (default cased). Son olarak tokenizer eğitimi sonucu çıkan tokenizer kaydedilir (örn: ``tr_wordpiece_tokenizer_cased.json``). 


Hangi model kullanılacak ise o model için uygun olan tokenizer klasörde bulundurulmalıdır. Model olarak "``custom``" kullanılacak ise bu klasördeki tokenizer kullanılacak. Eğer model "``custom``" değil de "``hf``" ise o zaman "HuggingFace BERTurk ağırlıkları" kullanılacağı anlamına gelir, bu durumda sistem otomatik olarak BERTurk tokenizer'ını çekecektir (``AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased") `` ile) 




Random Word Set Oluşturma
------------
Veri hazırlanırken yukarıda "``replace``" durumlarından bahsedildi. Bir kelime yerine rasgele kelime koymak için önce kelimenin kaç tokendan oluştuğu bilinmelidir. Daha sonra bu token sayısına eşit olan rasgele bir kelime seçilip o kelimenin tokenları döndürülebilmelidir. Bunun düzgün ve kolay şekilde sağlanabilmesi için kullanıcı tarafından belirtilen parametreler/konfigurasyonlar (konfigurasyonlar ile ilgili detaylar altta) ile sistem tarafından (``prepare_random_word_set.py``), ``random_word_set.json`` dosyası oluşturulmalıdır.


Dikkat kullanıcı, kullanılacak model ile uygun tokenizer tipini ("``custom``" veya "``hf``") doğru belirtmelidir. Ayrıca *use_number_of_line* parametresini büyük bir sayı ile kullanımda operasyon baya uzun sürebilir. *use_number_of_line* parametresi random word set oluştururken veri setimizde kaç line kullanılacağını belirtir.




Pretrain (MLM, NSP)
------------
Bu sistem (``pretrain_bert.py``) kullanıcı tarafından belirtilen konfigurasyon/parametreler ile pretrain taskler üzerinde sıfırdan custom bir model veya hf weight'leri kullanılarak (önceden eğitilmiş, random değil) eğitim yapılır. Ckpt sistemi sayesinde son kalınılan noktadan itibaren eğitim devam edebilir. Eğitim boyunca önceden tanımlanmış bazı text'ler üzerinde model çalıştırılır ve çıktıları "generated_samples.txt" üzerinde kaydedilir. Ayrıca *train_loss*, *val_loss*, *grad_norm* vs gibi metrik veya değişkenler üzerinde tracking/loglama yapılır. Kullanıcı ayrıca konfigürasyonlarda ayarlayarak mlflow ile tracking yapabilir. Eğtimin/optimizasyonun daha verimli stabil vs olması için bazı sistemler algoritmalar kullanıldı: *lr_schedular* ve *lr_warmup* (get_lr fonksiyonu içinde tanımlandılar.), *amp* (automatic mixed precision), *grad_accum*, *clip_norm*, *adamW fused* (weight decay sadece 2d parametre tensorleri üzerinde yapılacak şekilde ayarlandı.). 


Pretrain taskler üzerinde model eğitimi yapmadan hali hazırda eğitilmiş bir model üzerinde sadece "evaluation" da yapılabilir ("*do_eval_from_best_ckpt*" veya "*samples_for_nsp_generation*" ayarları ile). Modeli farklı text'ler üzerinde denemek istenirse (eğitim esnasında veya sadece evaluation'da) *pretrain_bert.py* içerisindeki "*samples_for_mlm_generation*" ve "*samples_for_nsp_generation*" list objeleri üzerinde değişiklikler yapılabilir (formatları bozmamak kaydıyla).


Bu sistemde genel hatlarıyla Andrej Karpathy'nin [build-nanogpt](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py) projesinden ilham alınmıştır.


**Dikkat**: "resume" en son kalınan ckpt üzerinden devam eder, sıfırdan custom veya hf ile model oluşturmada ise önceki eğitimden kalan tüm ckpt'ler ve log dosyaları (*log.txt* ve *generated_samples.txt*) resetlenir.




Kurulum
----------
Projenin düzgün çalışabilmesi için gerekli paketlerin yüklenmesi (önce sanal env oluşturulması önerilir):


```sh
pip install -r requirements.txt
```




Konfigürasyonlar
----------
``train_tokenizer.py``'da kullanılabilecek parametreler/konfigürasyonlar:
```python
@dataclass
class TrainTokenizerConfig:
    vocab_size: int = field(default=32_000, metadata={"description": "Vocab size of token embeddings"})
    limit_alphabet: int = field(default=200, metadata={"description": "Limit for alphabet"})
    min_frequency: int = field(default=2, metadata={"description": "Min frequency"})
    cased: bool = field(default=True, metadata={"description": "Cased or uncased"})
```
```sh
python train_tokenizer.py --vocab_size=32000 --cased=True ...
```




``prepare_data.py``'da kullanılabilecek parametreler/konfigürasyonlar:
```python
@dataclass
class DataConfig:
    block_size: int = field(default=128, metadata={"description": "Block size"})
    num_of_docs_per_shard: int = field(default=8_000, metadata={"description": "Number of docs per shard (for doc_shards and ab_shards creation)"})
    num_tokens_per_shard: int = field(default=10_000_000, metadata={"description": "Number of tokens per shard (for xy_shards creation)"})
    overlap: int = field(default=64, metadata={"description": "Overlap, how much overlap between windows when ab samples are generated (suggestion: make half of the block size)"})
    edge_buffer: int = field(default=10, metadata={"description": "prevents the use of a and b seperators near window ends by this many tokens (makes a and b more similar length-wise) (for ab_shards creation)"})
    seed: int = field(default=13013, metadata={"description": "Seed, for reproducibility"})
    rate_of_untouched_words: float = field(default=0.85, metadata={"description": "Rate of untouched words, these are not gonna replaced by [mask, identity, replaced] tokens"})
    mask_ratio: float = field(default=0.80, metadata={"description": "Mask ratio"})
    replace_ratio: float = field(default=0.10, metadata={"description": "Replace ratio"})
    identity_ratio: float = field(default=0.10, metadata={"description": "Identity ratio"})
    tokenizer_type: str = field(default="custom", metadata={"description": "Tokenizer type, Data has been prepared/tokenized with. choices: [custom, hf]"})
```
```sh
python prepare_data.py --block_size=512 --seed=1881 --tokenizer_type=hf ...
```




``prepare_random_word_set.py``'da kullanılabilecek parametreler/konfigürasyonlar:
```python
@dataclass
class RandomWordSetConfig:
    limit_for_token_group: int = field(default=5, metadata={"description": "Limit for token group"})
    max_word_limit_for_token_group: int = field(default=5_000, metadata={"description": "Max word limit for token group"})
    min_freq_for_words: int = field(default=150, metadata={"description": "Min freq for words"})
    random_sample: bool = field(default=False, metadata={"description": "Randomly sample words (not directly most frequent ones)"})
    use_number_of_line: int = field(default=None, metadata={"description": "Use specified number of lines for training (if None then use all lines) use small number, otherwise it will take a long time!"})
    tokenizer_type: str = field(default="custom", metadata={"description": "Tokenizer type, random_word_set.json prepared/tokenized with. choices: [custom, hf]"})
```
```sh
python prepare_random_word_set.py --block_size=512 --seed=1881 --tokenizer_type=hf --use_number_of_line=100000 ...
```




``pretrain_bert.py``'da kullanılabilecek parametreler/konfigürasyonlar (BertConfig ile PreTrainBertConfig):
```python
@dataclass
class BertConfig:
    vocab_size: int = field(default=32000, metadata={"description": "Vocabulary size of token embeddings"})
    hidden_size: int = field(default=768, metadata={"description": "Hidden size of feed-forward layers"})
    num_hidden_layers: int = field(default=12, metadata={"description": "Number of layers in encoder"})
    num_attention_heads : int = field(default=12, metadata={"description": "Number of attention heads"})
    hidden_act: str = field(default="gelu", metadata={"description": "Activation function type choices: [gelu, relu]"})
    intermediate_size: int = field(default=3072, metadata={"description": "Size of the intermediate layer in feed-forward"}) 
    hidden_dropout_prob: float = field(default=0.1, metadata={"description": "Dropout probability for hidden layers"})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"description": "Dropout probability for attention"})
    max_position_embeddings: int = field(default=512, metadata={"description": "Maximum number of position embeddings (max block size of the model)"})
    initializer_range: float = field(default=0.02, metadata={"description": "Standard deviation for linear and embedding initialization"})
    layer_norm_eps: float = field(default=1e-12, metadata={"description": "Layer norm epsilon"})
    type_vocab_size: int = field(default=2, metadata={"description": "Type vocabulary size (for token type embeddings)"})
    classifier_dropout: float = field(default=0.1, metadata={"description": "Dropout probability for classifier"})

@dataclass
class PreTrainBertConfig:
    """This is from scratch scenario default config!"""
    do_train_custom: bool = field(default=True, metadata={"description": "Do train with custom model or with BERTurk hf model weights."})
    do_eval_from_best_ckpt: bool = field(default=False, metadata={"description": "Just evaluate the model from_best_ckpt but not training."})
    do_eval_from_huggingface: bool = field(default=False, metadata={"description": "Just evaluate the model from_huggingface but not training."})
    resume: bool = field(default=False, metadata={"description": "Resume training from the last step"})
    stage1_ratio: float = field(default=0.9, metadata={"description": "Ratio of stage1 (e.g. 0.9 means use block_size_s1 and train_batch_size_s1 until the end of %90 training then switch to stage2)"})
    block_size_s1: int = field(default=128, metadata={"description": "Block size for stage1"})
    block_size_s2: int = field(default=512, metadata={"description": "Block size for stage2"})
    train_batch_size_s1: int = field(default=64, metadata={"description": "Training batch size for stage1"})
    train_batch_size_s2: int = field(default=16, metadata={"description": "Training batch size for stage2"})
    val_block_size: int = field(default=512, metadata={"description": "Validation block size"})
    val_batch_size: int = field(default=8, metadata={"description": "Validation batch size"})
    grad_accum_steps: int = field(default=1, metadata={"description": "Gradient accumulation steps (micro steps)"})
    max_learning_rate: float = field(default=1e-4, metadata={"description": "Maximum learning rate"})
    min_learning_rate: float = field(default=1e-4 * 0.001, metadata={"description": "Minimum learning rate"})
    lr_scheduler: str = field(default="cosine", metadata={"description": "Learning rate scheduler choices: [linear, cosine]"})
    num_train_steps: int = field(default=100_000, metadata={"description": "Number of training steps"}) 
    num_warmup_steps: int = field(default=5_000, metadata={"description": "Number of warmup steps"})
    save_checkpoints_steps: int = field(default=50, metadata={"description": "Save checkpoints steps"})
    val_check_steps: int = field(default=50, metadata={"description": "Validation check steps"})
    device: str = field(default="cuda", metadata={"description": "Device choices: [cpu, cuda, mps]"})
    max_eval_steps: int = field(default=50, metadata={"description": "Maximum evaluation steps (if validation set is small then max_eval_steps is gonna be treated as validation set size)"})
    weight_decay: float = field(default=0.01, metadata={"description": "Weight decay (L2 regularization)"})
    max_ckpt: int = field(default=5, metadata={"description": "Maximum number of last checkpoints to keep"})
    seed: int = field(default=13013, metadata={"description": "Random seed"})
    generate_samples: bool = field(default=True, metadata={"description": "Try model on predefined samples"})
    mlflow_tracking: bool = field(default=True, metadata={"description": "MLflow tracking"})      
    tokenizer_type: str = field(default="custom", metadata={"description": "Tokenizer type, Data has been prepared/tokenized with. choices: [custom, hf]"})
```



**Dikkat**: belirtilen "*block_size*" veya "*tokenizer_type*"'lar ile veri setinin hali hazırda **prepere_data.py** ile önceden hazırlanmış olması gerekmektedir



**Sıfırdan custom model oluşturup eğitme senaryosunda** kullanıcı bu parametreler dışında herhangi bir parametreyi verebilir ``["do_eval_from_best_ckpt", "do_eval_from_huggingface", "tokenizer_type", "resume"]``:
```sh
python pretrain_bert.py --block_size_s1=128 --block_size_s2=512 \
                        --grad_accum_steps=5 --tokenizer_type=custom \
                        --train_batch_size_s1=64 --train_batch_size_s2=16 \
                        --vocab_size=15000 --num_hidden_layers=8 --hidden_size=512 ...
```



**En son kalınılan model üzerinde eğitime devam etme senaryosunda** kullanıcı bu parametreler dışında herhangi bir ``PreTrainBertConfig`` parametresi verebilir ``["do_train_custom", "do_eval_from_best_ckpt", "do_eval_from_huggingface", "max_ckpt", "tokenizer_type"]``, verdiği parametreler en sondaki ckpt'nin pretrain_cfg üzerinde override edilir. Ancak hiç bir ``BertConfig`` parametresi verilemez, model cfg direkt olarak en sondaki ckpt'den alınıp kullanılır:
```sh
python pretrain_bert.py --resume=True ...
```



**do_eval_from_best_ckpt senaryosu**, kullanıcı sadece bu parametreleri girebilir ``["do_eval_from_best_ckpt", "val_block_size", "val_batch_size", "device", "max_eval_steps", "seed", "generate_samples", "tokenizer_type"]`` (**Dikkat** bazı parametreler fix'tir yani farklı değer girilemez, örn ``--tokenizer_type=custom`` olmak zorunda):
```sh
python pretrain_bert.py --do_eval_from_best_ckpt=True --val_block_size=512 ...
```



**do_eval_from_huggingface senaryosu**, kullanıcı sadece bu parametreleri girebilir `["do_eval_from_huggingface", "val_block_size", "val_batch_size", "device", "max_eval_steps", "seed"]` (**Dikkat** bazı parametreler fix'tir yani farklı değer girilemez, örn ``--tokenizer_type=hf`` olmak zorunda):
```sh
python pretrain_bert.py --do_eval_from_huggingface=True --val_batch_size=32 ...
```







TODO
----------

* Amp için bfloat16 precision belirtildi ancak ampere mimarisine sahip olmayan ekran kartlarında otomatik olarak float32 kullanılacaktır. Bu tarz donanımlar için float16 default olarak seçilebilmeli ve tabiki bu durum için gradscaler kullanılmalı.

* Modeli direkt kullanabilmek için ayrı ve daha temiz bir tür arayüz tasarlanabilir (şu an pretrain_bert.py'da örnek sample'ları elle değiştirerek bu mümkün)

* Train-val split orantısını parametrik/configurable hale getirme. (Sistem şu an sadece son shard'ı val için ayırıyor.) 

* Custom model oluşturma dışında yalnız BERTurk hf weightleri kullanılabiliyor. Farklı hf weightleri de kullanılabilmeli (örneğin klasik google BERT weightleri de kullanılabilmeli, düz "hf" yerine "hf_google_bert", "hf_berturk" vs gibi)

* Class imbalance'ı kaldırmak için, sistemin veri setindeki doküman sayısı ile çıkartılacak sample sayısını önceden hesaplayıp ona göre randomB oranını (default 0.5 idi) ayarlayabilir hale getirmek. Bu sayede isNext ile notNext sample sayısı 0.5 oranında sabit tutulabilir.

* Ddp yapılabilir (data distributed parallel)

* Eğitilen modeli deploy edebilme (huggingface'e)

* Pretrain edilen modeli farklı taskler üzerinde fine-tune edebilme (sentiment analysis, ner, qa, pos vs.) ve buna göre model pretraini daha iyi değerlendirebilme (benchmark 
taskleri, veri setleri vs kullanılabilir)

* Compile (modeli windows üzerinde compile etmede bazı sıkıntılar yaşandı (ilişkili güncel kütüphanenin windows için direkt bir dağıtımı yok) problemler halledilebilirse (wsl kullanma, linux'ta çalıştırma ya da ilgili kütüphanenin son sürümünün windows'da düzgün şekilde çalışabilmesini sağlayacak wheel'in çıkması durumunda) compile etme denenebilir ki eğitimi baya hızlandıracaktır)




----------

**Author:** *Muhammet Can Gümüşsu*

🔗 [LinkedIn Profilim](https://www.linkedin.com/in/muhammet-can￾g%C3%BCm%C3%BC%C5%9Fsu-876041174/)







