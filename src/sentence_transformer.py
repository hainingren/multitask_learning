import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
class SentenceTransformer:
    def __init__(self, model_name: str='dilbert-base-uncased', pool_mode: str='cls'):
        """param pool_mode can be 'cls' or 'mean' """
        self.model_name = model_name
        self.pool_mode = pool_mode
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
    def forward(self, sentences):
        """
        :param sentences: List of sentences.
        :return: Tensor of shape (batch_size, hidden_size) containing sentence embeddings. 
        """
        encoding = self.tokenizer(
            sentences,
            padding = True,
            truncation = True, 
            return_tensors = "pt" #Use option "pt" for pytorch
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        outputs = self.model(input_ids, attention_mask = attention_mask)
        last_hidden_state = outputs.last_hidden_state ## (batch_size, seq_len, hidden_size)
        
        if self.pool_mode == "cls": 
            ## BERT has a [CLS] token at index 0. Distilbert should have a a simlar CLS token - see f.e.
            #  https://discuss.huggingface.co/t/distilbert-and-cls-token/3700/3
            embeddings = last_hidden_state[:,0,:]
        elif self.pool_name == "mean":
            #mean pooling, only take the average of the unmasked
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            masked_embeddings = last_hidden_state * mask
            embeddings = masked_embeddings.sum(1) / mask.sum(1)
        else:
            raise ValueError(f"Invalid pool_mode: {self.pool_mode}")
       
        return embeddings 