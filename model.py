import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertModel, XLMRobertaConfig, XLMRobertaModel, AutoModel, AutoConfig


class BERTClassifier(nn.Module):
    def __init__(self,
                 model_name,
                 hidden_size = 768,
                 num_classes = 42,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()

        # load bert model from transformers
        bert_config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=bert_config)
        self.dr_rate = dr_rate 
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size , 500),
        #     nn.ReLU(),
        #     nn.Linear(500, num_classes)
        # )
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert( input_ids=input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids)['pooler_output']
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class XLMRoBERTAClassifier(nn.Module):
    def __init__(self,
                 model_name,
                 hidden_size = 1024,
                 num_classes = 42,
                 dr_rate=None,
                 params=None):
        super(XLMRoBERTAClassifier, self).__init__()

        # load bert model from transformers
        xlm_config = XLMRobertaConfig.from_pretrained(model_name)
        self.bert = XLMRobertaModel.from_pretrained(model_name, config=xlm_config)
        self.dr_rate = dr_rate 
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size , 1000),
        #     nn.ReLU(),
        #     nn.Linear(1000, num_classes)
        # )
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, input_ids, attention_mask):
        out = self.bert( input_ids=input_ids,
          attention_mask=attention_mask)['pooler_output']
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    def resize_token_embeddings(self, num):
        self.bert.resize_token_embeddings(num)


class BERTLarge(nn.Module):
    def __init__(self,
                 model_name,
                 hidden_size = 1024,
                 num_classes = 42,
                 dr_rate=None,
                 params=None):
        super(BERTLarge, self).__init__()

        # load bert model from transformers
        bert_config = BertConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, bert_config)
        self.dr_rate = dr_rate 
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size , 500),
        #     nn.ReLU(),
        #     nn.Linear(500, num_classes)
        # )
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, input_ids, attention_mask):
        out = self.bert( input_ids=input_ids,
          attention_mask=attention_mask)['pooler_output']
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class KoElectraClassifier(nn.Module):
    def __init__(self,
                 model_name,
                 hidden_size = 768,
                 num_classes = 42,
                 dr_rate=None,
                 params=None):
        super(KoElectraClassifier, self).__init__()

        # load bert model from transformers
        bert_config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=bert_config)
        self.dr_rate = dr_rate 
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size , 500),
        #     nn.ReLU(),
        #     nn.Linear(500, num_classes)
        # )
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert( input_ids=input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids)['last_hidden_state'][:, 0, :]

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class mbart(nn.Module):
    def __init__(self,
                 model_name,
                 hidden_size = 1024,
                 num_classes = 42,
                 dr_rate=None,
                 params=None):
        super(mbart, self).__init__()

        # load bert model from transformers
        bert_config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=bert_config)
        self.dr_rate = dr_rate 
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size , 500),
        #     nn.ReLU(),
        #     nn.Linear(500, num_classes)
        # )
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, input_ids, attention_mask):
        out = self.bert( input_ids=input_ids,
          attention_mask=attention_mask)['last_hidden_state'][:, 0, :]

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    def resize_token_embeddings(self, num):
        self.bert.resize_token_embeddings(num)