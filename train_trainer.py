import pickle as pickle
import os
import pandas as pd
import numpy as np
import torch
import random
import transformers
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizerFast, BertModel, XLMRobertaConfig, XLMRobertaModel
from transformers.utils import logging
from load_data import *

from torch.utils.tensorboard import SummaryWriter

# seed 고정 
def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train():
  # writer = SummaryWriter()

  # set random seed
  seed_everything(1024)

  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-large"
  # MODEL_NAME = "bert-base-multilingual-cased"
  # MODEL_NAME = "kykim/bert-kor-base"
  # MODEL_NAME = "monologg/kobert"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

  # load dataset
  if args.split_valid:
    train_dataset = load_data("/opt/ml/input/data/train/train.tsv")[:-200]
    print(len(train_dataset))

  else:
    train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    print(len(train_dataset))

  dev_dataset = load_data("/opt/ml/input/data/train/train.tsv")[-200:]
  print(len(dev_dataset))

  train_label = train_dataset['label'].values
  dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  # bert_config = BertConfig.from_pretrained(MODEL_NAME)
  xlm_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 42
  # model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config) 
  model = XLMRobertaModel.from_pretrained(MODEL_NAME, config=xlm_config)
  # model_bert = BertModel.from_pretrained("kykim/bert-kor-base")
  model.parameters
  model.to(device)
  # print(model_bert)

  # output directory
  output_dir = os.path.join('./results', MODEL_NAME)
  os.makedirs(output_dir, exist_ok=True)

  log_dir = os.path.join('./logs', MODEL_NAME)
  os.makedirs(output_dir, exist_ok=True)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    save_total_limit=5,              # number of total save model.
    save_strategy="epoch",
    # save_steps=500,                 # model saving step.
    num_train_epochs=10,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_strategy='epoch',
    logging_dir=log_dir,            # directory for storing logs
    # logging_steps=100,              # log saving step.
    evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    #eval_steps = 500,            # evaluation step.
    label_smoothing_factor=0.2
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()

def main():
  train()

if __name__ == '__main__':
  main()
