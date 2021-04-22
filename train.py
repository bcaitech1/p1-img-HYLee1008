import argparse
import pickle as pickle
import os
import pandas as pd
import numpy as np
import torch
import random
import transformers
import glob
import time
import json
import wandb
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizerFast, BertModel, XLMRobertaTokenizer
from pathlib import Path

from load_data import *
from model import BERTClassifier, XLMRoBERTAClassifier, BERTLarge, KoElectraClassifier, mbart
from loss import LabelSmoothingLoss

from torch.utils.tensorboard import SummaryWriter

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

# seed 고정 
def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def train(args):
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  
  # model save path
  save_dir = increment_path(os.path.join(args.model_dir, args.bert_model))
  os.makedirs(save_dir, exist_ok=True)

  # save args on .json file
  with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=4)
  # writer = SummaryWriter()

  # Auto ML using  wandb
  # hyperparameter_defaults = dict(
  #   batch_size = args.batch_size,
  #   learning_rate = args.learning_rate,
  #   epochs = args.epochs,
  #   bert_model = args.bert_model,
  #   smoothing = args.smoothing
  #   )
  # wandb.init(config=hyperparameter_defaults, project="sweep-test", dir=save_dir, name=save_dir)
  # config = wandb.config

  # set random seed
  seed_everything(args.seed)

  # load model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
  # tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_model)
  # tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

  # add special tokens
  # special_tokens = ['<e1>', '</e1>', '<e2>', '</e2>']
  special_tokens = '[CLS]'

  special_tokens_dct = {'cls_token': special_tokens}
  tokenizer.add_special_tokens(special_tokens_dct)

  # load dataset
  if args.augmentation:
    if args.split_valid:
      train_dataset = load_data("/opt/ml/input/data/train/augmented_google.tsv", augmented=True)
      augmented_dataset = train_dataset[9000:]
      augmented_dataset = augmented_dataset[augmented_dataset['label'] != 0]
      train_dataset = pd.concat([train_dataset[:8800], augmented_dataset])

    else:
      train_dataset = load_data("/opt/ml/input/data/train/augmented_google.tsv", augmented=True)
    dev_dataset = load_data("/opt/ml/input/data/train/augmented_google.tsv", augmented=True)[8800:9000]
        
  else:
    if args.split_valid:
      train_dataset = load_data("/opt/ml/input/data/train/train.tsv")[:-200]

    else:
      train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    dev_dataset = load_data("/opt/ml/input/data/train/train.tsv")[-200:]
  
  print(len(train_dataset))
  print(len(dev_dataset))

  train_label = train_dataset['label'].values
  dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  # make dataloader
  train_dataloader = torch.utils.data.DataLoader(RE_train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
  dev_dataloader = torch.utils.data.DataLoader(RE_dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load model
  # model = BERTClassifier(args.bert_model).to(device)
  # model = XLMRoBERTAClassifier(config.bert_model).to(device)
  # model = BERTLarge(config.bert_model).to(device)
  # model = KoElectraClassifier(args.bert_model).to(device)
  # model = MT5Classifier(args.bert_model).to(device)
  model = mbart(args.bert_model).to(device)
  model.resize_token_embeddings(len(tokenizer))
  # print(model_bert)

  # load optimizer & criterion
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
  # criterion = torch.nn.CrossEntropyLoss()
  criterion = LabelSmoothingLoss(smoothing=args.smoothing)
  # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100)
  # train model
  best_acc, last_epoch = 0, 0
  for epoch in range(1, args.epochs + 1):
    model.train()
    loss_value = 0
    start_time = time.time()
    for batch_id, item in enumerate(train_dataloader):
      input_ids = item['input_ids'].to(device)
      # token_type_ids = item['token_type_ids'].to(device)
      attention_mask = item['attention_mask'].to(device)
      labels = item['labels'].to(device)

      optimizer.zero_grad()
      output = model(input_ids, attention_mask)
      loss = criterion(output, labels)

      loss_value += loss.item()

      loss.backward()
      optimizer.step()
      # scheduler.step()
    
    train_loss = loss_value / (batch_id + 1)

    # evaluate model on dev set
    with torch.no_grad():
      model.eval()
      acc_vals = 0
      for batch_id, item in enumerate(dev_dataloader):
        input_ids = item['input_ids'].to(device)
        # token_type_ids = item['token_type_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)
        labels = item['labels'].to(device)

        output = model(input_ids, attention_mask)
        pred = torch.argmax(output, dim=-1)

        acc_item = (labels == pred).sum().item()
        acc_vals += acc_item

      val_acc = acc_vals / len(RE_dev_dataset)

    time_taken = time.time() - start_time

    # metric = {'val_acc': val_acc}
    # wandb.log(metric)
    
    print("epoch: {}, loss: {}, val_acc: {}, time taken: {}".format(epoch, train_loss, val_acc, time_taken))
    if best_acc < val_acc:
      print(f'best model! saved at epoch {epoch}')
      if os.path.isfile(f"{save_dir}/best_{last_epoch}.pth"):
        os.remove(f"{save_dir}/best_{last_epoch}.pth")
      torch.save(model.state_dict(), f"{save_dir}/best_{epoch}.pth")
      best_acc = val_acc
      last_epoch = epoch

  # save model
  torch.save(model.state_dict(), f"{save_dir}/last_{epoch}.pth")
    

      



def main(args):
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Data and model checkpoints directories
  parser.add_argument('--seed', type=int, default=1024, help='random seed (default: 1024)')
  parser.add_argument('--epochs', type=int, default=10, help='number of epochs for train (deafult: 10)')
  parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (deafult: 16)')
  parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader (default: 4)')
  parser.add_argument('--smoothing', type=float, default=0.2, help='label smoothing facotr for label smoothing loss (default: 0.2)')
  parser.add_argument('--augmentation', type=int, default=0, help='apply google translation augmentation (default: 0)')

  parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate for training (default: 1e-5)')
  parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (default: 0.01)')

  parser.add_argument('--model_dir', type=str, default='./results', help='directory where model would be saved (default: ./results)')

  # xlm-roberta-large
  # joeddav/xlm-roberta-large-xnli
  # monologg/koelectra-base-v3-discriminator
  # facebook/mbart-large-cc25
  parser.add_argument('--bert_model', type=str, default='facebook/mbart-large-cc25', help='backbone bert model for training (default: xlm-roberta-large)')
  parser.add_argument('--split_valid', type=int, default=1, help='whether split training set (default: 1)')

  args = parser.parse_args()

  main(args)
