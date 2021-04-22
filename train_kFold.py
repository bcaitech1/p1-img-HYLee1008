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
from sklearn.model_selection import KFold

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

  # set random seed
  seed_everything(args.seed)

  # load model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

  # load dataset
  train_dataset = load_data("/opt/ml/input/data/train/augmented_google.tsv", augmented=True)

  train_label = train_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  kfold = KFold(n_splits=5)

  for fold, (train_index, valid_index) in enumerate(kfold.split(train_dataset), 1):    
    train_sub = torch.utils.data.Subset(RE_train_dataset, train_index)
    valid_sub = torch.utils.data.Subset(RE_train_dataset, valid_index)
    
    train_loader = torch.utils.data.DataLoader(
        train_sub,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_sub,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # load model
    model = XLMRoBERTAClassifier(args.bert_model).to(device)
    model = mbart(args.bert_model).to(device)

    # load optimizer & criterion
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = LabelSmoothingLoss(smoothing=args.smoothing)

    best_acc, last_epoch = 0, 0
    for epoch in range(1, args.epochs + 1):
      model.train()
      loss_value = 0
      start_time = time.time()
      for batch_id, item in enumerate(train_loader):
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
        for batch_id, item in enumerate(valid_loader):
          input_ids = item['input_ids'].to(device)
          # token_type_ids = item['token_type_ids'].to(device)
          attention_mask = item['attention_mask'].to(device)
          labels = item['labels'].to(device)

          output = model(input_ids, attention_mask)
          pred = torch.argmax(output, dim=-1)

          acc_item = (labels == pred).sum().item()
          acc_vals += acc_item

        val_acc = acc_vals / len(valid_sub)

      time_taken = time.time() - start_time

      # metric = {'val_acc': val_acc}
      # wandb.log(metric)
      
      print("fold: {} epoch: {}, loss: {}, val_acc: {}, time taken: {}".format(fold, epoch, train_loss, val_acc, time_taken))
      if best_acc < val_acc:
        print(f'best model! saved at fold {fold} epoch {epoch}')
        if os.path.isfile(f"{save_dir}/{fold}_best_{last_epoch}.pth"):
          os.remove(f"{save_dir}/{fold}_best_{last_epoch}.pth")
        torch.save(model.state_dict(), f"{save_dir}/{fold}_best_{epoch}.pth")
        best_acc = val_acc
        last_epoch = epoch

    # save model
    torch.save(model.state_dict(), f"{save_dir}/{fold}_last_{epoch}.pth")
    

      



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

  parser.add_argument('--model_dir', type=str, default='./results/kfold', help='directory where model would be saved (default: ./results)')

  # xlm-roberta-large
  # joeddav/xlm-roberta-large-xnli
  # monologg/koelectra-base-v3-discriminator
  # facebook/mbart-large-cc25
  parser.add_argument('--bert_model', type=str, default='xlm-roberta-large', help='backbone bert model for training (default: xlm-roberta-large)')
  parser.add_argument('--split_valid', type=int, default=1, help='whether split training set (default: 1)')

  args = parser.parse_args()

  main(args)
