from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer, BertTokenizerFast
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from model import BERTClassifier, XLMRoBERTAClassifier, BERTLarge, KoElectraClassifier

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  logits_list = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          data['input_ids'].to(device),
          data['attention_mask'].to(device),
          # data['token_type_ids'].to(device)
          )
    logits = outputs
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    logits_list.append(logits)
  
  return np.array(output_pred).flatten(), np.array(logits_list).reshape(1000, -1)

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  # TOK_NAME = "bert-base-multilingual-cased"  
  TOK_NAME = 'xlm-roberta-large'
  # TOK_NAME = 'joeddav/xlm-roberta-large-xnli'
  # TOK_NAME = "kykim/bert-kor-base"
  # TOK_NAME = 'monologg/koelectra-base-v3-discriminator'
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
  # tokenizer = BertTokenizerFast.from_pretrained(TOK_NAME)

  # load my model
  # model = BERTClassifier(TOK_NAME)
  model = XLMRoBERTAClassifier(TOK_NAME)
  # model = BERTLarge(TOK_NAME)
  # model = KoElectraClassifier(TOK_NAME)
  # model_path = "./results/monologg/koelectra-base-v3-discriminator2/best_4.pth"
  # model_path = './results/joeddav/xlm-roberta-large-xnli2/best_4.pth'
  model_path = './results/kfold/xlm-roberta-large2/5_best_5.pth'
  model.load_state_dict(torch.load(model_path))
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  pred_answer, logits = inference(model, test_dataset, device)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv('./prediction/roberta_ensamble4.csv', index=False)

  np.save('./npy/roberta_ensamble4.npy', logits)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/bert-base-multilingual-cased/checkpoint-1692")
  args = parser.parse_args()
  print(args)
  main(args)
  
