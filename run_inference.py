import torch
import os
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def to_dict(text,labels):
  dictionary = []
  for txt, y in zip(text, labels):
    dictionary.append({
        "text": txt,
        "label": y
    })
  return dictionary


def compute_metrics(p):    
  pred, labels = p
  pred = np.argmax(pred, axis=1)
  accuracy = accuracy_score(y_true=labels, y_pred=pred)
  recall = recall_score(y_true=labels, y_pred=pred)
  precision = precision_score(y_true=labels, y_pred=pred)
  f1 = f1_score(y_true=labels, y_pred=pred)    
  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


true_csv = pd.read_csv("News_dataset/True.csv")
fake_csv = pd.read_csv("News_dataset/Fake.csv")
fake_csv["label"] = 1
true_csv["label"] = 0
news_df = pd.concat([true_csv, fake_csv], axis=0)
news_df["text"] = news_df["title"] + news_df["text"]

train_X, test_X, train_y, test_y = train_test_split(news_df["text"], news_df["label"], stratify=news_df["label"], test_size = 0.2, random_state = 10)
train_X, dev_X, train_y, dev_y = train_test_split(train_X, train_y, stratify=train_y, test_size = 0.25, random_state = 10) # 0.25 x 0.8 = 0.2

test_spans = to_dict(test_X, test_y)
test_spans_txt = [s['text'] for s in test_spans]
test_y = [s['label'] for s in test_spans]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# max sequence length for each document/sentence sample
max_length = 512
test_encodings = tokenizer(test_spans_txt, truncation=True, padding=True, max_length=max_length)

test_dataset = NewsDataset(test_encodings, test_y)

model = AutoModelForSequenceClassification.from_pretrained("model")
model.eval()

trainer = Trainer(model=model, compute_metrics=compute_metrics)
trainer.model = model.cuda()
trainer.evaluate(test_dataset)
