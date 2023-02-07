import torch
import os
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def to_dict(text,labels):
  dictionary = []
  for txt, y in zip(text.iterrows(), labels.iterrows()):
    dictionary.append({
        "text": txt[1]["text"], # retrieve values from tuples
        "label": y[1]["label"], # retrieve values from tuples
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


test_X = pd.read_csv("News_dataset/testX.csv")
test_y = pd.read_csv("News_dataset/testY.csv")

test_spans = to_dict(test_X, test_y)
test_spans_txt = [s['text'] for s in test_spans]
test_y = [s['label'] for s in test_spans]

tokenizer = AutoTokenizer.from_pretrained("tokenizer")
# max sequence length for each document/sentence sample
max_length = 512
test_encodings = tokenizer(test_spans_txt, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained("model")
model.eval()

outputs = model(**test_encodings)
predictions = outputs.logits.argmax(-1)

print("Performances on test set:")
print(compute_metrics(predictions, test_y))