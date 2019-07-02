# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:18:00 2019

@author: i7 Laptop
"""

import os
import wget
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np
import json
import ijson
annotated = open( r'C:\Users\i7 Laptop\Documents\Masters articles\data\data\train1.jsonl')
#print(annotated)

data= []
for i,line in enumerate(annotated):
    #json_data = json.reads(line)
    #data = json.loads(line)
    #data = 
    #print(data)

    json_data = json.loads(line.replace("`","\'"))
    data.append(json_data)
print(data)
    #for i in json_data:
     #   datatemp = json_data[i] 
      #  print(datatemp)
#annotated.close()   
data_pd = pd.DataFrame()
from pandas.io.json import json_normalize
df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
df = df.apply(lambda x: x.astype(str).str.lower())
df = df.drop(['sql.conds','sql.sel','table_id','phase'],axis = 1)
print(df)

def clean_text(text, remove_stopwords = True):
    text = text.lower()
    text = re.sub(r'[_"\-;%()|+&*%.,!?:#$@\[\]/]', ' ', text)
  
    return text


aggs = df['sql.agg']
question = df['question']
print(question,aggs)

from nltk.corpus import stopwords
words = list()
for ques in question:
    words.append(clean_text(ques, remove_stopwords=False))
    #print(words)



def build_word_dict():
    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<eos>"] = 2
    for word, _ in word_counter:
                word_dict[word] = len(word_dict)
    return word_dict
        
def build_word_dataset(word_dict,document_max_len):
    #if step == "train":
    x = list(map(lambda d: word_tokenize(clean_text(d)), df["question"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))

    y = list(map(lambda d: d,list(df["sql.agg"])))
    
    return x,y

def build_char_dataset(model, document_max_len, alphabet_size):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    #if step == "train":
    char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    for c in alphabet:
        char_dict[c] = len(char_dict)

    alphabet_size = len(alphabet) + 2

    x = list(map(lambda content: list(map(lambda d: char_dict.get(d, char_dict["<unk>"]), question.lower())), df["question"]))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [char_dict["<pad>"]], x))

    y = list(map(lambda d: d, list(df["sql.aggs"])))

    return x, y, alphabet_size

def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
        
        
        