#!/usr/bin/env python

import pandas as pd
import os
import sys

df = pd.read_json('data.json')

print("	> Todo a lowercase")

df['context'] = df['context'].apply(str.lower)
df['questions'] = df['questions'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])
df['ans'] = df['ans'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])
#print(df.head(10))

print("	> Elimina stopwords")

with open('stopwords.txt', 'r') as f:
    words = f.read().split()

#def remove_list_of_words(column_to_modify, stopwords):
#    for word in stopwords:
#        column_to_modify = column_to_modify.replace(word, "")
#    return column_to_modify
#remove_list_of_words(df['context'], words)

#df['context'] = remove_list_of_words(df['context'],words)

def remove_words(text):
    text_words = text.split()
    text_words = [w for w in text_words if w not in words]
    text = ' '.join(text_words)
    return text

df['context'] = df['context'].apply(remove_words)
#print(df.head(10))
