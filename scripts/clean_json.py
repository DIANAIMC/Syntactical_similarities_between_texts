#! /usr/bin/env python

import numpy as np
import pandas as pd
import os
import sys

df = pd.read_json('data.json')

print('\n--------------------- Preprocessing ---------------------')
print("\n > Todo a lowercase")
df['context'] = df['context'].apply(str.lower)
df['questions'] = df['questions'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])
df['ans'] = df['ans'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])

print(" > Elimina stopwords")

with open('stopwords.txt', 'r') as f:
    words = f.read().split()

def remove_words(text):
    text_words = text.split()
    text_words = [w for w in text_words if w not in words]
    text = ' '.join(text_words)
    return text

def iterate_list(my_list,words):
    result = []
    for elem in my_list:
        if type(elem) == str:
            elem = elem.replace("?"," ?")
            text_words = elem.split()
            text_words = [w for w in text_words if w not in words]
            text = ' '.join(text_words)
            result.append(text)
    return result

df['context'] = df['context'].apply(remove_words)
df['questions'] = df['questions'].apply(iterate_list,words=words)
df['ans'] = df['ans'].apply(iterate_list,words=words)

#aca empezamos a explotar las cosas.
#ahora tenemos que usar para todo el exploded. please remember.
df['context'] = df['context'].str.split('.')

def without_spaces(my_list):
    new_list = [x for x in my_list if x != '']
    return new_list

print(" > Elimina strings vacíos")

df['context'] = df['context'].apply(without_spaces)
df_exploded = df.explode('context')
df_exploded=df_exploded.set_index(['context']).apply(pd.Series.explode).reset_index()

def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)
    j = float(len(a.intersection(b))) / len(a.union(b))
    return j

def words_in_string(words, string):
    words_list = words.split()
    for word in words_list:
        if word not in string:
            return 0
    return 1

print(" > Calculamos jaccard_similarity")

df_exploded['jaccard_similarity'] = df_exploded.apply(lambda x: jaccard_similarity(x['context'], x['questions']), axis=1)

print(" > Revisa que la oración contenga la respuesta")

df_exploded['contains_ans'] = df_exploded.apply(lambda x: words_in_string(x['ans'], x['context']), axis=1)

print(" > Obtenemos la tabla jaccard")

cols_to_select = ["jaccard_similarity", "contains_ans"]
df_j = df_exploded.loc[:, cols_to_select]

print('\n--------------------- Model training ---------------------')
print("\n > Tomamos una muestra")

np.random.seed(123454321)
sample_size = 0.05 # Sample size as a percentage of the total population
sample = df_j.sample(frac=sample_size)
print(sample.head(50))
