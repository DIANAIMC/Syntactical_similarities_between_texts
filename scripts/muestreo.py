#! /usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import logit

directorio_actual = os.getcwd()
os.chdir(f'{directorio_actual}/data')

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

def without_spaces(my_list):
    new_list = [x for x in my_list if x != '']
    return new_list

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

print('\n--------------------- Preprocessing ---------------------')
df = pd.read_json('data.json')
print("\n > Todo a lowercase")

df['context'] = df['context'].apply(str.lower)
df['questions'] = df['questions'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])
df['ans'] = df['ans'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])

print(" > Elimina stopwords")

with open('stopwords.txt', 'r') as f:
    words = f.read().split()
df['context'] = df['context'].apply(remove_words)
df['questions'] = df['questions'].apply(iterate_list,words=words)
df['ans'] = df['ans'].apply(iterate_list,words=words)
df['context'] = df['context'].str.split('.')

print(" > Elimina strings vacíos")

df['context'] = df['context'].apply(without_spaces)
df_exploded = df.explode('context')
df_exploded=df_exploded.set_index(['context']).apply(pd.Series.explode).reset_index()

print(" > Calculamos jaccard_similarity")

df_exploded['jaccard_similarity'] = df_exploded.apply(lambda x: jaccard_similarity(x['context'], x['questions']), axis=1)

print(" > Revisa que la oración contenga la respuesta")

df_exploded['contains_ans'] = df_exploded.apply(lambda x: words_in_string(x['ans'], x['context']), axis=1)

print(" > Obtenemos la tabla jaccard")

cols_to_select = ["jaccard_similarity", "contains_ans"]
df_j = df_exploded.loc[:, cols_to_select]

print('\n--------------------- Model training ---------------------')
os.chdir(f'{directorio_actual}/scripts')
print("\n > Tomamos una muestra de 0.05")

np.random.seed(123454321)
sample_size5 = 0.05 # Sample size as a percentage of the total population
sample5 = df_j.sample(frac=sample_size5)
sample_x5 = sample5.iloc[:,0].values.reshape(-1, 1)
sample_y5 = sample5.iloc[:,-1].values.reshape(-1, 1)
log5 = logit.Logit(X=sample_x5, y=sample_y5)
print("\n > Lo entrenamos")
log5.train()
plt.plot(range(len(log5.loss_hist)), log5.loss_hist)
os.chdir(f'{directorio_actual}/graficas')
plt.savefig('loss_project5.png')

print("\n > Tomamos una muestra de 0.01")
os.chdir(f'{directorio_actual}/scripts')
np.random.seed(123454321)
sample_size1 = 0.01 # Sample size as a percentage of the total population
sample1 = df_j.sample(frac=sample_size1)
sample_x1 = sample1.iloc[:,0].values.reshape(-1, 1)
sample_y1 = sample1.iloc[:,-1].values.reshape(-1, 1)
log1 = logit.Logit(X=sample_x1, y=sample_y1)
print("\n > Lo entrenamos")
log1.train()
plt.plot(range(len(log1.loss_hist)), log1.loss_hist)
os.chdir(f'{directorio_actual}/graficas')
plt.savefig('loss_project1.png')

#print(log5.theta[0][0])
#print(log1.theta[0][1])

bootstrap_samples_5 = [np.random.choice(sample5, size=len(sample5), replace=True) for _ in range(100)]
print(bootstrap_samples_5)
sample_x5_bootstrap = [bootstrap_samples.iloc[:,0].values.reshape(-1, 1)]
sample_y5_bootstrap = [bootstrap_samples.iloc[:,-1].values.reshape(-1, 1)]

bootstrap_samples_1 = [np.random.choice(sample1, size=len(sample1), replace=True) for _ in range(100)]
print(bootstrap_samples_1)
sample_x1_bootstrap = [bootstrap_samples.iloc[:,0].values.reshape(-1, 1)]
sample_y1_bootstrap = [bootstrap_samples.iloc[:,-1].values.reshape(-1, 1)]

theta_0 = []
theta_1 = []

for i in bootstrapsamples_5:
    lg5 = logit.Logit(X=sample_x5_bootstrap[i], Y=sample_y5_boostrap[i])
    lg5.train()
    theta_0.append(lg.theta[0][0])
    theta_1.append(lg.theta[0][1])
    
for i in bootstrapsamples_1:
    lg1 = logit.Logit(X=sample_x1_bootstrap[i], Y=sample_y1_bootstrap[i])
    lg1.train()
    theta_0.append(lg.theta[0][0])
    theta_1.append(lg.theta[0][1])
