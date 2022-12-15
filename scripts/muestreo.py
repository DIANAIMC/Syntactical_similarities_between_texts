#! /usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
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

print("\n > Almacenamos theta´s")
theta5_0=log5.theta[0][0]
theta5_1=log5.theta[0][1]

theta1_0=log1.theta[0][0]
theta1_1=log1.theta[0][1]

print("\n > Obtenemos bootstrap´s")

theta5_0_bootstrap = []
theta5_1_bootstrap = []

theta1_0_bootstrap = []
theta1_1_bootstrap = []

# Define the function that trains a logistic regression model
def train_model(bootstrap_sample):
    bootstrap_5x = bootstrap_sample['jaccard_similarity'].values.reshape(-1, 1)
    bootstrap_5y = bootstrap_sample['contains_ans'].values.reshape(-1, 1)
    log5_b = logit.Logit(X=bootstrap_5x, y=bootstrap_5y)
    log5_b.train()
    return (log5_b.theta[0][0], log5_b.theta[0][1])

# Define the main function that uses multiple processes to train the models
def parallel_train(bootstrap_sample_5, num_processes):
    # Create a pool of processes
    with multiprocessing.Pool(num_processes) as pool:
        # Use the pool of processes to train a model for each bootstrap sample
        results = pool.map(train_model, bootstrap_sample_5)

        # Return the list of results
        return results

# Define the list of bootstrap samples
bootstrap_sample_5 = [sample5.sample(n=len(sample5), replace=True) for _ in range(25)]

# Train the models using 2 processes
results = parallel_train(bootstrap_sample_5, 4)

for thetas in results:
    theta5_0_bootstrap.append(thetas[0])
    theta5_1_bootstrap.append(thetas[1])

print("Sin filtrar")
print(theta5_0_bootstrap)
print(theta5_0_bootstrap)

quantile_975 = np.quantile(theta5_0_bootstrap, 0.975)
quantile_025 = np.quantile(theta5_0_bootstrap, 0.025)
theta_confidence95_0 = np.where((theta5_0_bootstrap < quantile_975) & (theta5_0_bootstrap > quantile_025), theta5_0_bootstrap, np.nan)
theta_confidence95_0 = np.nan_to_num(theta_confidence95_0)
theta_confidence95_0 = [x for x in theta_confidence95_0 if x != 0]

quantile_975 = np.quantile(theta5_1_bootstrap, 0.975)
quantile_025 = np.quantile(theta5_1_bootstrap, 0.025)
theta_confidence95_1 = np.where((theta5_1_bootstrap < quantile_975) & (theta5_1_bootstrap > quantile_025), theta5_1_bootstrap, np.nan)
theta_confidence95_1 = np.nan_to_num(theta_confidence95_1)
theta_confidence95_1 = [x for x in theta_confidence95_1 if x != 0]

print("Filtrados")
print(theta_confidence95_0)
print(theta_confidence95_1)

#plt.scatter(range(len()), log1.loss_hist)
#os.chdir(f'{directorio_actual}/graficas')
#plt.savefig('confidence_interval_5.png')
