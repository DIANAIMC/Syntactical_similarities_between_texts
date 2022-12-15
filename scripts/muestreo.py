#! /usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import sys
import logit
import statistics
from scipy.stats import norm, stats

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

# Define the function that trains a logistic regression model
def train_model(bootstrap_sample):
    bootstrap_x = bootstrap_sample['jaccard_similarity'].values.reshape(-1, 1)
    bootstrap_y = bootstrap_sample['contains_ans'].values.reshape(-1, 1)
    log_b = logit.Logit(X=bootstrap_x, y=bootstrap_y)
    log_b.train()
    return (log_b.theta[0][0], log_b.theta[0][1])

def append_results(result):
    results.append(result)

# Define the main function that uses multiple processes to train the models
def parallel_train(bootstrap_sample, num_processes):
    # Create a pool of processes
    results=[]
    with multiprocessing.Pool(num_processes) as pool:
        # Use the pool of processes to train a model for each bootstrap sample
        #results = pool.map(train_model, bootstrap_sample_5)
        for x in bootstrap_sample:
            pool.apply_async(train_model, x,callback=append_results)
        pool.close()
        pool.join()
        # Return the list of results
    return results

if __name__ == "__main__":
    directorio_actual = os.getcwd()
    os.chdir(f'{directorio_actual}/data')
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
    print("\n > Bootstrap´s")
    print("\n 	> Obtenemos bootstrap´s de 1%")

    theta_0_bootstrap = []
    theta_1_bootstrap = []

    # Define the list of bootstrap samples
    bootstrap_sample_1 = [sample1.sample(n=len(sample1), replace=True) for _ in range(100)]
    # Train the models using 2 processes
    # results=parallel_train(bootstrap_sample_5, 4)
    results=[]
    pool=multiprocessing.Pool(processes=10)
    for rep in bootstrap_sample_1:
        pool.apply_async(train_model,[rep],callback=append_results)
    pool.close()
    pool.join()
    for thetas in results:
        theta_0_bootstrap.append(thetas[0])
        theta_1_bootstrap.append(thetas[1])

    print("\n 	> Obtenemos las gráficas")
    #theta 0
    os.chdir(f'{directorio_actual}/graficas')
    x=np.linspace(0, 100, len(theta_0_bootstrap))
    y=theta_0_bootstrap 

    p5 = np.percentile(theta_0_bootstrap, 5)
    p95 = np.percentile(theta_0_bootstrap, 95)
    plt.clf()
    plt.rcParams["figure.figsize"] = (20,10)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Theta 0 vs Theta 1')
    x=np.linspace(0, 100, len(theta_0_bootstrap))
    y=theta_0_bootstrap 
    ax1.plot(x, y, 'o')
    x=np.linspace(0, 100, len(theta_1_bootstrap))
    y=theta_1_bootstrap 
    ax2.plot(x, y, 'o')

    for i, val in enumerate(theta_0_bootstrap):
        if val <= p5:
            ax1.scatter(i, val, color='red')
        elif val>=p95:
            ax1.scatter(i, val, color='red')
    p5 = np.percentile(theta_1_bootstrap, 5)
    p95 = np.percentile(theta_1_bootstrap, 95)

    ax1.axhline(y = np.percentile(theta_0_bootstrap, 5), color = 'r', label = 'axvline - full height')
    ax1.axhline(y = np.percentile(theta_0_bootstrap, 95), color = 'r', label = 'axvline - full height')
    #theta 1
    for i, val in enumerate(theta_1_bootstrap):
        if val <= p5:
            ax2.scatter(i, val, color='yellow')
        elif val>=p95:
            ax2.scatter(i, val, color='yellow')

    ax2.axhline(y = np.percentile(theta_1_bootstrap, 5), color = 'r', label = 'axvline - full height')
    ax2.axhline(y = np.percentile(theta_1_bootstrap, 95), color = 'r', label = 'axvline - full height')
    plt.savefig("Scatter_theta_1.png")
    
    #PARA LA NORMALIZADA Y DISTRIBUCION
    plt.clf()
    desviacion_0 = statistics.stdev(theta_0_bootstrap)
    media_0 = statistics.mean(theta_0_bootstrap)
    ci_0 = norm.interval(0.95, loc=media_0, scale=desviacion_0)

    desviacion_1 = statistics.stdev(theta_1_bootstrap)
    media_1 = statistics.mean(theta_1_bootstrap)
    ci_1 = norm.interval(0.95, loc=media_1, scale=desviacion_1)


    x_norm_0 = np.linspace(media_0 - 3*desviacion_0, media_0 + 3*desviacion_0, 100)
    y_norm_0 = norm.pdf(x_norm_0, media_0, desviacion_0)
    x_norm_1 = np.linspace(media_1 - 3*desviacion_1, media_1 + 3*desviacion_1, 100)
    y_norm_1 = norm.pdf(x_norm_1, media_1, desviacion_1)

    plt.rcParams["figure.figsize"] = (20,10)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Distribucion Theta 0 vs Theta 1')
    ax1.plot(x_norm_0, y_norm_0)
    ax2.plot(x_norm_1, y_norm_1)
    ax1.axvline(x = np.percentile(x_norm_0, 5), color = 'r', label = 'axvline - full height')
    ax1.axvline(x = np.percentile(x_norm_0, 95), color = 'r', label = 'axvline - full height')
    ax2.axvline(x = np.percentile(x_norm_1, 5), color = 'r', label = 'axvline - full height')
    ax2.axvline(x = np.percentile(x_norm_1, 95), color = 'r', label = 'axvline - full height')
    plt.savefig("Distribucion_normalizada_1.png")

    #DISTRIBUCION
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Theta 0 vs Theta 1')
    #x=np.linspace(0, 100, len(theta_0_bootstrap))
    #y=theta_0_bootstrap 
    #ax1.plot(x, y, 'o')
    df = pd.DataFrame(theta_0_bootstrap)
    df.plot.kde(ax=ax1)

    p5 = np.percentile(theta_0_bootstrap, 5)
    p95 = np.percentile(theta_0_bootstrap, 95)

    #for i, val in enumerate(theta_0_bootstrap):
    #    if val <= p5:
    #        ax1.scatter(i, val, color='red')
    #    elif val>=p95:
    #        ax1.scatter(i, val, color='red')
    
    ax1.axvline(x = np.percentile(theta_0_bootstrap, 5), color = 'r', label = 'axvline - full height')
    ax1.axvline(x = np.percentile(theta_0_bootstrap, 95), color = 'r', label = 'axvline - full height')

    #x=np.linspace(0, 100, len(theta_1_bootstrap))
    #y=theta_1_bootstrap 
    #ax2.plot(x, y, 'o')
    df = pd.DataFrame(theta_1_bootstrap)
    df.plot.kde(ax=ax2)

    p5 = np.percentile(theta_1_bootstrap, 5)
    p95 = np.percentile(theta_1_bootstrap, 95)

    #for i, val in enumerate(theta_1_bootstrap):
    #    if val <= p5:
    #        ax2.scatter(i, val, color='yellow')
    #    elif val>=p95:
    #        ax2.scatter(i, val, color='yellow')

    ax2.axvline(x = np.percentile(theta_1_bootstrap, 5), color = 'r', label = 'axvline - full height')
    ax2.axvline(x = np.percentile(theta_1_bootstrap, 95), color = 'r', label = 'axvline - full height')

    plt.savefig("Distribucion_1.png")
    
    
    print("\n   > Obtenemos bootstrap´s de 5%")

    theta_0_bootstrap = []
    theta_1_bootstrap = []

    # Define the list of bootstrap samples
    bootstrap_sample_1 = [sample5.sample(n=len(sample5), replace=True) for _ in range(100)]
    # Train the models using 2 processes
    # results=parallel_train(bootstrap_sample_5, 4)
    results=[]
    pool=multiprocessing.Pool(processes=10)
    for rep in bootstrap_sample_1:
        pool.apply_async(train_model,[rep],callback=append_results)
    pool.close()
    pool.join()
    for thetas in results:
        theta_0_bootstrap.append(thetas[0])
        theta_1_bootstrap.append(thetas[1])

    print("\n   > Obtenemos las gráficas")
    #theta 0
    os.chdir(f'{directorio_actual}/graficas')
    print("\n 	> Obtenemos las gráficas")
    #theta 0
    os.chdir(f'{directorio_actual}/graficas')
    x=np.linspace(0, 100, len(theta_0_bootstrap))
    y=theta_0_bootstrap 

    p5 = np.percentile(theta_0_bootstrap, 5)
    p95 = np.percentile(theta_0_bootstrap, 95)
    plt.clf()
    plt.rcParams["figure.figsize"] = (20,10)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Theta 0 vs Theta 1')
    x=np.linspace(0, 100, len(theta_0_bootstrap))
    y=theta_0_bootstrap 
    ax1.plot(x, y, 'o')
    x=np.linspace(0, 100, len(theta_1_bootstrap))
    y=theta_1_bootstrap 
    ax2.plot(x, y, 'o')

    for i, val in enumerate(theta_0_bootstrap):
        if val <= p5:
            ax1.scatter(i, val, color='red')
        elif val>=p95:
            ax1.scatter(i, val, color='red')
    p5 = np.percentile(theta_1_bootstrap, 5)
    p95 = np.percentile(theta_1_bootstrap, 95)

    ax1.axhline(y = np.percentile(theta_0_bootstrap, 5), color = 'r', label = 'axvline - full height')
    ax1.axhline(y = np.percentile(theta_0_bootstrap, 95), color = 'r', label = 'axvline - full height')
    #theta 1
    for i, val in enumerate(theta_1_bootstrap):
        if val <= p5:
            ax2.scatter(i, val, color='yellow')
        elif val>=p95:
            ax2.scatter(i, val, color='yellow')

    ax2.axhline(y = np.percentile(theta_1_bootstrap, 5), color = 'r', label = 'axvline - full height')
    ax2.axhline(y = np.percentile(theta_1_bootstrap, 95), color = 'r', label = 'axvline - full height')
    plt.savefig("Scatter_theta_5.png")
    
    #PARA LA NORMALIZADA Y DISTRIBUCION
    plt.clf()
    desviacion_0 = statistics.stdev(theta_0_bootstrap)
    media_0 = statistics.mean(theta_0_bootstrap)
    ci_0 = norm.interval(0.95, loc=media_0, scale=desviacion_0)

    desviacion_1 = statistics.stdev(theta_1_bootstrap)
    media_1 = statistics.mean(theta_1_bootstrap)
    ci_1 = norm.interval(0.95, loc=media_1, scale=desviacion_1)


    x_norm_0 = np.linspace(media_0 - 3*desviacion_0, media_0 + 3*desviacion_0, 100)
    y_norm_0 = norm.pdf(x_norm_0, media_0, desviacion_0)
    x_norm_1 = np.linspace(media_1 - 3*desviacion_1, media_1 + 3*desviacion_1, 100)
    y_norm_1 = norm.pdf(x_norm_1, media_1, desviacion_1)

    plt.rcParams["figure.figsize"] = (20,10)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Distribucion Theta 0 vs Theta 1')
    ax1.plot(x_norm_0, y_norm_0)
    ax2.plot(x_norm_1, y_norm_1)
    ax1.axvline(x = np.percentile(x_norm_0, 5), color = 'r', label = 'axvline - full height')
    ax1.axvline(x = np.percentile(x_norm_0, 95), color = 'r', label = 'axvline - full height')
    ax2.axvline(x = np.percentile(x_norm_1, 5), color = 'r', label = 'axvline - full height')
    ax2.axvline(x = np.percentile(x_norm_1, 95), color = 'r', label = 'axvline - full height')
    plt.savefig("Distribucion_normalizada_5.png")

    #DISTRIBUCION
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Theta 0 vs Theta 1')
    #x=np.linspace(0, 100, len(theta_0_bootstrap))
    #y=theta_0_bootstrap 
    #ax1.plot(x, y, 'o')
    df = pd.DataFrame(theta_0_bootstrap)
    df.plot.kde(ax=ax1)

    p5 = np.percentile(theta_0_bootstrap, 5)
    p95 = np.percentile(theta_0_bootstrap, 95)

    #for i, val in enumerate(theta_0_bootstrap):
    #    if val <= p5:
    #        ax1.scatter(i, val, color='red')
    #    elif val>=p95:
    #        ax1.scatter(i, val, color='red')
    
    ax1.axvline(x = np.percentile(theta_0_bootstrap, 5), color = 'r', label = 'axvline - full height')
    ax1.axvline(x = np.percentile(theta_0_bootstrap, 95), color = 'r', label = 'axvline - full height')

    #x=np.linspace(0, 100, len(theta_1_bootstrap))
    #y=theta_1_bootstrap 
    #ax2.plot(x, y, 'o')
    df = pd.DataFrame(theta_1_bootstrap)
    df.plot.kde(ax=ax2)

    p5 = np.percentile(theta_1_bootstrap, 5)
    p95 = np.percentile(theta_1_bootstrap, 95)

    #for i, val in enumerate(theta_1_bootstrap):
    #    if val <= p5:
    #        ax2.scatter(i, val, color='yellow')
    #    elif val>=p95:
    #        ax2.scatter(i, val, color='yellow')

    ax2.axvline(x = np.percentile(theta_1_bootstrap, 5), color = 'r', label = 'axvline - full height')
    ax2.axvline(x = np.percentile(theta_1_bootstrap, 95), color = 'r', label = 'axvline - full height')

    plt.savefig("Distribucion_5.png")
    