#!/usr/bin/env python

import pandas as pd
import os
import sys

df = pd.read_json('data.json')


print("Todo a lowercase")
df['context'] = df['context'].apply(str.lower)
df['questions'] = df['questions'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])
df['ans'] = df['ans'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])
print("Muestra los primeros 10 datos")
print(df.head(10))

print("Eliminar stopwords")
# Read the words from the text file into a list
with open('stopwords.txt', 'r') as f:
    words = f.read().split()

# Define a custom function that will remove the words from the text file
def remove_words(text):
    # Split the text into a list of words
    text_words = text.split()
    
    # Remove any words that are in the list of words from the text file
    text_words = [w for w in text_words if w not in words]
    
    # Reconstruct the text from the remaining words
    text = ' '.join(text_words)
    
    return text

# Apply the custom function to each element of the 'column1' column
df['context'] = df['context'].apply(remove_words)
