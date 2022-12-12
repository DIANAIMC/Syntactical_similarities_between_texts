#!/usr/bin/env python

import pandas as pd
import os
import sys

df = pd.read_json('data.json')

print("")
df['context'] = df['context'].apply(str.lower)
df['questions'] = df['questions'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])
df['ans'] = df['ans'].apply(lambda x: [e.lower() if isinstance(e, str) else e for e in x])
print("Muestra los primeros 10 datos")
print(df.head(10))

