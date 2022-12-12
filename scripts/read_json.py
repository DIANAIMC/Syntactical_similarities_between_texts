#!/usr/bin/env python

import pandas as pd
import os
import sys
# Agrega la ruta de la carpeta que contiene el archivo JSON a la ruta de búsqueda de Python
# Primero obtiene la ruta absoluta del archivo actual
#ruta_actual = os.path.dirname(os.path.abspath('read_json.py'))
# Luego, construye la ruta completa de la carpeta que contiene el archivo JSON
#ruta_carpeta = os.path.join(ruta_actual, '/home/dianamunnoz/Syntactical_similarities_between_texts/data')
# Finalmente, agrega la ruta de la carpeta a la ruta de búsqueda de Python
#print(ruta_carpeta)
#sys.path.append(ruta_carpeta)
df = pd.read_json('data.json')
print("Muestra los primeros 10 datos")
df.head(10)

