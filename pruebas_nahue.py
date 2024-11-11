# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:38:06 2024

@author: nahue
"""

# %%
# Pruebo cómo funciona pandas con las matrices

A = np.array([
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34],
    [41,42,43,44]
])
# ¿Qué busco? que cada fila sea fila y cada columna sea columna. Easy

DF = pd.DataFrame(A)

# Queda cómo quiero 😎
#     0   1   2   3
# 0  11  12  13  14
# 1  21  22  23  24
# 2  31  32  33  34
# 3  41  42  43  44

#%%
# Imports
import helpers as hlp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree
from sklearn.metrics import accuracy_score 

print("Hello World")
# %%
# Cargar imágenes
imagenes = hlp.Imagenes()

# %%
# Separamos un conjunto de desarrollo y otro de prueba
x = imagenes.atributos
y = imagenes.clases
x_dev, x_eval, y_dev, y_eval = train_test_split(x,y, random_state=1, test_size=0.1)