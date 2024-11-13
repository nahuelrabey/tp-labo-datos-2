# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:38:06 2024

@author: nahue
"""

#%%
# Imports
from sklearn.ensemble import RandomForestClassifier
import helpers as hlp
import experimentos as exp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

print("Hello World")
# %%
# Cargar imágenes
imagenes = hlp.Imagenes()

# %%
# Resultados experimento 1
exactitudes, labels = exp.cargar_resultados(1)
promedio = exactitudes.mean(axis=0)

plt.scatter(x = labels, y = promedio)
plt.show()

# Podemos ver que, a partir de la altura 10, llegamos a la asíntota de máxima
# exactitud, en este caso, del 80%

# %% 
# Experimento KNN

imagenes = hlp.Imagenes()
X = imagenes.x_dev
y = imagenes.y_dev

nsplits = 5
kf = KFold(n_splits=nsplits)
labels = []
resultados = np.zeros(nsplits)

cluster = KMeans(n_clusters=10, random_state=1, n_init=100)
cluster.fit(X)

labels.append(cluster.labels_)
pred = cluster.predict(imagenes.x_heldout)

exactitud = accuracy_score(imagenes.y_heldout, pred)
print(f"exactitud utilizando 10 clusters KNN: {exactitud}")
# No tiene muy buena pinta la dvd
# parece estar entre 8% y 20% de exactitud
#%%
# Experimento Random Forest
