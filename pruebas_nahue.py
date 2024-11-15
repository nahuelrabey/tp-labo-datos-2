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

imagenes = hlp.Imagenes()
numbers = imagenes.df_E2["number"]
numbers_count = numbers.value_counts()

total = numbers.count()
percentage = (numbers_count/total)*100
avg_percentage = percentage.sum()/percentage.count()
print(avg_percentage)

#%% Prueba layout 5 graficos

import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(x)
y4 = np.log(x+1)
y5 = x**2

# Create a 2x3 grid of subplots
fig1, axs1 = plt.subplots(1,3, figsize=(12, 4))
fig2, axs2 = plt.subplots(1,2, figsize=(12, 4))

# Plot on each subplot
axs1[0].plot(x, y1)
axs1[0].set_title('Sine Wave')

axs1[1].plot(x, y2)
axs1[1].set_title('Cosine Wave')

axs1[2].plot(x, y3)
axs1[2].set_title('Exponential')

axs2[0].plot(x, y4)
axs2[0].set_title('Logarithmic')

axs2[1].plot(x, y5)
axs2[1].set_title('Quadratic')

plt.show()