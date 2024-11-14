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
