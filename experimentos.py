# %% Imports

import helpers as hlps
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

#%%
FOLDER = "./archivos/"

#%% Preset para gráficos
# Visualizaciones
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams['font.size'] = 20           
plt.rcParams['axes.labelsize'] = 20      
plt.rcParams['axes.titlesize'] = 20      
plt.rcParams['legend.fontsize'] = 16    
plt.rcParams['xtick.labelsize'] = 16      
plt.rcParams['ytick.labelsize'] = 16 
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

#%%

def crear_path(n: str):
    res = FOLDER + f"exp_{n}_res.npy"    
    labels = FOLDER + f"exp_{n}_label.npy"
    return res, labels
    
def cargar_resultados(n: int): 
    res, labels = crear_path(n)
    return np.load(res), np.load(labels)

def experimento_max_depth(criterion: str):
    imagenes = hlps.Imagenes()
    X = imagenes.x_dev
    y = imagenes.y_dev
    
    alturas = [1,2,3,5,10,20]
    nsplits = 5
    kf = KFold(n_splits=nsplits)
    resultados = np.zeros((nsplits, len(alturas)))
    split = kf.split(X)
    # print(len(split))
    for i, (train_index, test_index) in enumerate(split):
        kf_x_train = X.iloc[train_index]
        kf_x_test = X.iloc[test_index]
        
        kf_y_train = y.iloc[train_index]
        kf_y_test = y.iloc[test_index]
        
        for j, altura_max in enumerate(alturas):
            arbol = tree.DecisionTreeClassifier(max_depth = altura_max, criterion=criterion) 
            arbol.fit(kf_x_train, kf_y_train)
            pred = arbol.predict(kf_x_test)
            exactitud = accuracy_score(kf_y_test, pred)
            resultados[i,j] = exactitud

    res,labels = crear_path(f"max_depth_{criterion}")
    np.save(res, resultados)
    np.save(labels, alturas)
    
def cargar_experimento_max_depth(criterion:str):
    return cargar_resultados(f"max_depth_{criterion}")

def experimento_max_feature(criterion:str):
    """
    Usa el mejor criterio obtenido de los experimentos de criterio
    """
    imagenes = hlps.Imagenes()
    X = imagenes.x_dev
    y = imagenes.y_dev

    max_features = [None, "sqrt", "log2"]
    nsplits = 5
    kf = KFold(n_splits=nsplits)
    resultados = np.zeros((nsplits, len(max_features)))

    split = kf.split(X)
    # print(len(split))
    for i, (train_index, test_index) in enumerate(split):
        kf_x_train = X.iloc[train_index]
        kf_x_test = X.iloc[test_index]
        
        kf_y_train = y.iloc[train_index]
        kf_y_test = y.iloc[test_index]

        for j, feature in enumerate(max_features):
            arbol = tree.DecisionTreeClassifier(max_depth = 10, criterion='gini', max_features=feature) 
            arbol.fit(kf_x_train, kf_y_train)
            pred = arbol.predict(kf_x_test)
            exactitud = accuracy_score(kf_y_test, pred)
            resultados[i,j] = exactitud

    res,labels = crear_path(f"max_feature_{criterion}")
    np.save(res, resultados)
    np.save(labels, np.array([str(f) for f in max_features], dtype=str))

def cargar_experimento_max_features(criterion: str):
    return cargar_resultados(f"max_feature_{criterion}")


#%% Ejecuto experimentos 
print("Creando Gini")
experimento_max_depth("gini")
print("Creando Entropy")
experimento_max_depth("entropy")
print("Creando Log Loss")
experimento_max_depth("log_loss")
print("Listo")
#%% Cargo experimentos
acc_gini, labels = cargar_experimento_max_depth("gini")
acc_entropy, _ = cargar_experimento_max_depth("entropy")
acc_log_loss, _ = cargar_experimento_max_depth("log_loss")

print(acc_gini.shape)

acc_gini = acc_gini.mean(axis=0)
acc_entropy = acc_entropy.mean(axis=0)
acc_log_loss = acc_log_loss.mean(axis=0)
#%% Graficamos la precisión promedio en función de la profundidad del árbol


plt.plot(labels, acc_gini, marker='o', linestyle='-', label="Gini")
plt.plot(labels, acc_entropy, marker='o', linestyle='-', label="Entropy")
plt.plot(labels, acc_log_loss, marker='o', linestyle='-', label="log_loss")
plt.legend()
plt.xlabel("Profundidad del árbol de decisión")
plt.ylabel("Exactitud promedio")
plt.title("Exactitud vs Profundidad del Árbol de Decisión")
plt.grid(True)
plt.show()

#%% Experimento Max Features
print("Creando Gini")
experimento_max_feature("gini")
print("Creando Entropy")
experimento_max_feature("entropy")
print("Creando Log Loss")
experimento_max_feature("log_loss")
print("Listo")

#%% Experimento  max_features
acc_gini, labels = cargar_experimento_max_features("gini")
acc_entropy, _ = cargar_experimento_max_features("entropy")
acc_log_loss, _ = cargar_experimento_max_features("log_loss")


acc_gini = acc_gini.mean(axis=0)
acc_entropy = acc_entropy.mean(axis=0)
acc_log_loss = acc_log_loss.mean(axis=0)


#%% Plotear max_features
max_features = []
value = []
criterio = []

for i in range(3):
    value.append(acc_gini[i]) 
    criterio.append("gini")
    max_features.append(labels[i])

    value.append(acc_entropy[i]) 
    criterio.append("entropy")
    max_features.append(labels[i])

    value.append(acc_log_loss[i]) 
    criterio.append("log_loss")
    max_features.append(labels[i])

df = pd.DataFrame(data={"max_features":max_features, "value":value, "criterio":criterio})

g = sns.catplot(
    data=df, kind="bar",
    x="max_features", y="value", hue="criterio",
    errorbar="sd"
)

g.despine(left=True)
g.set_axis_labels("max features", "Exactitud")
g.legend.set_title("")
plt.show()