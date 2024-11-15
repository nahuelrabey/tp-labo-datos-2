# %% Imports

import os
import helpers as hlps
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
    labels = FOLDER + f"exp_{n}_label.npy"
    acc = FOLDER + f"exp_{n}_cc.npy"    
    # precision = FOLDER + f"exp_{n}_precision.npy"
    # recall = FOLDER + f"exp_{n}_recall.npy"
    return labels, acc
    
def cargar_resultados(n: int): 
    paths = crear_path(n)
    res = [np.load(p) for p in paths]
    return res

def experimento_max_depth():
    imagenes = hlps.Imagenes()
    X = imagenes.x_dev
    y = imagenes.y_dev
    
    alturas = [1,2,3,5,10,15,20]
    nsplits = 5
    kf = KFold(n_splits=nsplits)
    acc = np.zeros((nsplits, len(alturas)))
    # precision = np.zeros((nsplits, len(alturas)))
    # recall = np.zeros((nsplits, len(alturas)))
    split = kf.split(X)
    # print(len(split))
    for i, (train_index, test_index) in enumerate(split):
        kf_x_train = X.iloc[train_index]
        kf_x_test = X.iloc[test_index]
        
        kf_y_train = y.iloc[train_index]
        kf_y_test = y.iloc[test_index]
        
        for j, altura_max in enumerate(alturas):
            arbol = tree.DecisionTreeClassifier(max_depth = altura_max, criterion='gini') 
            arbol.fit(kf_x_train, kf_y_train)
            pred = arbol.predict(kf_x_test)
            a = metrics.accuracy_score(kf_y_test, pred)
            # p = metrics.precision_score(kf_y_test, pred)
            # r= metrics.recall_score(kf_y_test, pred)
            acc[i,j] = a
            # precision[i,j] = p
            # recall[i,j] = r


    paths = crear_path("max_depth")
    np.save(paths[0], alturas)
    np.save(paths[1], acc)
    # np.save(paths[2], precision)
    # np.save(paths[3], recall)
    
def cargar_experimento_max_depth():
    return cargar_resultados("max_depth")

def experimento_criterion(criterion: str):
    imagenes = hlps.Imagenes()
    X = imagenes.x_dev
    y = imagenes.y_dev
    
    alturas = [1,2,3,5,10,15,20]
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
            exactitud = metrics.accuracy_score(kf_y_test, pred)
            resultados[i,j] = exactitud

    paths = crear_path(f"max_depth_{criterion}")
    np.save(paths[0], alturas)
    np.save(paths[1], resultados)
    
def cargar_experimento_criterion(criterion:str):
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
            exactitud = metrics.accuracy_score(kf_y_test, pred)
            resultados[i,j] = exactitud

    paths = crear_path(f"max_feature_{criterion}")
    np.save(paths[0], np.array([str(f) for f in max_features], dtype=str))
    np.save(paths[1], resultados)

def cargar_experimento_max_features(criterion: str):
    return cargar_resultados(f"max_feature_{criterion}")

MODELO_DIR = FOLDER + "arbol.pkl"
def crear_modelo():
    img = hlps.Imagenes()
    X = img.x_dev
    y = img.y_dev

    arbol = tree.DecisionTreeClassifier(max_depth=10, criterion="gini", max_features=None)
    arbol.fit(X,y)
    
    with open(MODELO_DIR, 'wb') as file:
        pickle.dump(arbol, file)

def cargar_modelo():
    if not os.path.exists(MODELO_DIR):
        crear_modelo()
    
    arbol: tree.DecisionTreeClassifier = None
    with open(MODELO_DIR, 'rb') as file:
        arbol = pickle.load(file)
    return arbol

#%% Cargo experimento max depth
print("Creando max_depth")
experimento_max_depth()
print("listo")

#%% Cargo experimento
labels, acc= cargar_experimento_max_depth()
acc = acc.mean(axis=0)

#%% Grafico
plt.plot(labels, acc, marker='o', linestyle='-')
plt.xlabel("Profundidad del árbol de decisión")
plt.ylabel("Exactitud promedio")
plt.title("Exactitud vs Profundidad del Árbol de Decisión")
plt.grid(True)
plt.tight_layout()
plt.savefig('./imagenes/exactitud_vs_profundidad.pdf')
plt.show()

#%% Experimento Criterio
print("Creando Gini")
experimento_criterion("gini")
print("Creando Entropy")
experimento_criterion("entropy")
print("Creando Log Loss")
experimento_criterion("log_loss")
print("Listo")

#%% Cargo experimentos
labels, acc_gini = cargar_experimento_criterion("gini")
_, acc_entropy = cargar_experimento_criterion("entropy")
_, acc_log_loss = cargar_experimento_criterion("log_loss")

print(acc_gini.shape)

acc_gini = acc_gini.mean(axis=0)
acc_entropy = acc_entropy.mean(axis=0)
acc_log_loss = acc_log_loss.mean(axis=0)

#%% Graficamos la precisión promedio en función de la profundidad del árbol


plt.plot(labels, acc_gini, marker='o', linestyle='-', label="Gini")
plt.plot(labels, acc_entropy, marker='o', linestyle='-', label="Entropy")
plt.plot(labels, acc_log_loss, marker='o', linestyle='-', label="Log loss")
plt.legend()
plt.xlabel("Profundidad del árbol de decisión")
plt.ylabel("Exactitud promedio")
plt.title("Exactitud vs Profundidad y Criterio de selección")
plt.grid(True)
plt.tight_layout()
plt.savefig('./imagenes/exactitud_vs_profundidad_vs_criterio.pdf')
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
labels, acc_gini= cargar_experimento_max_features("gini")
_, acc_entropy = cargar_experimento_max_features("entropy")
_, acc_log_loss = cargar_experimento_max_features("log_loss")


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

scores_df = pd.DataFrame(data={"max_features":max_features, "value":value, "criterio":criterio})


# Crear gráfico de barras
g = sns.catplot(
    data=scores_df, kind="bar",
    x="max_features", y="value", hue="criterio",
    errorbar="sd",
    height=8, aspect=1.25
)

g.despine(left=True)
g.set_axis_labels("max_features", "Exactitud promedio")
g.legend.set_title('')
plt.title("Exactitud vs Características máximas y Criterio de selección")
plt.savefig('./imagenes/exactitud_vs_max_features_vs_criterio.pdf')
plt.show()

#%% Cargamos el modelo
img = hlps.Imagenes()
arbol = cargar_modelo()
predict = arbol.predict(img.x_heldout)

# Utilizamos para métricas con un estilo OvR
label_binarizer = LabelBinarizer()
label_binarizer.fit(img.y_dev)

# mps dice realmente en qué clase está
y_digit_heldout = label_binarizer.transform(img.y_heldout)

# nos da la probabilidad de que entre en tal clase
predict_proba = arbol.predict_proba(img.x_heldout)

#%% Calculamos puntajes
# TODO: Hacer un gráfico de esto
exactitud = metrics.accuracy_score(img.y_heldout, predict)
# Podemos utilizar 'macro' pues cada clase tiene, en promedio, una aparición del
# 20%, y al ser 5 clase, está todo bien balanceado para llegar al 100% de la
# muestra
precision = metrics.precision_score(img.y_heldout, predict, average="macro")
recall = metrics.recall_score(img.y_heldout, predict, average="macro")

roc_auc = np.zeros(5)
for i in range(5):
    res = metrics.roc_auc_score(
        y_true = y_digit_heldout[:,i], 
        y_score = predict_proba[:,i], 
        average="macro", 
        multi_class="ovr", 
        labels=hlps.NUMEROS_GRUPO_14
    )
    roc_auc[i] = res

roc_auc_mean = roc_auc.mean()


#%% Plot scores
values = [exactitud, precision, recall, roc_auc_mean]
score = ["exactitud","precisión", "recall", "AUC-ROC"]
scores_df = pd.DataFrame({"values": values, "score":score})
sns.barplot(data=scores_df, x="score",y="values")
plt.show()

#%% Curvas ROC
# Para calcular una curva ROC para cada clase. Cómo el método está pensado en
# términos binarios, no para una multiclase, hay que hacer una transformación.
# Para esto, se considera la estrategia One-vs-All. Es decir, "true positive"
# son los que entren en la clase setudiada, "false positive" todos los demás, lo
# mismo para "true negative" y "false negative"

# metrics.roc_curve(img.y_heldout, predict_proba)
label_binarizer = LabelBinarizer()
label_binarizer.fit(img.y_dev)

axs = []
figs = []
for digit in hlps.NUMEROS_GRUPO_14:
    vs_digtos = np.setdiff1d(hlps.NUMEROS_GRUPO_14, [digit])
    print(vs_digtos)
    digito_id = np.flatnonzero(label_binarizer.classes_ == digit)[0]
    display = metrics.RocCurveDisplay.from_predictions(
        y_true = y_digit_heldout[:, digito_id], 
        y_pred = predict_proba[:, digito_id],
        name=f"{digit} vs el resto de dígitos",
        color="darkorange",
        plot_chance_level=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"One-vs-Rest ROC curves:\n{digit} vs {vs_digtos}",
    )
    axs.append(display.ax_)
    plt.savefig(f"./imagenes/roc_curve_{digit}.pdf")
    plt.show()

# metrics.roc_auc_score

#%% Matriz de confusión
matriz = metrics.confusion_matrix(img.y_heldout, predict)
sns.heatmap(matriz, annot=False, cmap="YlGnBu", cbar=True)
plt.savefig('./imagenes/matriz_confusion_modelo.pdf')
plt.show()