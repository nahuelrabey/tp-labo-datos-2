from sklearn.ensemble import RandomForestClassifier
import helpers as hlps
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
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

digitos_especificos = [0,2,4,6,7]
criterio = 'gini'

def crear_path(n: int):
    res = FOLDER + f"exp_{n}_res.npy"    
    labels = FOLDER + f"exp_{n}_label.npy"
    return res, labels
    
def cargar_experimento(n: int): 
    res, labels = crear_path(n)
    return np.load(res), np.load(labels)

def experimento_1(X, y):
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
            arbol = tree.DecisionTreeClassifier(max_depth = altura_max,criterion = criterio) 
            arbol.fit(kf_x_train, kf_y_train)
            pred = arbol.predict(kf_x_test)
            exactitud = accuracy_score(kf_y_test, pred)
            resultados[i,j] = exactitud
            
    res,labels = crear_path(1)
    np.save(res, resultados)
    np.save(labels, alturas)
    
if __name__ == "__main__":
    imagenes = hlps.Imagenes()
    mask = imagenes.clases.isin(digitos_especificos)  # Filtramos las imágenes para obtener solo de los dígitos específicos
    x = imagenes.atributos[mask]
    y = imagenes.clases[mask]
    x_dev, x_eval, y_dev, y_eval = train_test_split(x,y, random_state=1, test_size=0.1)
    experimento_1(x_dev, y_dev)
    
#%% Graficamos la precisión promedio en función de la profundidad del árbol
res, labels = cargar_experimento(1)
exactitud_promedio = res.mean(axis=0)

plt.plot(labels, exactitud_promedio, marker='o', linestyle='-')
plt.xlabel("Profundidad del árbol de decisión")
plt.ylabel("Exactitud promedio")
plt.title("Exactitud vs Profundidad del Árbol de Decisión")
plt.grid(True)
plt.show()    
plt.savefig('precision vs profundidad_entropia.pdf')
