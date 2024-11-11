import helpers as hlps
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

FOLDER = "./archivos/"

def crear_path(n: int):
    res = FOLDER + f"exp_{n}_res.npy"    
    labels = FOLDER + f"exp_{n}_label.npy"

def cargar_experrimento(n: int): 
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
            arbol = tree.DecisionTreeClassifier(max_depth = altura_max) 
            arbol.fit(kf_x_train, kf_y_train)
            pred = arbol.predict(kf_x_test)
            exactitud = accuracy_score(kf_y_test, pred)
            resultados[i,j] = exactitud

    res,labels = crear_path(1)
    np.save(res, resultados)
    np.save(labels, alturas)


if __name__ == "__main__":
    imagenes = hlps.Imagenes()
    x = imagenes.atributos
    y = imagenes.clases
    x_dev, x_eval, y_dev, y_eval = train_test_split(x,y, random_state=1, test_size=0.1)

    experimento_1(x_dev, y_dev)