from sklearn.ensemble import RandomForestClassifier
import helpers as hlps
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

FOLDER = "./archivos/"

def crear_path(name: str):
    res = FOLDER + f"exp_{name}_res.npy"    
    labels = FOLDER + f"exp_{name}_label.npy"
    return res, labels

def cargar_resultados(n: int): 
    res, labels = crear_path(n)
    return np.load(res), np.load(labels)

def experimento_decision_tree():
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
            arbol = tree.DecisionTreeClassifier(max_depth = altura_max, random_state=1) 
            arbol.fit(kf_x_train, kf_y_train)
            pred = arbol.predict(kf_x_test)
            exactitud = accuracy_score(kf_y_test, pred)
            resultados[i,j] = exactitud

    res,labels = crear_path("decision_tree")
    np.save(res, resultados)
    np.save(labels, alturas)

def experimento_random_forest():
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
            clf = RandomForestClassifier(random_state=1, max_depth=altura_max)
            clf.fit(kf_x_train, kf_y_train)
            pred = clf.predict(kf_x_test)
            exactitud = accuracy_score(kf_y_test, pred)
            resultados[i,j] = exactitud

    res,labels = crear_path("random_forest")
    np.save(res, resultados)
    np.save(labels, alturas)
    
if __name__ == "__main__":

    experimento_decision_tree()
    experimento_random_forest()