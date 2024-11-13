from sklearn import tree
import os
import pickle
from sklearn.metrics import accuracy_score
import helpers as hlps
import numpy as np

FOLDER = "./archivos/"
DECISION_DIR = FOLDER+"model_decision_tree.pkl"
def crear_modelo_decision_tree():
    imagenes = hlps.Imagenes()
    X = imagenes.x_dev
    y = imagenes.y_dev

    arbol = tree.DecisionTreeClassifier(max_depth = 10, random_state=1) 
    arbol.fit(X, y)

    with open(DECISION_DIR, 'wb') as file:
        pickle.dump(arbol, file)    

def cargar_modelo_decision_tree():
    if not os.path.exists(DECISION_DIR):
        crear_modelo_decision_tree()
    
    arbol: tree.DecisionTreeClassifier = None 
    with open('model.pkl', 'rb') as file:
        arbol = pickle.load(file)
    return arbol
    
RANDOM_FOREST = FOLDER+"random_forest.pkl"
def crear_modelo_random_forest():
    imagenes = hlps.Imagenes()
    X = imagenes.x_dev
    y = imagenes.y_dev

    arbol = tree.DecisionTreeClassifier(max_depth = 10, random_state=1) 
    arbol.fit(X, y)

    with open(DECISION_DIR, 'wb') as file:
        pickle.dump(arbol, file)    