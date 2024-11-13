import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% Cargo variables globales 

FOLDER = "./archivos/"
DATAFRAME_COMPLETO_DIR = FOLDER + "mnistic_dataframe.pkl"
DATAFRAME_E2_DIR = FOLDER + "mnistic_dataframe_EJ2.pkl"

X_DEV_DIR = FOLDER + "X_DEV.pkl"
Y_DEV_DIR = FOLDER + "Y_DEV.pkl"

X_HELDOUT_DIR = FOLDER + "X_HELDOUT.pkl"
Y_HELDOUT_DIR = FOLDER + "Y_HELDOUT.pkl"
    
PIXELES_POR_FILA = 28
CANTIDAD_ATRIBUTOS = PIXELES_POR_FILA*PIXELES_POR_FILA

NUMEROS_GRUPO_14 = [0,2,4,6,7]

def transformar_imagen_en_arreglo(imagen: np.ndarray):
    """
    Una imagen en este set de datos es de 28x28 pixeles. Concateno las filas de cada imagen para tener un sólo arreglo de 784 elementos. Estos serán los atributos para entrenar los modelos

    Args:
        imagen (np.ndarray): imagen a transformar, debe tener una forma de (28,28,1)

    Returns:
        np.ndarray: arreglo de 784 elementos
    """
    atributos_imagen = np.zeros(CANTIDAD_ATRIBUTOS)
    for i in range(0, CANTIDAD_ATRIBUTOS):
        # la división entera nos da la fila en la que trabajamos
        # ejemplo: 100//28 = 3
        fila = i // PIXELES_POR_FILA

        # el módulo nos da la columna en la que trabajamos
        # ejemplo 100 % 28 = 16
        columna = i % PIXELES_POR_FILA 

        # => nos queda el pixel (3,16)

        # 1ra dimensión, la fila del arreglo de pixeles
        # 2da dim, la columna del arreglo de pixeles
        # 3ra dim, el valor entre 0 y 255 de la 'blancura' del pixel
        # (3era dim tiene un sólo elemento)
        atributos_imagen[i] = imagen[fila][columna][0]
    return atributos_imagen

def transformar_arreglo_en_imagen(arreglo: np.ndarray):
    imagen = np.zeros((PIXELES_POR_FILA, PIXELES_POR_FILA, 1))
    for i in range(CANTIDAD_ATRIBUTOS):
        fila = i // PIXELES_POR_FILA 
        columna = i % PIXELES_POR_FILA
        imagen[fila][columna][0] = arreglo[i]
    return imagen

def crear_nombre_atributos():
    nombres = []
    for i in range(0, CANTIDAD_ATRIBUTOS):
        fila = i // PIXELES_POR_FILA
        columna = i % PIXELES_POR_FILA
        nombres.append(f"({fila},{columna})")
    return nombres

def crear_dataframe():
    data_imgs: np.ndarray = np.load('./archivos/mnistc_images.npy')
    data_chrs: np.ndarray = np.load('./archivos/mnistc_labels.npy')[:,np.newaxis]

    # Crear Dataframe
    columnas = crear_nombre_atributos()
    
    largo = len(data_imgs)
    matriz_imagenes = np.zeros((largo, 28*28))
    for i in range(largo):
        matriz_imagenes[i] = transformar_imagen_en_arreglo(data_imgs[i])

    df = pd.DataFrame(matriz_imagenes, columns=columnas)
    df["number"] = data_chrs

    df.to_pickle(DATAFRAME_COMPLETO_DIR)

    mask = df["number"].isin(NUMEROS_GRUPO_14)
    df_E2 = df[mask]
    df_E2.to_pickle(DATAFRAME_E2_DIR)
    
    x = df_E2.drop(["number"], axis=1)
    y = df_E2["number"]

    TRAIN = train_test_split(x,y, random_state=1, test_size=0.1)
    x_dev:pd.DataFrame =    TRAIN[0]
    x_eval:pd.DataFrame =   TRAIN[1]
    y_dev:pd.DataFrame =    TRAIN[2]
    y_eval:pd.DataFrame =   TRAIN[3]

    x_dev.to_pickle(X_DEV_DIR)
    y_dev.to_pickle(Y_DEV_DIR)

    x_eval.to_pickle(X_HELDOUT_DIR)
    y_eval.to_pickle(Y_HELDOUT_DIR)

def comprobar_dataframe():
    df = os.path.exists(DATAFRAME_COMPLETO_DIR)
    df_E2 = os.path.exists(DATAFRAME_E2_DIR)
    x_dev = os.path.exists(X_DEV_DIR)
    y_dev = os.path.exists(Y_DEV_DIR)
    
    x_heldout = os.path.exists(X_HELDOUT_DIR)
    y_heldout = os.path.exists(Y_HELDOUT_DIR)
    if df and df_E2 and x_dev and y_dev and x_heldout and y_heldout:
        return
    else:
        crear_dataframe()

#%%
class Imagenes:
    def __init__(self):
            self.data_imgs: np.ndarray = np.load('./archivos/mnistc_images.npy')
            self.data_chrs: np.ndarray = np.load('./archivos/mnistc_labels.npy')[:,np.newaxis]
    
            # Crear Dataframe
            comprobar_dataframe()
    
            self.df = pd.read_pickle(DATAFRAME_COMPLETO_DIR)
            self.df_E2 = pd.read_pickle(DATAFRAME_E2_DIR)
            self.atributos = self.df_E2.drop(["number"], axis=1)
            self.clases = self.df_E2["number"]
            self.x_dev = pd.read_pickle(X_DEV_DIR)
            self.y_dev = pd.read_pickle(Y_DEV_DIR)
            self.x_heldout = pd.read_pickle(X_HELDOUT_DIR)
            self.y_heldout = pd.read_pickle(Y_HELDOUT_DIR)
        
    def buscar_indices(self, numero: int):
            """ 
            Busco los índices que corresponden las imáganes 
    
            A partir de un número de entrada entero me devuelve una lista con las posiciones dónde hay imágenes que corresponden a ese número
            """
            indices = [i for i, num in enumerate(self.data_chrs) if num == numero]
            return indices
    
    def obtener_imagenes(self, numero: int):
            """_summary_
    
            Args:
                numero (int): el número que repreentan las imágenes 
    
            Returns:
                numpy.ndarray: un arreglo con las imagenes que representan a 'numero'
            """
            indices = self.buscar_indices(numero)
    
            # obtengo la forma del arreglo de imágenes
            imgs_shape = self.data_imgs.shape
            # me interesa guardar la 2da,3ra,4ta dimensión, pero no la 1ra
            # la 1ra dimensión (cantidad de imágenes) será la cantidad de índices que tenga
            new_shape = (len(indices),)+imgs_shape[1:]
    
            # este será el array donde guarde las nuevas imágenes
            imgs = np.zeros(shape=new_shape, dtype=np.uint8)
            for i in range(0,len(indices)):
                index = indices[i]
                imgs[i] = self.data_imgs[index].astype(np.uint8)  
            
            return imgs
