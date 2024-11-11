import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Imagenes:
    def __init__(self):
        self.data_imgs: np.ndarray = np.load('./archivos/mnistc_images.npy')
        self.data_chrs: np.ndarray = np.load('./archivos/mnistc_labels.npy')[:,np.newaxis]
    
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
            numpi.ndarray: un arreglo con las imagenes que representan a 'numero'
        """
        indices = self.buscar_indices(numero)

        # obtengo la forma del arreglo de imágenes
        imgs_shape = self.data_imgs.shape
        # me interesa guardar la 2da,3ra,4ta dimensión, pero no la 1ra
        # la 1ra dimensión (cantidad de imágenes) será la cantidad de índices que tenga
        new_shape = (len(indices),)+imgs_shape[1:]

        # este será el array dónde cuarde las nuevas imágenes
        imgs = np.zeros(shape=new_shape)
        for i in range(0,len(indices)):
            index = indices[i]
            imgs[i] = self.data_imgs[index]
        
        return imgs
    
    def guardar_numero(self, numero: int):
        imgs = self.obtener_imagenes(numero)
        np.save("f/archivos/numero_{x}.npy", imgs)
    
    def plotaer_primeros(self, n: int):
        for i in range(5):
            img_raw = self.atributos.iloc[i].to_numpy()
            imagen = transformar_arreglo_en_imagen(img_raw)
            label = self.clases.iloc[i]

            plt.figure(figsize=(10,8))
            plt.imshow(imagen, cmap='gray')
            plt.title('caracter: ' + str(label))
            plt.axis('off')  
            plt.show()
    
    
    
PIXELES_POR_FILA = 28
CANTIDAD_ATRIBUTOS = PIXELES_POR_FILA*PIXELES_POR_FILA

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

        # => nos queda el pixe (3,16)

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

