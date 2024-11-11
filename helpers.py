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
        np.save("f/archivos/numero_{x}.npy")
    
