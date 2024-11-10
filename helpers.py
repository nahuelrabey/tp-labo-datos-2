import numpy as np

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
    
    def guardar_numer(self, numero: int):
        imgs = self.obtener_imagenes(numero)
        np.save("f/archivos/numero_{x}.npy")
    
# def 

#Selecciono y guardo las imágenes correspondientes a esos índices 
def guardar_numero(x, indices, data_imgs, data_chrs):
    # global lista_data, lista_chrs
    lista_chrs = []
    lista_data = []
    for i in indices:
        im = data_imgs[i, :, :, 0]
        ch = data_chrs[i]
        lista_data.append(im)
        lista_chrs.append(ch)

    imgset = np.array(lista_data)
    img_chr = np.array(lista_chrs)
    imgset, img_chr
    np.save(f"./archivos/numero_{x}.npy", imgset)
    np.save(f"./archivos/numero_{x}_chrs.npy", img_chr)