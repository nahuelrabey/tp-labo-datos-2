
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim 

#%% Cargar datos

# un array para las imágenes, otro para las etiquetas (por qué no lo ponen en el mismo array #$%@*)
data_imgs = np.load('./archivos/mnistc_images.npy')
data_chrs = np.load('./archivos/mnistc_labels.npy')[:,np.newaxis]

# mostrar forma del array:
# 1ra dimensión: cada una de las imágenes en el dataset
# 2da y 3ra dimensión: 28x28 píxeles de cada imagen
print(data_imgs.shape)
print(data_chrs.shape)


#%% Separamos el número de interés

indices = []
lista_data = []
lista_chrs = []

def separar_numero(x):
    global indices
    indices = [i for i, numero in enumerate(data_chrs) if numero == x]

def guardar_numero(x):
    global lista_data, lista_chrs
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
    


separar_numero(9)
guardar_numero(9)

#%% Determinamos similitud usando ssim

imgs0 = np.load('./archivos/numero_0.npy')[:,:,:,np.newaxis]
imgs1 = np.load('./archivos/numero_1.npy')[:,:,:,np.newaxis]
imgs2 = np.load('./archivos/numero_2.npy')[:,:,:,np.newaxis]
imgs3 = np.load('./archivos/numero_3.npy')[:,:,:,np.newaxis]    
imgs4 = np.load('./archivos/numero_4.npy')[:,:,:,np.newaxis]
imgs5 = np.load('./archivos/numero_5.npy')[:,:,:,np.newaxis]
imgs6 = np.load('./archivos/numero_6.npy')[:,:,:,np.newaxis]
imgs7 = np.load('./archivos/numero_7.npy')[:,:,:,np.newaxis]
imgs8 = np.load('./archivos/numero_8.npy')[:,:,:,np.newaxis]
imgs9 = np.load('./archivos/numero_9.npy')[:,:,:,np.newaxis]


ssim_valores = []
def comparar_ssim (x):
    for i in range(0,min(len(x), len(imgs7))-1):
        image_1 = imgs7[i,:,:,0]
        image_2 = x[i+1,:,:,0]
        valores = ssim(image_1,image_2)
        ssim_valores.append(valores)
    return np.mean(ssim_valores)
#%%
valores_comparacion = []

#%%
valores_comparacion = []

# Loop through files named imgs0, imgs1, ..., imgs9
for i in range(0,10):
    x = globals().get(f'imgs{i}')  # Get imgs0, imgs1, ..., imgs9 from global scope
    if x is not None:
        comparacion = comparar_ssim(x)
        valores_comparacion.append(comparacion)
    else:
        print(f"Warning: imgs{i} is not defined.") 



    


#%%
n_digit = 80
image5 = imgs[n_digit,:,:,0]
image_l5 = chrs[n_digit]


n_digit1 = 68
image15 = imgs[n_digit1,:,:,0]
image_l15 = chrs[n_digit1]





ssim(image15,image5)



