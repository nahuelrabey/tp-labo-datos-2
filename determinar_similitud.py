#%%===========================================================================
# Grupo 14: Nahuel Rabey y Sammy Vallejo 
#Este código sirve para el análisis exploratorio del dataset, se utiliza el método ssim para determinar 
#similitud tanto entre las imágenes que corresponden a un único dígito como con las imágenes que corresponden  
#a los distintos dígitos. 
#=============================================================================

#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim 
import seaborn as sns

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


#%% Cargar datos crudos

# un array para las imágenes, otro para las etiquetas
data_imgs = np.load('./archivos/mnistc_images.npy')
data_chrs = np.load('./archivos/mnistc_labels.npy')[:,np.newaxis]


#%% Función para guardar un archivo que contenga imágenes de un único número

indices = []
lista_data = []
lista_chrs = []

#Busco los índices que corresponden las imágenes de un dígito en particular 
def separar_numero(x):
    global indices
    indices = [i for i, numero in enumerate(data_chrs) if numero == x]

#Selecciono y guardo las imágenes correspondientes a esos índices 
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

#%% Función para comparar imágenes usando ssim

def comparar_ssim (x,y): #x : número fijo, y: número a comparar
    ssim_valores = []
    for i in range(0,min(len(y), len(x))-1): #así comparamos con el mismo número de imágenes
        image_1 = x[i,:,:,0]
        image_2 = y[i+1,:,:,0]
        valores = ssim(image_1,image_2)
        ssim_valores.append(valores)
    return np.mean(ssim_valores)

#%% Función para comparar un número con los demás 
def obtener_valores_comparacion(x):    
    valores_comparacion = []
    for i in range(0,10):
        y = globals().get(f'imgs{i}')
        val = comparar_ssim(x, y)
        valores_comparacion.append(val)
    return valores_comparacion
#%% Cargo las imágenes a comparar

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

#%% Comparamos todos los números entre sí y creamos un dataframe
resultados_totales = {}
numeros = np.arange(0,10,1 )

for i in range(0, 10):
    x = globals().get(f'imgs{i}')
    if x is not None:
        resultados_totales[f'imgs{i}'] = obtener_valores_comparacion(x)

df_resultados = pd.DataFrame(resultados_totales, index=numeros,
                             columns=numeros)
#%%
resultados_totales = {}
numeros = np.arange(0,10,1) 

for i in range(0, 10):
    x = globals().get(f'imgs{i}')
    resultados_totales[f'imgs{i}'] = obtener_valores_comparacion(x)

df_resultados = pd.DataFrame(resultados_totales)

df_resultados.columns = numeros


#%%

sns.heatmap(df_resultados, annot=False, cmap="YlGnBu", cbar=True)
plt.title('Valor promedio de SSIM entre dígitos')
plt.xlabel('Dígitos')
plt.ylabel('Dígitos')
plt.tight_layout()
plt.show()

plt.savefig('similitud entre digitos.pdf')


