#%%===========================================================================
# Grupo 14: Nahuel Rabey y Sammy Vallejo 
#Este código sirve para el análisis exploratorio del dataset, se utiliza el método ssim para determinar 
#similitud tanto entre las imágenes que corresponden a un único dígito como con las imágenes que corresponden  
#a los distintos dígitos. 
#=============================================================================

#%%
import helpers as hlps
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

#%%

#%%
imagenes = hlps.Imagenes()


img5 = imagenes.obtener_imagenes(5)

#%%
imgs_list = []
for i in range(10):  
    imgs = imagenes.obtener_imagenes(i)
    imgs_list.append(imgs)
    
#%% Función para comparar imágenes usando ssim

def comparar_ssim (x,y): #x : número fijo, y: número a comparar
    ssim_valores = []
    maximo = min(len(y), len(x)) - 1 
    for i in range(maximo): #así comparamos con el mismo número de imágenes
        image_1 = x[i,:,:,0]
        image_2 = y[i+1,:,:,0] #así se comparan distintas imágenes correspondientes al mismo dígito
        valores = ssim(image_1,image_2)
        ssim_valores.append(valores)
    return np.mean(ssim_valores)


#%% Función para comparar un número con los demás 
def obtener_valores_comparacion(x):    
    valores_comparacion = []
    for i in range(10):
        y = imgs_list[i]
        val = comparar_ssim(x, y)
        valores_comparacion.append(val)
    return valores_comparacion

#%% Comparamos todos los números entre sí y creamos un dataframe
resultados_totales = {}
numeros = np.arange(0,10,1)

for i in range(10):
    x = imgs_list[i]
    resultados_totales[i] = obtener_valores_comparacion(x)

df_resultados = pd.DataFrame(resultados_totales, index=numeros)
#%%
sns.heatmap(df_resultados, annot=False, cmap="YlGnBu", cbar=True)
plt.title('Valor promedio de SSIM entre dígitos')
plt.xlabel('Dígitos')
plt.ylabel('Dígitos')
plt.tight_layout()
plt.show()

plt.savefig('similitud entre digitos.pdf')

#%%
promedios_por_digito = []

# Obtener imágenes y calcular el promedio para cada dígito
for i in range(10):  
    promedio_digito = np.mean(imgs_list[i], axis=0)  # Calcular el promedio en el eje de la cantidad de imágenes
    promedios_por_digito.append(promedio_digito)  # Guardar el promedio


# Visualizar los promedios en una cuadrícula
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Imagen promedio de cada dígito")

for i, promedio in enumerate(promedios_por_digito):
    ax = axes[i // 5, i % 5]
    ax.imshow(promedio, cmap='gray')  # Mostrar el promedio como imagen
    ax.set_title(f"Dígito {i}")
    ax.axis('off')

plt.show()

#%%
variances_por_digito = []
atributos_significativos = []
umbral_varianza = 10  # Ajusta este valor según el nivel de significancia deseado

# Calcular el promedio y la varianza para cada dígito
for i in range(10):
    # Calcular promedio y varianza en el eje de la cantidad de imágenes
    varianza_digito = np.var(imgs_list[i], axis=0)
    
    # Crear la máscara de atributos significativos basada en el umbral
    mask_significativa = varianza_digito > umbral_varianza
    
    # Guardar el promedio y la máscara
    variances_por_digito.append(varianza_digito)
    atributos_significativos.append(mask_significativa)

# Visualizar las imágenes promedio filtradas
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Imagen promedio de cada dígito (filtrando atributos significativos)")

for i, promedio in enumerate(promedios_por_digito):
    # Aplicar la máscara de atributos significativos al promedio del dígito
    promedio_filtrado = np.where(atributos_significativos[i], promedio, np.nan)  # Coloca NaN en píxeles no significativos
    
    ax = axes[i // 5, i % 5]
    im = ax.imshow(promedio_filtrado, cmap='gray')  # Mostrar el promedio filtrado como imagen
    ax.set_title(f"Dígito {i}")
    ax.axis('off')

plt.show()
