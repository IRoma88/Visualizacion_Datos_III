# -*- coding: utf-8 -*-
"""

#Introducción

Instagram es una de las aplicaciones de redes sociales más populares hoy en día. Las personas que utilizan Instagram profesionalmente lo hacen para promocionar su negocio, crear un portafolio, bloguear y crear varios tipos de contenido. Como Instagram es una aplicación popular utilizada por millones de personas con diferentes nichos, Instagram sigue cambiando para mejorar para los creadores de contenido y los usuarios. Pero como esto sigue cambiando, afecta al alcance de nuestras publicaciones que nos afecta a largo plazo. Así que si un creador de contenido quiere hacerlo bien en Instagram a largo plazo, tiene que mirar los datos de su alcance en Instagram. Ahí es donde entra el uso de la Ciencia de Datos en las redes sociales.

Si quieres analizar el alcance de tu cuenta de Instagram, tienes que recopilar tus datos manualmente ya que hay algunas APIs, pero no funcionan bien. Así que es mejor recopilar tus datos de Instagram manualmente.

Puedes descargar el conjunto de datos que utilizaremos para el proyecto de análisis de alcance de Instagram desde aquí:

https://raw.githubusercontent.com/amankharwal/Website-data/master/Instagram.csv


Los datos pertenecen a la cuenta the.clever.programmer, que cuenta con 17.9k seguidores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.read_csv("instagram.csv", encoding = 'latin1')
data = pd.read_csv("instagram.csv", encoding = 'latin1') # Intenta sin encoding primero
print(data.head())

"""
Si te da un error como:

UnicodeDecodeError: 'utf-8' codec can't decode byte...

Entonces sí debes añadir encoding='latin1' u otro que funcione (como 'ISO-8859-1' o 'cp1252').
"""

"""Lo primero que debes hacer es buscar cuantos valores nulos hay en cada columna."""

data.isnull().sum()

"""Vamos a comprobar cuantas filas contiene el dataset para ver si podríamos eliminar los valores nulos directamente o perderíamos mucha información."""

data.shape[0]

"""Dado que tiene 100 filas podemos permitirnos el lujo de eliminar los datos nulos."""

data = data.dropna()

"""Por último comprueba que tipo de datos contiene cada columna."""

data.info()

data.head()

"""#Pregunta 1

Una vez realizado el análisis preliminar de los datos vamos a realizar diferentes gráficas para responder preguntas como:

¿Las impresiones desde la pestaña Inicio (Home/TimeLine) siguen alguna distribución?

Realiza un histograma y/o un gráfico de densidad (recuerda que debes importar seaborn) de los datos de la columna 'From Home' y almacena en la variable solución_1 la letra correspondiente a la distribución que más se asemeja:

a) Exponencial

b) Normal

c) Geométrica

d) Uniforme


"""

import seaborn as sns

plt.hist(data['From Home'])
plt.show()

sns.kdeplot(data['From Home'])
plt.show()

Home = sns.kdeplot(data['From Home'])
Impressions = sns.kdeplot(data['Impressions'])
sns.kdeplot(data['From Home'])
sns.kdeplot(data['Impressions'])
plt.show()

sns.histplot(data['From Home'])
plt.show()

solucion_1='b'

"""#Pregunta 2

Vamos a realizar el mismo análisis pero con las impresiones recibidas a traves de los hashtags, que son herramientas que utilizamos para categorizar nuestras publicaciones en Instagram y así poder llegar a más personas en función del tipo de contenido que estamos creando.

Realiza un histograma y/o un gráfico de densidad (recuerda que debes importar seaborn) de la columna 'From Hashtags' y almacena en la variable solución_2 la letra correspondiente a la distribución que más se asemeja:

a) Exponencial

b) Normal

c) Geométrica

d) Uniforme
"""

sns.histplot(data['From Hashtags'])
plt.show()

solucion_2='c'

"""#Pregunta 3

Analiza también mediante un histograma y/o un gráfico de densidad la distribucción de las impresiones que vienen de la sección explorar, es decir, del sistema de recomendación de Instagram.


"""

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(8, 5))

# KDE plot (densidad)
sns.kdeplot(data['From Explore'], ax=ax, color='red',label='From Explore')
sns.kdeplot(data['From Home'], ax=ax, color='blue',label='From Home')
sns.kdeplot(data['From Hashtags'], ax=ax, color='green',label='From Hashtags')

# Configuración del gráfico
ax.set_title('Histograma')
ax.set_ylabel('Densidad')
ax.set_xlabel('Impresiones')
ax.legend()

plt.show()

"""¿A qué distribución se parece más a la de las impresiones recibidas desde el inicio o las recibidas por hashtag? Responde en la variable solucion_3 con 'home' o 'hashtags' según consideres que es más parecida."""

solucion_3='hashtags'

"""#Pregunta 4

Una vez analizado cada fuente de impresiones por separado realiza un gráfico de tarta para comprobar las cantidades totales que vienen de cada fuente. Incluye el porcentaje dentro del gráfico con formato '%.2f%%' y los nombres de las columnas como etiquetas.
"""

totales = data.select_dtypes(include='number').sum()
totales

plt.pie(totales, labels=totales.index, autopct='%.2f%%')
plt.show()

"""Almacena True (booleano) en la variable solucion_4 si la suma de impresiones totales recibidas a través de Hashtags y la sección explorar es mayor que las impresiones totales recibidas desde la sección de inicio, sino almacena False."""

solucion_4=False

"""#Pregunta 5

Vamos a analizar algunas relaciones entre variables, por ejemplo:

¿Están relacionadas las columnas 'Impresiones' y 'Likes'? Crea un gráfico de dispersión cuyo eje x sean las impresiones, el eje y los likes y los puntos tengan distinto tamaño en función de los likes.
"""

sns.scatterplot(data=data, x='Impressions', y='Likes', size='Likes')
plt.show()

"""Realiza el mismo tipo de gráfico para comprobar si existe relación entre las impresiones y el número de comentarios."""

sns.scatterplot(data=data, x='Impressions', y='Comments', size='Comments')
plt.show()

"""Almacena en la variable solución_5 la letra de la frase correcta:

a) Tanto los likes como los comentarios están relacionados con las impresiones.

b) Los likes están relacionados con las impresiones pero los comentarios no.

c) Los comentarios están relacionados con las impresiones pero los likes no.

d) Ni los likes ni los comentarios están relacionados con las impresiones.
"""

solucion_5='b'

"""#Pregunta 6

En Matemáticas para Data Science vimos como calcular la correlación entre las columnas de una matriz usando:

    np.corrcoef('nombre_matriz'))

Podemos comprobar la correlación de un dataframe entero de forma similar utilizando:

    nombre_dataframe.corr()


Utiliza el valor de correlación entre columnas para generar un mapa de calor con estilo de color 'hot' y plt.colorbar() para mostrar la leyenda del gráfico y plt.yticks para añadir etiquetas al eje y.
"""

total_corregido = data.select_dtypes(include=np.number)

total_corregido.corr()

plt.figure(figsize=(10, 8))
plt.imshow(total_corregido.corr(), cmap='hot')
plt.colorbar()
plt.yticks(range(len(total_corregido.columns)), total_corregido.columns)
plt.xticks(range(len(total_corregido.columns)), total_corregido.columns, rotation=90)
plt.show()

"""¿A que relación corresponden los cuadrados de color negro? Almacena la letra de la respuesta correcta en la variable pregunta 6

a) Impressions-From Hashtags

b) Follows-Profile Visits

c) Shares-Likes

d) Comments-From Others
"""

solucion_6='d'

"""¿Cuál es la acción que más relacionada está con los follows? Escribe el nombre de la columna que corresponda en la variable solucion_7."""

solucion_7='Profile Visits'

#@title Ejecuta para obtener el token
import hashlib

correct = str(solucion_1)+ str(solucion_2)+ str(solucion_3)+ str(solucion_4)+ str(solucion_5)+ str(solucion_6)+ str(solucion_7)
pwd = hashlib.sha256(str(correct).encode())
#print('El token para corregir en Nodd3r es:\n',pwd.hexdigest())
if pwd.hexdigest()[0:6] == '4ad819':
  print('¡Felicidades! puedes avanzar al siguiente modulo \n El token es: ',pwd.hexdigest())
else:
  print('Hay algún error en el código o tu forma es diferente a la planteada, pregunta por el foro si no lo ves claro.')

"""#Bonus: Introducción a Machine Learning

Como adelanto de lo que aprenderéis en el próximo curso vamos a predecir el alcance que tendría un post en instagram en funcion del número de Likes, veces que ha sido guardado, cantidad de comentarios, veces compartido,visitas al perfil y seguidores.
"""

#Importamos de sklearn lo que necesitaremos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

#Seleccionamos los datos que utilizaremos para predecir (x) y los datos que queremos medir (y)
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares',
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
#los dividimos en dos partes, la primera para entrenar el algoritmo y la segunda para testearlo.
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2,
                                                random_state=42)

#Utilizamos un modelo de regresión para entrenar el algoritmo y comprobar los resultados con los datos de test
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

#Realizamos la predicción en funcion de los parámetros seleccionados (features=[['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)
