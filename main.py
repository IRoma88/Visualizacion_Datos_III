import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Cargar el dataset
data = pd.read_csv("Instagram (1).csv", encoding='latin1')

# Análisis preliminar
print(data.head())
print(data.isnull().sum())
print(f'Número de filas: {data.shape[0]}')

# Eliminar valores nulos
data = data.dropna()

# Información del dataset
data.info()

# Pregunta 1 - Distribución de impresiones desde Home
sns.histplot(data['From Home'])
plt.show()

sns.kdeplot(data['From Home'])
plt.show()

solucion_1 = 'b'

# Pregunta 2 - Distribución de impresiones desde Hashtags
sns.histplot(data['From Hashtags'])
plt.show()

solucion_2 = 'c'

# Pregunta 3 - Comparar distribuciones de Home, Hashtags, y Explore
sns.kdeplot(data['From Explore'], color='red', label='From Explore')
sns.kdeplot(data['From Home'], color='blue', label='From Home')
sns.kdeplot(data['From Hashtags'], color='green', label='From Hashtags')

plt.title('Histograma de Impresiones')
plt.ylabel('Densidad')
plt.xlabel('Impresiones')
plt.legend()
plt.show()

solucion_3 = 'hashtags'

# Pregunta 4 - Gráfico de tarta de las impresiones por fuente
totales = data.select_dtypes(include='number').sum()
plt.pie(totales, labels=totales.index, autopct='%.2f%%')
plt.show()

solucion_4 = False

# Pregunta 5 - Relación entre impresiones y likes
sns.scatterplot(data=data, x='Impressions', y='Likes', size='Likes')
plt.show()

# Relación entre impresiones y comentarios
sns.scatterplot(data=data, x='Impressions', y='Comments', size='Comments')
plt.show()

solucion_5 = 'b'

# Pregunta 6 - Correlación entre columnas
total_corregido = data.select_dtypes(include=np.number)
corr_matrix = total_corregido.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='hot')
plt.colorbar()
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.show()

solucion_6 = 'd'

# Pregunta 7 - Columna más relacionada con los follows
solucion_7 = 'Profile Visits'

# Bonus: Modelo de Machine Learning para predecir impresiones
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
print(f'Modelo R^2 en test: {model.score(xtest, ytest)}')

features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
print(f'Predicción de impresiones: {model.predict(features)}')
