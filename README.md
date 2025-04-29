# 📊 Análisis de Alcance en Instagram con Python

Este proyecto aplica técnicas de análisis de datos y machine learning para estudiar el alcance de publicaciones de Instagram a partir de datos reales de la cuenta [the.clever.programmer](https://www.instagram.com/the.clever.programmer/), con más de 17.9k seguidores.

## 🔍 Objetivo

Analizar la fuente de impresiones (Inicio, Hashtags, Explorar) y su relación con métricas como likes, comentarios, compartidos y seguidores. Al final del proyecto, se construye un modelo de regresión para predecir el alcance futuro de una publicación en función de su rendimiento.

---

## 🗂️ Contenido

- Análisis exploratorio de datos
- Visualización de distribuciones (histogramas, KDE)
- Análisis correlacional y gráficos de dispersión
- Mapa de calor de correlaciones
- Gráficos de pastel para analizar contribuciones
- Modelo de regresión (`PassiveAggressiveRegressor`) para predecir impresiones

---

## 📁 Dataset
 - Está subido en formato .csv con el nombre de **Instagram (1).csv**

## ▶️ Ejecución

1. Clona este repositorio o copia el código en Google Colab.
2. Asegúrate de tener el archivo `instagram.csv` en el mismo directorio o cárgalo a Colab.
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ````
4. Ejecuta el notebook línea por línea o carga todo el script.

## 🧠 Machine Learning
El modelo de regresión toma como variables independientes:

  - Likes

  - Saves

  - Comments

  - Shares

  - Profile Visits

  - Follows

Y predice el número de **Impresiones** esperadas para una publicación.

## 🧩 Dependencias
Instalación vía requirements.txt:

  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

## 📎 Archivos ignorados (.gitignore)
Este repositorio ignora automáticamente:

  - Archivos temporales de Python y Jupyter

  - Archivos del sistema (Windows/macOS)

  - Checkpoints y backups

  - Archivos de entorno virtual

✅ Autor
Curso de Ciencia de Datos con Python — Análisis de Alcance en Redes Sociales
Repositorio basado en los datos de Aman Kharwal

