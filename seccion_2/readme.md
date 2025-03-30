# README: Ejemplos de Redes Neuronales con Keras y PyTorch

Este README proporciona una explicación detallada de los notebooks de ejemplo de redes neuronales implementados con Keras y PyTorch que se encuentran en la carpeta `seccion_2`. Estos notebooks forman parte del módulo "Modelos y Métodos de Sistemas Inteligentes" de la Maestría en Análisis de Datos y Sistemas Inteligentes.

## Contenido

* [Introducción](#introducción)
* [Requisitos previos](#requisitos-previos)
* [Notebook de Keras](#notebook-de-keras)
    * [Clasificación con Iris (Keras)](#clasificación-con-iris-keras)
    * [Regresión con California Housing (Keras)](#regresión-con-california-housing-keras)
* [Notebook de PyTorch](#notebook-de-pytorch)
    * [Clasificación con Iris (PyTorch)](#clasificación-con-iris-pytorch)
    * [Regresión con California Housing (PyTorch)](#regresión-con-california-housing-pytorch)
* [Comparación entre Keras y PyTorch](#comparación-entre-keras-y-pytorch)
* [Referencias](#referencias)

## Introducción

Los notebooks en esta sección están diseñados para proporcionar ejemplos prácticos de implementación de redes neuronales para tareas de clasificación y regresión utilizando dos de los frameworks más populares: Keras (con TensorFlow como backend) y PyTorch. Cada notebook demuestra cómo:

* Cargar y preprocesar datos
* Crear y entrenar modelos de redes neuronales
* Evaluar el rendimiento de los modelos
* Visualizar los resultados

Los ejemplos utilizan conjuntos de datos estándar:

* **Iris Dataset:** Un conjunto de datos clásico para clasificación multiclase.
* **California Housing Dataset:** Un conjunto de datos para regresión sobre precios de viviendas.

## Requisitos previos

Para ejecutar los notebooks, necesitarás:

* Python 3.6+
* Jupyter Notebook o JupyterLab
* Bibliotecas:
    * NumPy
    * Pandas
    * Matplotlib
    * Scikit-learn
    * TensorFlow 2.x (para el notebook de Keras)
    * PyTorch (para el notebook de PyTorch)

Puedes instalar las dependencias necesarias con:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch torchvision
```

## Notebook de Keras

### Estructura General

El notebook de Keras (`Example_with_keras_for_classification_and_regression.ipynb`) está organizado en dos partes principales:

* Clasificación con el dataset Iris
* Regresión con el dataset de viviendas de California

Cada parte incluye las siguientes secciones:

* Carga y visualización de datos
* Limpieza de datos
* Selección de características
* División en conjuntos de entrenamiento y prueba
* Definición del modelo
* Entrenamiento del modelo
* Evaluación del modelo

### Clasificación con Iris (Keras)

#### Carga y Visualización de Datos

```python
# Importar bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Cargar el dataset Iris
iris = load_iris()
data = iris.data
target = iris.target # Índices numéricos (0, 1, 2)
feature_names = iris.feature_names
target_names = iris.target_names # Nombres ('setosa', 'versicolor', 'virginica')

# Convertir a DataFrame para facilitar manipulación inicial
df = pd.DataFrame(data, columns=feature_names)
# Es útil mantener el índice numérico para Keras (si se usa sparse loss)
# y el nombre para referencia/visualización.
df['target_idx'] = target
df['target_name'] = df['target_idx'].apply(lambda x: target_names[x])

# (Aquí normalmente se añadiría código de visualización, p.ej.)
# print(df.head())
# pd.plotting.scatter_matrix(df[feature_names], c=df['target_idx'], figsize=(10, 10))
# plt.show()
```

Esta celda carga el dataset Iris utilizando Scikit-learn y lo convierte a un DataFrame de pandas para facilitar su manipulación. Se asume que también se visualizaría la distribución de características (por ejemplo, usando una matriz de dispersión como se sugiere en el código comentado) para entender mejor las relaciones entre variables y la separabilidad de las clases.

#### Limpieza y Preparación de Datos

```python
# Asumiendo que 'stats' de scipy ya fue importado (from scipy import stats)
# y que 'df' es el DataFrame del paso anterior con las columnas 'feature_names'.

# Eliminar outliers basados en Z-score (este paso puede ser opcional para Iris,
# pero es útil como ejemplo general).
z_scores = np.abs(stats.zscore(df[feature_names]))

# Crear un nuevo DataFrame manteniendo solo las filas sin outliers
# (donde el Z-score absoluto es < 3 para todas las características).
# Usar .copy() para evitar advertencias de SettingWithCopyWarning.
df_cleaned = df[(z_scores < 3).all(axis=1)].copy()

print(f"Filas originales: {df.shape[0]}")
print(f"Filas después de quitar outliers: {df_cleaned.shape[0]}")

# Verificar si hay valores vacíos en el DataFrame limpio
print("\nValores vacíos después de limpieza:")
print(df_cleaned.isnull().sum())
# Si hubiera valores vacíos, aquí se aplicarían estrategias como
# df_cleaned.dropna(inplace=True) o imputación.
```

Esta celda demuestra cómo eliminar valores atípicos (_outliers_) utilizando el método del `Z-score`. Se consideran outliers aquellos puntos cuyos valores en alguna característica se desvían más de 3 desviaciones estándar de la media. También se verifica si existen valores faltantes (`NaN`) en el conjunto de datos resultante (`df_cleaned`). Este paso es **crucial** para asegurar la calidad de los datos que se usarán para entrenar el modelo, aunque para el dataset **Iris estándar**, podría no ser estrictamente necesario.

#### Selección de Características

```python
# Asumiendo que 'SelectKBest' y 'f_classif' han sido importados de sklearn.feature_selection
# Asumiendo que 'np' es numpy y 'df_cleaned', 'feature_names', 'target_idx' existen del paso anterior.

# Usar los datos limpios del paso anterior
X = df_cleaned[feature_names].values
y = df_cleaned['target_idx'].values # Usar el índice numérico limpio

# Seleccionar las k=2 características más relevantes usando ANOVA F-test (f_classif)
selector = SelectKBest(score_func=f_classif, k=2)

# Ajustar el selector a los datos y transformar X para obtener solo las características seleccionadas
X_new = selector.fit_transform(X, y)

# Obtener los nombres de las características seleccionadas (opcional, para información)
selected_features_indices = selector.get_support(indices=True)
selected_features = np.array(feature_names)[selected_features_indices]

print(f"Características seleccionadas: {selected_features}")
print(f"Forma de los datos antes de selección: {X.shape}")
print(f"Forma de los datos después de selección (X_new): {X_new.shape}")
```

Esta celda selecciona las características más relevantes para el modelo utilizando el método `SelectKBest` con la función de puntuación `f_classif` (adecuada para clasificación). Se configura para conservar las `k=2` características más importantes según esta prueba estadística. El resultado, `X_new`, contiene únicamente los datos correspondientes a estas dos características seleccionadas y será el que se use en los siguientes pasos.








