# Sección de Ejemplos: Preprocesamiento de Datos y Modelos de Redes Neuronales

En este repositorio, encontrarás ejemplos prácticos sobre diversas técnicas de preprocesamiento de datos para imágenes, texto y series temporales, así como implementaciones de Redes Neuronales Convolucionales (CNN) para clasificación de imágenes y Redes Neuronales Recurrentes (RNN) para previsión de series temporales. Se utilizan bibliotecas populares como OpenCV, Pillow, NLTK, Pandas, Keras (TensorFlow) y PyTorch. Estos notebooks y scripts están diseñados para facilitar la comprensión del código y su ejecución.

---

## Sección 1: Preprocesamiento de Datos

Esta sección contiene ejemplos prácticos de técnicas de preprocesamiento para diferentes tipos de datos. Los scripts están diseñados para ilustrar los pasos comunes y fundamentales en la preparación de datos para análisis y modelado.

### Contenido

1.  [Ejemplo_Preprocesamiento_Imagenes.py](Ejemplo_Preprocesamiento_Imagenes.py)
2.  [Ejemplo_Preprocesamiento_Texto_NLP.py](Ejemplo_Preprocesamiento_Texto_NLP.py)
3.  [Ejemplo_Preprocesamiento_Series_Temporales.py](Ejemplo_Preprocesamiento_Series_Temporales.py)

---

### Ejemplo 1: Preprocesamiento de Imágenes

#### Descripción

Este script demuestra varias técnicas comunes de preprocesamiento de imágenes utilizando bibliotecas como OpenCV y Pillow. Se aplica cada técnica a una imagen de ejemplo (cargada localmente o desde el dataset CIFAR-10) y se visualizan los resultados para facilitar la comprensión de su efecto.

#### Técnicas Demostradas

* **Carga y Conversión de Espacios de Color**: Incluye la carga de imágenes y su conversión a diferentes espacios de color como escala de grises y HSV.
* **Redimensionamiento y Recorte**: Métodos para ajustar las dimensiones de las imágenes y seleccionar regiones de interés.
* **Normalización de Píxeles y Ecualización de Histograma**: Técnicas para estandarizar el rango de intensidad de los píxeles y mejorar el contraste de la imagen.
* **Aplicación de Filtros**: Uso de filtros para suavizado, reducción de ruido y detección de bordes, como el Desenfoque Gaussiano, Desenfoque de Mediana, Filtro Bilateral y Filtro Laplaciano.
* **Aumentación de Datos Básica**: Técnicas simples para incrementar artificialmente la diversidad del conjunto de datos, como volteo, rotación y adición de ruido.

---

### Ejemplo 2: Preprocesamiento de Texto para PNL

#### Descripción

Este script ilustra pasos fundamentales de preprocesamiento de texto específicos para tareas de Procesamiento del Lenguaje Natural (PNL). Se utilizan principalmente las bibliotecas NLTK y TextBlob, aplicando las técnicas a cadenas de texto de ejemplo para mostrar su funcionamiento.

#### Técnicas Demostradas

* **Conversión a Minúsculas**: Estandarización del texto a minúsculas.
* **Eliminación de Elementos Irrelevantes**: Limpieza de etiquetas HTML, URLs y signos de puntuación.
* **Tratamiento de Palabras de Chat**: Conversión de jerga o abreviaturas comunes en chats a sus formas completas.
* **Corrección Ortográfica**: Uso de TextBlob para corregir errores de ortografía.
* **Eliminación de Palabras Vacías (Stop Words)**: Filtrado de palabras comunes con poco valor semántico.
* **Manejo de Emojis**: Técnica para eliminar emojis del texto.
* **Tokenización**: División del texto en unidades más pequeñas (palabras y frases).
* **Derivación (Stemming)**: Reducción de palabras a su forma raíz.
* **Lematización (Lemmatization)**: Reducción de palabras a su forma base significativa (lemma).

---

### Ejemplo 3: Preprocesamiento de Series Temporales

#### Descripción

Este script se enfoca en las técnicas de preprocesamiento específicas para datos de series temporales, utilizando Pandas para la manipulación de datos y Matplotlib para la visualización. Se emplea el dataset "Air Passengers" (o un conjunto de datos sintético como alternativa) para demostrar los métodos.

#### Técnicas Demostradas

* **Estructuración de Datos de Series Temporales**: Conversión de campos de fecha al formato datetime, ordenación cronológica de los datos y establecimiento de la fecha como índice del DataFrame.
* **Imputación de Valores Faltantes**: Relleno de datos ausentes utilizando métodos de interpolación apropiados para series temporales (lineal, spline, basada en tiempo), demostrado sobre valores faltantes introducidos artificialmente.
* **Eliminación de Ruido (Suavizado)**: Aplicación de medias móviles (rolling means) con diferentes tamaños de ventana para suavizar la serie y reducir fluctuaciones aleatorias.
* **Detección de Valores Atípicos (Outliers)**: Identificación de observaciones anómalas utilizando bandas estadísticas móviles (media móvil ± k * desviación estándar móvil), demostrado sobre outliers introducidos artificialmente.

---

## Sección 2: Redes Neuronales Convolucionales (CNN) y Recurrentes (RNN)

En esta sección, encontrarás ejemplos prácticos sobre cómo implementar Redes Neuronales Convolucionales (CNN) para clasificación de imágenes y Redes Neuronales Recurrentes (RNN) para previsión de series temporales. Se utilizan las bibliotecas Keras (TensorFlow) y PyTorch. Estos notebooks están diseñados para facilitar la comprensión del código y su ejecución.

### Contenido

1.  [Ejemplo_con_Keras_CNN_para_clasificacion.ipynb](Ejemplo_con_Keras_CNN_para_clasificacion.ipynb)
2.  [Ejemplo_con_pytorch_CNN_para_clasificacion.ipynb](Ejemplo_con_pytorch_CNN_para_clasificacion.ipynb)
3.  [Ejemplo_con_Keras_RNN_para_prevision.ipynb](Ejemplo_con_Keras_RNN_para_prevision.ipynb)
4.  [Ejemplo_con_pytorch_RNN_para_prevision.ipynb](Ejemplo_con_pytorch_RNN_para_prevision.ipynb)

---

### Ejemplo 1: Clasificación de Imágenes con CNN usando Keras

#### Descripción

En este notebook, se utiliza la biblioteca Keras (sobre TensorFlow) para construir y entrenar una Red Neuronal Convolucional (CNN) para un problema de clasificación de imágenes. El ejemplo se centra en el dataset "Piedra, Papel o Tijeras". Incluye pasos desde la carga de datos, preprocesamiento (incluyendo aumento de datos), definición del modelo, entrenamiento y evaluación.

#### Pasos Clave del Notebook

1.  **Importar las librerías necesarias**:
    ```python
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
    import numpy as np
    ```
    - `tensorflow.keras.models.Sequential`: Permite construir un modelo de red neuronal secuencial.
    - `tensorflow.keras.layers.Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`: Capas comunes en una CNN.
    - `ImageDataGenerator`: Para el preprocesamiento y aumento de datos en tiempo real.

2.  **Cargar y Preprocesar el Dataset**:
    - Carga del dataset "rock_paper_scissors" desde `tensorflow_datasets`.
    - Normalización de las imágenes.
    - Creación de generadores de datos para entrenamiento y validación, aplicando aumento de datos al conjunto de entrenamiento.

3.  **Definir la Arquitectura del Modelo CNN**:
    ```python
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax') # 3 clases: piedra, papel, tijera
    ])
    ```
    - Se define una CNN con múltiples capas convolucionales, de pooling, una capa densa y una capa de dropout para regularización.

4.  **Compilar y Entrenar el Modelo**:
    ```python
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator
    )
    ```
    - Se compila el modelo usando el optimizador `adam` y la función de pérdida `categorical_crossentropy`.
    - Se entrena el modelo utilizando los generadores de datos.

5.  **Evaluar el Modelo y Realizar Predicciones**:
    - Se visualizan las curvas de aprendizaje (precisión y pérdida).
    - Se evalúa el modelo en el conjunto de prueba.
    - Se realizan predicciones sobre nuevas imágenes.

---

### Ejemplo 2: Clasificación de Imágenes con CNN usando PyTorch

#### Descripción

Este notebook adapta el problema de clasificación de imágenes (dataset "Piedra, Papel o Tijeras") utilizando la biblioteca PyTorch. Se muestra cómo definir un `Dataset` personalizado, construir una CNN como una clase `nn.Module`, e implementar los bucles de entrenamiento y evaluación.

#### Pasos Clave del Notebook

1.  **Importar las librerías necesarias**:
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, datasets
    import tensorflow_datasets as tfds # Para cargar el dataset
    import numpy as np
    import matplotlib.pyplot as plt
    ```
    - `torch.nn.Module`: Clase base para todos los modelos de redes neuronales en PyTorch.
    - `torch.optim`: Contiene varios algoritmos de optimización.
    - `DataLoader`, `Dataset`: Utilidades para la carga y manejo de datos.
    - `torchvision.transforms`: Para preprocesamiento de imágenes y aumento de datos.

2.  **Cargar y Preprocesar el Dataset**:
    - Carga del dataset "rock_paper_scissors" vía TFDS y conversión a arrays NumPy.
    - Creación de una clase `RockPaperScissorsDataset` personalizada que hereda de `torch.utils.data.Dataset`.
    - Definición de transformaciones de datos (`transforms.Compose`) para preprocesamiento y aumento.
    - Creación de `DataLoader` para los conjuntos de entrenamiento y validación.

3.  **Definir la Arquitectura del Modelo CNN**:
    ```python
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=3):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # ... más capas convolucionales y de pooling ...
            # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            # El tamaño de entrada a la capa fc dependerá de la salida de las capas anteriores
            # Ejemplo: para (150,150,3) -> (N, 128, 18, 18) después de 3 pooling layers (150/2/2/2 = 18.75 ~ 18)
            # fc_in_features = 128 * 18 * 18 # Esto debe calcularse correctamente
            self.fc1 = nn.Linear(CONFIG_MODELO_SIMPLE['fc_in_features'], 512) # Usar un valor precalculado o dinámico
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            # ... más pasadas por capas conv/relu/pool ...
            # x = self.pool(self.relu(self.conv2(x)))
            # x = self.pool(self.relu(self.conv3(x)))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    ```
    - La CNN se define como una clase que hereda de `nn.Module`, con capas definidas en `__init__` y la lógica de propagación hacia adelante en el método `forward`. (Nota: el código de la arquitectura es un esqueleto y necesitaría completarse/ajustarse).

4.  **Entrenar el Modelo**:
    ```python
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # ... Lógica de validación ...
    ```
    - Se define la función de pérdida (`CrossEntropyLoss`) y el optimizador (`Adam`).
    - El entrenamiento se realiza en un bucle explícito, gestionando la propagación hacia atrás y la actualización de los pesos.

5.  **Evaluar el Modelo y Realizar Predicciones**:
    - Se evalúa el modelo en el conjunto de validación/prueba.
    - Se realizan inferencias con el modelo entrenado.

---

### Ejemplo 3: Previsión de Series Temporales con RNN (LSTM/GRU) usando Keras

#### Descripción

En este notebook, se implementan Redes Neuronales Recurrentes (específicamente LSTM y GRU) utilizando Keras para la previsión de series temporales. El ejemplo utiliza el dataset "Air Passengers" para predecir el número futuro de pasajeros.

#### Pasos Clave del Notebook

1.  **Importar las librerías necesarias**:
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense
    ```
    - `LSTM`, `GRU`: Capas recurrentes proporcionadas por Keras.
    - `MinMaxScaler`: Para escalar los datos de la serie temporal.

2.  **Cargar y Preprocesar el Dataset**:
    - Carga del dataset "Air Passengers".
    - Escalado de los datos usando `MinMaxScaler`.
    - Creación de secuencias de entrada (ventana de tiempo) y salida (valor a predecir).
    - División en conjuntos de entrenamiento y prueba.

3.  **Definir la Arquitectura del Modelo RNN**:
    ```python
    # Ejemplo con LSTM
    model_lstm = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(units=50),
        Dense(units=1)
    ])
    # Se define un modelo similar para GRU
    ```
    - Se construyen modelos secuenciales con capas LSTM o GRU, seguidas de una capa densa para la predicción.

4.  **Compilar y Entrenar el Modelo**:
    ```python
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    history_lstm = model_lstm.fit(X_train, y_train, epochs=100, batch_size=CONFIG_TRAIN['batch_size'], verbose=1)
    ```
    - Se compilan los modelos utilizando el optimizador `adam` y la función de pérdida `mean_squared_error`, adecuada para regresión.
    - Se entrenan los modelos con los datos de secuencia.

5.  **Evaluar el Modelo y Realizar Predicciones**:
    - Se realizan predicciones sobre el conjunto de prueba.
    - Se desescalan las predicciones para compararlas con los valores originales.
    - Se visualizan las predicciones junto a los valores reales.

---

### Ejemplo 4: Previsión de Series Temporales con RNN (LSTM/GRU) usando PyTorch

#### Descripción

Este notebook aborda la tarea de previsión de series temporales (dataset "Air Passengers") utilizando modelos LSTM y GRU implementados en PyTorch. Cubre la creación de `Dataset` y `DataLoader` para series temporales, la definición de modelos RNN como clases `nn.Module`, y los bucles de entrenamiento y evaluación.

#### Pasos Clave del Notebook

1.  **Importar las librerías necesarias**:
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    ```
    - `nn.LSTM`, `nn.GRU`: Módulos de capas recurrentes en PyTorch.

2.  **Cargar y Preprocesar el Dataset**:
    - Carga del dataset "Air Passengers".
    - Escalado de los datos.
    - Creación de una clase `TimeSeriesDataset` personalizada.
    - Creación de `DataLoader` para manejar los lotes de secuencias.

3.  **Definir la Arquitectura del Modelo RNN**:
    ```python
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=50, num_layers=1, output_size=1):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size
            self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_layer_size, output_size)

        def forward(self, input_seq):
            lstm_out, _ = self.lstm(input_seq)
            # Tomar la última salida de la secuencia LSTM
            predictions = self.linear(lstm_out[:, -1, :])
            return predictions

    # Se define una clase similar para GRUModel
    ```
    - Los modelos LSTM y GRU se definen como clases que heredan de `nn.Module`.

4.  **Entrenar el Modelo**:
    ```python
    criterion = nn.MSELoss()
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=CONFIG_TRAIN['learning_rate'])

    for epoch in range(CONFIG_TRAIN['num_epochs']):
        model_lstm.train()
        for seq, labels in train_loader_lstm:
            seq, labels = seq.to(device), labels.to(device)
            optimizer_lstm.zero_grad()
            y_pred = model_lstm(seq)
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer_lstm.step()
        # ... Lógica de evaluación por época ...
    ```
    - Se utiliza `MSELoss` como función de pérdida.
    - El entrenamiento se realiza en un bucle explícito.

5.  **Evaluar el Modelo y Realizar Predicciones**:
    - Se evalúa el modelo en modo `eval()`.
    - Se realizan predicciones y se desescalan.
    - Se comparan gráficamente las predicciones con los datos reales.

---

## Conclusión General

Estos notebooks y scripts proporcionan ejemplos detallados y prácticos para comenzar a trabajar con preprocesamiento de datos y la implementación de Redes Neuronales Convolucionales y Recurrentes, utilizando tanto Keras como PyTorch. Permiten comparar los enfoques y la sintaxis de ambas bibliotecas para tareas similares y fundamentales en el campo del Machine Learning y Deep Learning.