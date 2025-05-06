# Sección 2: Clasificación y Regresión con Redes Neuronales

En esta sección, encontrarás ejemplos prácticos sobre cómo implementar redes neuronales para resolver problemas de clasificación y regresión utilizando las bibliotecas Keras y PyTorch. Estos notebooks están diseñados para facilitar la comprensión del código y su ejecución.

## Contenido

1. [Ejemplo_con_keras_para_clasificacion_y_regresion.ipynb](Ejemplo_con_keras_para_clasificacion_y_regresion.ipynb)
2. [Ejemplo_con_pytorch_para_clasificacion_y_regresion.ipynb](Ejemplo_con_pytorch_para_clasificacion_y_regresion.ipynb)

---

### Ejemplo 1: Clasificación y Regresión usando Keras

#### Descripción

En este notebook, se utiliza la biblioteca Keras para construir y entrenar redes neuronales que abordan tanto problemas de clasificación como de regresión. Incluye ejemplos paso a paso, desde la carga de datos hasta la evaluación del modelo.

#### Cómo usar el notebook

1. **Importar las librerías necesarias**:
    ```python
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ```
    - `tensorflow.keras.models.Sequential`: Permite construir un modelo de red neuronal secuencial.
    - `tensorflow.keras.layers.Dense`: Crea capas densas completamente conectadas.
    - `sklearn.model_selection.train_test_split`: Divide el conjunto de datos en entrenamiento y prueba.
    - `sklearn.preprocessing.StandardScaler`: Escala las características para mejorar la convergencia del modelo.

2. **Cargar y preprocesar los datos**:
    ```python
    data = pd.read_csv('ruta/del/archivo.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ```
    - `data.drop('target', axis=1)`: Separa las características independientes (X) del objetivo (y).
    - `train_test_split`: Divide los datos en 70% para entrenamiento y 30% para prueba.
    - `StandardScaler`: Normaliza las características para que tengan media 0 y desviación estándar 1.

3. **Definir y entrenar el modelo**:
    ```python
    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
    ```
    - La red tiene:
        - 16 neuronas en la primera capa oculta.
        - 8 neuronas en la segunda capa oculta.
        - Una salida con activación Sigmoid para problemas binarios.
    - `model.compile`: Configura la función de pérdida, el optimizador y las métricas de evaluación.
    - `model.fit`: Entrena el modelo con 50 épocas y un tamaño de lote de 10.

4. **Evaluar el modelo**:
    ```python
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    ```
    - Evalúa el modelo en el conjunto de prueba y calcula la precisión.

---

### Ejemplo 2: Clasificación y Regresión usando PyTorch

#### Descripción

En este notebook, se utiliza PyTorch para construir y entrenar redes neuronales que abordan problemas de clasificación y regresión. Presenta un enfoque detallado para definir el modelo, entrenarlo y evaluarlo.

#### Cómo usar el notebook

1. **Importar las librerías necesarias**:
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ```
    - `torch`: Biblioteca principal de PyTorch que incluye tensores y funciones de aprendizaje profundo.
    - `torch.nn`: Módulo para crear arquitecturas de redes neuronales.
    - `torch.optim`: Módulo para configuraciones de optimización.

2. **Preparar los datos de entrada**:
    ```python
    data = pd.read_csv('ruta/del/archivo.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ```
    - Similar al preprocesamiento en Keras, se divide y escala el conjunto de datos.

3. **Definir el modelo en PyTorch**:
    ```python
    class NeuralNet(nn.Module):
        def __init__(self, input_size):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    model = NeuralNet(X_train.shape[1])
    ```
    - La arquitectura incluye:
        - Una capa oculta con 16 neuronas.
        - Una segunda capa oculta con 8 neuronas.
        - Una salida con activación Sigmoid.

4. **Entrenar el modelo**:
    ```python
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        outputs = model(torch.from_numpy(X_train).float())
        loss = criterion(outputs, torch.from_numpy(y_train).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```
    - La función de pérdida utilizada es `BCELoss` (entropía cruzada binaria).
    - Se entrena durante 50 épocas.

5. **Evaluar el modelo**:
    ```python
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test).float()).round()
        accuracy = (y_pred.numpy() == y_test).mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")
    ```
    - Se calcula la precisión redondeando las predicciones para comparar con los valores reales.

---

## Conclusión

Estos notebooks proporcionan ejemplos prácticos y detallados de cómo implementar redes neuronales con Keras y PyTorch para resolver problemas de clasificación y regresión. Puedes modificar los parámetros y configuraciones para explorar diferentes escenarios y optimizar los resultados.