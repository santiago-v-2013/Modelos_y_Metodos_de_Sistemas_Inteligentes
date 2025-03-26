# Sección 1: Lógica Fuzzy y Árboles de Decisión

En esta sección, encontrarás ejemplos prácticos sobre la implementación de lógica fuzzy y árboles de decisión. Estos ejemplos están desarrollados en Jupyter Notebooks para facilitar la comprensión y ejecución del código.

## Contenido

1. [Ejemplo_Arboles_de_decision.ipynb](Ejemplo_Arboles_de_decision.ipynb)
2. [Ejemplo_Logica_Fuzzy.ipynb](Ejemplo_Logica_Fuzzy.ipynb)

### Ejemplo 1: Árboles de Decisión

#### Descripción

Este notebook presenta un ejemplo de cómo construir y utilizar un árbol de decisión para la clasificación de datos. Los árboles de decisión son una herramienta poderosa para el análisis de datos y la toma de decisiones.

#### Cómo usar el notebook

1. **Importar las librerías necesarias**: 
    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics
    ```
    Estas librerías son esenciales para la manipulación de datos (`pandas`), la división de datos en conjuntos de entrenamiento y prueba (`train_test_split`), la creación del modelo de árbol de decisión (`DecisionTreeClassifier`) y la evaluación del modelo (`metrics`).

2. **Cargar el conjunto de datos**:
    ```python
    data = pd.read_csv('ruta/del/archivo.csv')
    ```
    Aquí se carga el conjunto de datos que se utilizará. Modifica `'ruta/del/archivo.csv'` con la ruta a tu archivo de datos.

3. **Preprocesar los datos**:
    ```python
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    ```
    - `X = data.drop('target', axis=1)`: Separa las características (X) eliminando la columna 'target'.
    - `y = data['target']`: Asigna la columna 'target' a la variable objetivo (y).
    - `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)`: Divide los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba).

4. **Crear y entrenar el modelo**:
    ```python
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    ```
    - `clf = DecisionTreeClassifier()`: Crea una instancia del clasificador de árbol de decisión.
    - `clf = clf.fit(X_train, y_train)`: Entrena el modelo con los datos de entrenamiento.

5. **Hacer predicciones y evaluar el modelo**:
    ```python
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    ```
    - `y_pred = clf.predict(X_test)`: Realiza predicciones con el conjunto de prueba.
    - `print("Accuracy:", metrics.accuracy_score(y_test, y_pred))`: Imprime la precisión del modelo comparando las predicciones con los valores reales.

### Ejemplo 2: Lógica Fuzzy

#### Descripción

Este notebook proporciona un ejemplo de cómo implementar lógica fuzzy para la toma de decisiones en sistemas inciertos. La lógica fuzzy es una extensión de la lógica booleana que maneja valores de verdad parciales.

#### Cómo usar el notebook

1. **Importar las librerías necesarias**:
    ```python
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    ```
    Estas librerías permiten trabajar con números difusos y reglas de lógica fuzzy (`skfuzzy`).

2. **Definir las variables fuzzy**:
    ```python
    calidad = ctrl.Antecedent(np.arange(0, 11, 1), 'calidad')
    servicio = ctrl.Antecedent(np.arange(0, 11, 1), 'servicio')
    propina = ctrl.Consequent(np.arange(0, 26, 1), 'propina')
    ```
    - `calidad = ctrl.Antecedent(np.arange(0, 11, 1), 'calidad')`: Define la variable fuzzy 'calidad' con un rango de 0 a 10.
    - `servicio = ctrl.Antecedent(np.arange(0, 11, 1), 'servicio')`: Define la variable fuzzy 'servicio' con un rango de 0 a 10.
    - `propina = ctrl.Consequent(np.arange(0, 26, 1), 'propina')`: Define la variable fuzzy 'propina' con un rango de 0 a 25.

3. **Definir las funciones de membresía**:
    ```python
    calidad.automf(3)
    servicio.automf(3)
    propina['baja'] = fuzz.trimf(propina.universe, [0, 0, 13])
    propina['media'] = fuzz.trimf(propina.universe, [0, 13, 25])
    propina['alta'] = fuzz.trimf(propina.universe, [13, 25, 25])
    ```
    - `calidad.automf(3)`: Genera automáticamente tres niveles de calidad (baja, media, alta).
    - `servicio.automf(3)`: Genera automáticamente tres niveles de servicio (baja, media, alta).
    - `propina['baja'] = fuzz.trimf(propina.universe, [0, 0, 13])`: Define la función de membresía 'baja' para la propina.
    - `propina['media'] = fuzz.trimf(propina.universe, [0, 13, 25])`: Define la función de membresía 'media' para la propina.
    - `propina['alta'] = fuzz.trimf(propina.universe, [13, 25, 25])`: Define la función de membresía 'alta' para la propina.

4. **Definir las reglas fuzzy**:
    ```python
    regla1 = ctrl.Rule(calidad['poor'] | servicio['poor'], propina['baja'])
    regla2 = ctrl.Rule(servicio['average'], propina['media'])
    regla3 = ctrl.Rule(servicio['good'] | calidad['good'], propina['alta'])
    ```
    - `regla1 = ctrl.Rule(calidad['poor'] | servicio['poor'], propina['baja'])`: Si la calidad o el servicio son bajos, la propina es baja.
    - `regla2 = ctrl.Rule(servicio['average'], propina['media'])`: Si el servicio es promedio, la propina es media.
    - `regla3 = ctrl.Rule(servicio['good'] | calidad['good'], propina['alta'])`: Si el servicio o la calidad son buenos, la propina es alta.

5. **Crear y simular el sistema de control fuzzy**:
    ```python
    propina_ctrl = ctrl.ControlSystem([regla1, regla2, regla3])
    propina_simulacion = ctrl.ControlSystemSimulation(propina_ctrl)
    propina_simulacion.input['calidad'] = 6.5
    propina_simulacion.input['servicio'] = 9.8
    propina_simulacion.compute()
    print(propina_simulacion.output['propina'])
    ```
    - `propina_ctrl = ctrl.ControlSystem([regla1, regla2, regla3])`: Crea el sistema de control fuzzy con las reglas definidas.
    - `propina_simulacion = ctrl.ControlSystemSimulation(propina_ctrl)`: Crea una simulación del sistema de control.
    - `propina_simulacion.input['calidad'] = 6.5`: Asigna un valor de 6.5 a la calidad.
    - `propina_simulacion.input['servicio'] = 9.8`: Asigna un valor de 9.8 al servicio.
    - `propina_simulacion.compute()`: Ejecuta la simulación para calcular la propina.
    - `print(propina_simulacion.output['propina'])`: Imprime el valor de la propina calculada.

## Conclusión

Estos notebooks proporcionan ejemplos prácticos y detallados de cómo implementar y utilizar modelos de árboles de decisión y lógica fuzzy. Puedes modificar los parámetros y reglas según tus necesidades específicas para explorar diferentes escenarios y resultados.

¡Esperamos que encuentres útiles estos ejemplos en tu aprendizaje y aplicación de sistemas inteligentes!