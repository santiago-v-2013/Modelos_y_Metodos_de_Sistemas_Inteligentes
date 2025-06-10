# Sección de Ejemplos: Algoritmos Genéticos con DEAP

En este repositorio, encontrarás ejemplos prácticos de cómo utilizar algoritmos genéticos para resolver diversos problemas de optimización usando la librería DEAP (Distributed Evolutionary Algorithms in Python). Los ejemplos cubren minimización no restringida, maximización con restricciones lineales y minimización con restricciones no lineales. Estos notebooks están diseñados para facilitar la comprensión y ejecución del código.

---

## Sección 1: Ejemplos de Algoritmos Genéticos

Esta sección contiene ejemplos prácticos de problemas de optimización resueltos con algoritmos genéticos. Los notebooks están diseñados para ilustrar los pasos fundamentales en la definición y solución de estos problemas utilizando DEAP.

### Contenido

1.  [Ejemplo_1_Algoritmos_genéticos.ipynb](Ejemplo_1_Algoritmos_genéticos.ipynb)
2.  [Ejemplo_2_Algoritmos_genéticos.ipynb](Ejemplo_2_Algoritmos_genéticos.ipynb)
3.  [Ejemplo_3_Algoritmos_genéticos.ipynb](Ejemplo_3_Algoritmos_genéticos.ipynb)

---

### Ejemplo 1: Minimización No Restringida

#### Descripción

Este notebook demuestra cómo resolver un problema de minimización no restringida. El objetivo es ajustar una función exponencial del tipo `$ae^{bx}$` a un conjunto de datos dado, minimizando el Error Cuadrático Medio (MSE) entre los valores predichos y los valores reales.

#### Pasos Clave del Notebook

* **Definición de la Función Objetivo**: La función a minimizar es el MSE, que calcula el error entre las predicciones del modelo y los datos verdaderos.
    ```python
    def objective_function(individual):  # a = individual[0], b = individual[1]
        x_true = [-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0]
        y_true = [36.547, 17.264, 8.155, 3.852, 1.820, 0.860, 0.406, 0.246]
        y_pred = []

        for x_test in x_true:
            Eval = individual[0]*np.exp(individual[1]*x_test)
            y_pred.append(Eval)

        mse = mean_squared_error(y_true, y_pred)
        return mse,
    ```
* **Configuración del Algoritmo Genético con DEAP**:
    * **Creación de Tipos**: Se define un tipo `FitnessMin` para minimización y un `Individual` como una lista que contiene los parámetros `a` y `b`.
    * **Inicialización de la Toolbox**: Se registran funciones en la `Toolbox` para generar individuos con dos atributos de punto flotante (`a` y `b`).
* **Registro de Operadores Genéticos**:
    * `evaluate`: La función objetivo (MSE).
    * `mate`: Cruce binario simulado (`cxSimulatedBinary`).
    * `mutate`: Mutación gaussiana (`mutGaussian`).
    * `select`: Selección por torneo (`selTournament`).
* **Optimización y Resultados**:
    * El algoritmo genético se ejecuta utilizando el algoritmo `eaSimple` de DEAP.
    * Se muestra el mejor individuo encontrado (valores de `a` y `b`) y su aptitud correspondiente (MSE mínimo).
    * Se genera un gráfico para visualizar la curva exponencial ajustada frente a los puntos de datos originales.

---

### Ejemplo 2: Maximización con Restricciones Lineales

#### Descripción

Este notebook ilustra cómo resolver un problema de maximización sujeto a un conjunto de restricciones lineales. El objetivo es maximizar la función `$Z = 3x_1 + 5x_2$` respetando todas las restricciones definidas.

#### Pasos Clave del Notebook

* **Planteamiento del Problema**:
    * **Función Objetivo**: `$\text{Maximizar } Z = 3x_1 + 5x_2$`
    * **Restricciones**:
        * `$x_1 \leq 4000$`
        * `$2x_2 \leq 12000$`
        * `$3x_1 + 2x_2 \leq 18000$`
        * `$x_1 \geq 0, \quad x_2 \geq 0$`
* **Funciones de Evaluación, Factibilidad y Penalización**:
    * Una **función objetivo** calcula el valor de Z.
    * Una **función de factibilidad** (`feasible`) comprueba si un individuo (`x₁`, `x₂`) satisface todas las restricciones, devolviendo `True` o `False`.
    * Una **función de penalización** (`distance`) calcula una penalización para las soluciones no factibles, que DEAP utiliza para ajustar su aptitud.
* **Configuración del Algoritmo Genético con DEAP**:
    * **Creación de Tipos**: Se define un tipo `FitnessMax` para maximización.
    * **Inicialización de la Toolbox**: La `Toolbox` se configura para crear individuos con dos atributos enteros (`x₁` y `x₂`).
    * **Registro de Operadores Genéticos**:
        ```python
        # Registrar la función objetivo y decorarla con la penalización
        toolbox.register('evaluate', objective_function)
        toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0, distance))

        # Registrar otros operadores
        toolbox.register('mate', tools.cxOnePoint)
        toolbox.register('mutate', mutation_gaussian_int, indp=0.7)
        toolbox.register('select', tools.selTournament, tournsize=3)
        ```
        - El decorador `DeltaPenalty` es una característica clave, que penaliza automáticamente a los individuos que la función `feasible` marca como inválidos.
* **Optimización y Resultados**:
    * El algoritmo se ejecuta usando `eaSimple`.
    * Se muestra la mejor solución encontrada para `x₁` y `x₂` y el valor máximo de `Z`.
    * Un gráfico visualiza el espacio de búsqueda, la región factible (el área que satisface todas las restricciones) y la solución óptima encontrada por el algoritmo.

---

### Ejemplo 3: Minimización con Restricciones No Lineales

#### Descripción

Este notebook demuestra cómo manejar un problema de minimización con restricciones no lineales. El objetivo es encontrar un punto `$(x, y)$` que minimice la suma de sus distancias euclidianas a tres centros fijos, con la condición de que la solución no debe estar a más de 10 unidades de distancia de ninguno de los centros.

#### Pasos Clave del Notebook

* **Planteamiento del Problema**:
    * **Función Objetivo**: Minimizar la suma de distancias desde un punto (x, y) a tres centros fijos `$C_1, C_2, C_3$`.
        `$ \text{Minimizar } Z = \sum_{i=1}^{3} \sqrt{(x_{C_i} - x)^2 + (y_{C_i} - y)^2} $`
    * **Restricciones No Lineales**: La distancia desde el punto solución a cada centro debe ser menor o igual a 10.
        `$ \sqrt{(x_{C_i} - x)^2 + (y_{C_i} - y)^2} \leq 10 $`
* **Funciones de Evaluación, Factibilidad y Penalización**:
    * Al igual que en el ejemplo anterior, se definen una **función objetivo**, una **función de factibilidad** y una **función de penalización** para manejar las restricciones no lineales.
* **Configuración del Algoritmo Genético con DEAP**:
    * **Creación de Tipos**: Se utiliza `FitnessMin` para la minimización.
    * **Inicialización de la Toolbox**: La `Toolbox` se configura para crear individuos con dos atributos enteros (coordenadas `x` e `y`).
    * **Registro de Operadores Genéticos**: Se utiliza nuevamente el decorador `DeltaPenalty` para gestionar las restricciones no lineales.
        ```python
        # Registrar la función objetivo y decorarla con la penalización
        toolbox.register('evaluate', objective_function)
        toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0, distance))
        ```
* **Optimización y Resultados**:
    * Se ejecuta el algoritmo `eaSimple` para encontrar la solución óptima.
    * El resultado final muestra las coordenadas del punto (`x`, `y`) que minimiza la función objetivo cumpliendo todas las restricciones de distancia.
    * Un mapa de contorno visualiza el espacio de búsqueda, donde el color indica el valor de la función objetivo. También se trazan la región factible (la intersección de los tres círculos de restricción) y la solución óptima.

---

## Sección 2: Optimización por Enjambre de Partículas (PSO)

Esta sección se centra en el uso del algoritmo de Optimización por Enjambre de Partículas (PSO), otra potente metaheurística para resolver problemas de optimización. Los ejemplos muestran cómo implementar PSO utilizando la librería DEAP.

### Contenido

1.  [Ejemplo_1_particle_swarm_optimization.ipynb](Ejemplo_1_particle_swarm_optimization.ipynb)
2.  [Ejemplo_2_particle_swarm_optimization.ipynb](Ejemplo_2_particle_swarm_optimization.ipynb)

---

### Ejemplo 4: Maximización de Área con Restricción Lineal (PSO)

#### Descripción

Este notebook utiliza el algoritmo PSO para maximizar el área de un terreno rectangular (`Área = x * y`) con un presupuesto limitado para la valla (`Perímetro = 2x + y ≤ 500`). Es un problema de maximización con una restricción lineal.

#### Pasos Clave del Notebook

* **Definición del Problema**: Maximizar `f(x, y) = x * y` sujeto a `2x + y ≤ 500`.
* **Configuración de PSO en DEAP**:
    * Se crea un tipo `Particle` que hereda de una lista y tiene atributos como `speed`, `pbest` (mejor posición personal) y `sbest` (mejor velocidad).
    * La `Toolbox` se configura para generar partículas y registrar el operador `update`, que es el núcleo del PSO y se encarga de actualizar la velocidad y posición de las partículas.
    ```python
    # Registrar la función de actualización de partículas (núcleo de PSO)
    toolbox.register("update", pso.generate, smin=smin, smax=smax)
    ```
* **Manejo de Restricciones**: Se utiliza una función de factibilidad y una de penalización, similar a los ejemplos de algoritmos genéticos, para guiar al enjambre hacia la región factible.
* **Ejecución y Resultados**: El algoritmo se ejecuta durante un número de generaciones, y al final se muestra el área máxima obtenida y las dimensiones `x` e `y` óptimas.

---

### Ejemplo 5: Maximización de Volumen con Restricción de Igualdad (PSO)

#### Descripción

Este ejemplo aborda un problema de ingeniería clásico: determinar las dimensiones de una caja con base cuadrada que maximicen su volumen, utilizando una cantidad fija de material (10 m²). Esto se traduce en una restricción de igualdad no lineal (`Área = x² + 4xh = 10`).

#### Pasos Clave del Notebook

* **Planteamiento del Problema**: Maximizar `Volumen = x² * h` sujeto a la restricción de área `x² + 4xh = 10`.
* **Manejo de la Restricción de Igualdad**: La función de factibilidad se adapta para manejar una restricción de igualdad, permitiendo una pequeña tolerancia para considerarla cumplida.
    ```python
    def feasible(individual):
        x, h = individual
        area = x**2 + 4 * x * h
        # Permitir una pequeña tolerancia para la restricción de igualdad
        return abs(area - 10.0) < 1e-6
    ```
* **Configuración de PSO**: La configuración es análoga al ejemplo anterior, definiendo partículas, límites de velocidad y la función de actualización.
* **Ejecución y Resultados**: El algoritmo PSO encuentra las dimensiones `x` y `h` que maximizan el volumen de la caja sin exceder el material disponible. Se imprime el volumen máximo y las dimensiones óptimas.

---

## Conclusión General

Estos notebooks proporcionan ejemplos detallados y prácticos para comenzar a trabajar tanto con Algoritmos Genéticos como con Optimización por Enjambre de Partículas usando la librería DEAP. Demuestran su flexibilidad para manejar diferentes tipos de problemas y restricciones, convirtiéndola en una herramienta poderosa para resolver problemas complejos con computación evolutiva.