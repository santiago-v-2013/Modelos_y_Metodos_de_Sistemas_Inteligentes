# Repositorio de Ejemplos de Modelado de la sección 1

Este repositorio contiene una colección de scripts de Jupyter Notebooks (`.ipynb`) que demuestran diversas técnicas de modelado de datos. Cada ejemplo está diseñado para ser educativo, mostrando paso a paso cómo aplicar diferentes algoritmos a conjuntos de datos específicos, con un enfoque en la estructura clara y la interpretabilidad.

## Motivación

El objetivo principal de estos ejemplos es proporcionar una guía práctica y bien documentada para:
* Implementar y comprender algoritmos de lógica difusa.
* Implementar y comprender árboles de decisión y ensambles como Random Forest.
* Comparar diferentes enfoques de modelado para un mismo problema.

## Estructura General de los Ejemplos

Cada ejemplo en este repositorio sigue una estructura estandarizada para facilitar su comprensión y replicación:

1.  **Título y Objetivo:**
    * Nombre claro del ejercicio y la técnica principal utilizada.
    * Descripción de la disciplina y el objetivo principal del notebook.
2.  **Carga de Librerías y Configuración Inicial:**
    * Importación de todas las bibliotecas necesarias.
    * Configuraciones globales para visualizaciones o recarga de módulos.
3.  **Funciones Personalizadas:**
    * Definición de funciones auxiliares para modularizar el código (carga de datos, entrenamiento, visualización, etc.).
    * Cada función incluye una descripción detallada de su objetivo, parámetros y valor de retorno.
4.  **Desarrollo del Ejercicio / Análisis Principal:**
    * Subsecciones detalladas que cubren:
        * Carga y exploración inicial de datos.
        * Preprocesamiento (si aplica).
        * Definición, entrenamiento y evaluación del modelo.
        * Visualización de resultados (gráficos del modelo, importancia de características, etc.).
        * Análisis de ejemplos específicos.
5.  **Conclusiones del Ejercicio:**
    * Resumen de los hallazgos clave.
    * Discusión sobre las fortalezas y debilidades del modelo/técnica aplicada.
    * Aprendizajes generales y posibles mejoras o trabajos futuros.
    * En los ejemplos comparativos, se incluye una sección dedicada a la comparación detallada de los modelos.

## Ejemplos Incluidos

A continuación, se listan los principales ejemplos desarrollados hasta la fecha:

### 1. Lógica Difusa

* **`Ejemplo_problema_de_la_propina_fuzzy.ipynb`**
    * **Descripción:** Implementa un sistema de control de lógica difusa para resolver el clásico "problema de la propina". Se basa en la calidad del servicio y la comida para determinar el porcentaje de propina.
    * **Dataset:** No aplica (problema sintético basado en reglas).
    * **Técnicas:** Lógica Difusa, `scikit-fuzzy`, Antecedentes, Consecuentes, Funciones de Pertenencia (automáticas y personalizadas), Reglas Difusas, Simulación de Control.
    * **Puntos Clave:** Definición de variables lingüísticas, creación de reglas intuitivas, visualización de funciones de pertenencia y proceso de inferencia.

* **`Ejemplo_planta_versicolor_fuzzy.ipynb`**
    * **Descripción:** Construye un sistema de control de lógica difusa para clasificar las especies de flores del dataset Iris basado en sus características (longitud y ancho de sépalos y pétalos).
    * **Dataset:** Iris (`sklearn.datasets.load_iris`).
    * **Técnicas:** Lógica Difusa, `scikit-fuzzy`, Clasificación, Funciones de Pertenencia Triangulares, Reglas Difusas para Clasificación.
    * **Puntos Clave:** Aplicación de lógica difusa a un problema de clasificación real, definición de reglas para distinguir entre clases, evaluación de la precisión del sistema difuso. Se corrigió el número de reglas para un funcionamiento adecuado.

* **`Ejemplo_control_fuzzy_simple.ipynb`**
    * **Descripción:** Define un sistema de control difuso simple para determinar la velocidad de un ventilador en función de la temperatura y la humedad ambiente.
    * **Dataset:** No aplica (problema sintético basado en reglas).
    * **Técnicas:** Lógica Difusa, `scikit-fuzzy`, Visualización de Funciones de Pertenencia y Salida de Simulación.
    * **Puntos Clave:** Ejemplo conciso para ilustrar la definición de variables, funciones de pertenencia, reglas y la simulación de un sistema de control con entradas específicas.

### 2. Árboles de Decisión y Ensambles

* **`Ejemplo_planta_versicolor_AD.ipynb`**
    * **Descripción:** Implementa un clasificador de Árbol de Decisión único para clasificar las especies de flores del dataset Iris.
    * **Dataset:** Iris (`sklearn.datasets.load_iris`).
    * **Técnicas:** Árbol de Decisión (`DecisionTreeClassifier`), `train_test_split`, `export_text` (para reglas), `plot_tree` (para visualización), evaluación de precisión.
    * **Puntos Clave:** Énfasis en la interpretabilidad del árbol (visualización y reglas textuales), análisis de predicciones para ejemplos específicos mostrando la clase real.

* **`Ejemplo_diagnostico_de_cancer_AD.ipynb`**
    * **Descripción:** Construye un ensamble de Árboles de Decisión (Random Forest) para predecir si un tumor de mama es maligno o benigno.
    * **Dataset:** Breast Cancer Wisconsin (`sklearn.datasets.load_breast_cancer`).
    * **Técnicas:** Random Forest (`RandomForestClassifier`), `train_test_split`, Métricas de Evaluación (Precisión, Reporte de Clasificación, Matriz de Confusión), Importancia de Características, Visualización de un árbol individual del ensamble.
    * **Puntos Clave:** Introducción a los ensambles, evaluación más completa del rendimiento, interpretabilidad a través de la importancia de características.

* **`Ejemplo_vino_AD.ipynb`**
    * **Descripción:** Compara directamente el rendimiento y las características de un Árbol de Decisión único frente a un ensamble Random Forest en la tarea de clasificar tipos de vino.
    * **Dataset:** Wine (`sklearn.datasets.load_wine`).
    * **Técnicas:** `DecisionTreeClassifier`, `RandomForestClassifier`, comparación de métricas, interpretabilidad, robustez y sobreajuste.
    * **Puntos Clave:** Discusión detallada de las ventajas y desventajas de cada modelo, ayudando a decidir cuál usar según el contexto del problema.

## Requisitos

Para ejecutar estos ejemplos, necesitará tener Python 3 instalado, junto con las siguientes bibliotecas principales:

* `numpy`
* `pandas`
* `scikit-learn`
* `scikit-fuzzy` (para los ejemplos de lógica difusa)
* `matplotlib`
* `seaborn`
* `jupyter` (si desea trabajar con los archivos `.ipynb` o convertir los `.py` a formato notebook)
* `jupytext` (opcional, para la conversión entre formatos `.py` y `.ipynb`)
* `graphviz` (opcional, si desea usar la función `export_graphviz` para visualizaciones avanzadas de árboles; requiere instalación a nivel de sistema operativo además del paquete Python).

Puede instalar la mayoría de estas dependencias usando pip:
```bash
pip install numpy pandas scikit-learn scikit-fuzzy matplotlib seaborn jupyter jupytext graphviz