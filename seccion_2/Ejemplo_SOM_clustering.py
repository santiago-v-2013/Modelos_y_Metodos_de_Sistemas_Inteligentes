#%% [markdown]
# # Mapas Autoorganizados (SOM) para Visualización y Clustering del Dataset Breast Cancer Wisconsin
#
# **Disciplina:** Aprendizaje No Supervisado, Redes Neuronales, Mapas Autoorganizados de Kohonen, Visualización de Datos, Clustering
#
# **Objetivo:**
# El objetivo de este notebook es implementar un Mapa Autoorganizado (SOM) para analizar el dataset `Breast Cancer Wisconsin (Diagnostic)`. Las SOMs se utilizarán para producir una representación de baja dimensión (mapa 2D) de estos datos tabulares (30 características), con la intención de visualizar la estructura inherente, identificar posibles clusters de muestras (malignas vs. benignas) y ver cómo los diferentes tipos de diagnóstico se mapean en la SOM. Se utilizará la biblioteca `MiniSom`.

#%% [markdown]
# ## 1. Carga de Librerías y Configuración Inicial
#
# **Propósito de esta sección:**
# Importar todas las bibliotecas necesarias y configurar el entorno para el análisis.
#
# **Bibliotecas Clave:**
# * **`numpy`, `pandas`**: Para manipulación de datos.
# * **`matplotlib.pyplot`, `seaborn`**: Para visualizaciones.
# * **`sklearn.datasets`**: Para cargar el dataset `Breast Cancer Wisconsin`.
# * **`sklearn.preprocessing`**: Para `StandardScaler`.
# * **`minisom`**: La biblioteca para implementar el SOM.
#
# **Nota de Dependencia:**
# Este ejemplo requiere la biblioteca `MiniSom`. Puedes instalarla usando pip:
# `pip install MiniSom`

#%%
# Comandos mágicos de IPython (opcional en scripts)
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

#%%
# Importación de bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict # Importar defaultdict

from sklearn.datasets import load_breast_cancer # Cambiado
from sklearn.preprocessing import StandardScaler

# Importar MiniSom (asegúrate de que esté instalado)
MINISOM_IMPORTED_SUCCESSFULLY = False
try:
    from minisom import MiniSom
    print(f"Biblioteca 'MiniSom' importada correctamente.")
    MINISOM_IMPORTED_SUCCESSFULLY = True
except ImportError as e:
    print(f"Error al importar 'MiniSom': {e}")
    print("Por favor, instálala con 'pip install MiniSom'")
    print("El script continuará, pero las secciones de SOM probablemente fallarán.")

# Configuración para reproducibilidad
SEED = 42
np.random.seed(SEED) 

# Configuración de estilo y visualización
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 8] 
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

#%% [markdown]
# ## 2. Funciones Personalizadas

#%% [markdown]
# ### Descripción de la Función: `cargar_y_preparar_datos_cancer_som`
#
# **Objetivo Principal:**
# Cargar el dataset `Breast Cancer Wisconsin` y preprocesarlo para su uso con SOMs, escalando las características.
#
# **Características:**
# * **Procesamiento:**
#     1. Carga el dataset `Breast Cancer`.
#     2. Separa características (X) y objetivo (y).
#     3. Escala las características X usando `StandardScaler`.
# * **Valor de Retorno:**
#     * `X_scaled_np`: Características escaladas.
#     * `y_np`: Etiquetas originales (0 para maligno, 1 para benigno).
#     * `scaler`: El objeto `StandardScaler` ajustado.
#     * `feature_names`, `target_names` (lista de strings 'malignant', 'benign').

#%%
def cargar_y_preparar_datos_cancer_som():
    """
    Carga y preprocesa el dataset Breast Cancer Wisconsin para SOM.
    """
    print("Cargando y preparando el dataset Breast Cancer Wisconsin para SOM...")
    cancer = load_breast_cancer()
    X_np = cancer.data # (569, 30)
    y_np = cancer.target # 0: malignant, 1: benign
    feature_names = cancer.feature_names
    target_names = list(cancer.target_names) # ['malignant', 'benign']

    # Crear DataFrame para exploración
    df = pd.DataFrame(X_np, columns=feature_names)
    df['diagnosis_code'] = y_np
    df['diagnosis_name'] = df['diagnosis_code'].map({i: name for i, name in enumerate(target_names)})
    print("\nPrimeras filas del dataset Breast Cancer:")
    print(df.head(3))
    print(f"\nNúmero de características: {X_np.shape[1]}")
    print(f"\nDistribución de clases: \n{df['diagnosis_name'].value_counts(normalize=True)}")


    # Escalar características es crucial para SOMs
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X_np)
    
    print(f"\nDimensiones de datos escalados: X_scaled_np: {X_scaled_np.shape}")
    return X_scaled_np, y_np, scaler, feature_names, target_names

#%% [markdown]
# ### Descripción de la Función: `inicializar_y_entrenar_som`
# (Sin cambios respecto a la versión anterior, es genérica)
#%%
def inicializar_y_entrenar_som(data, map_x_dim, map_y_dim, input_len, 
                               sigma=1.0, learning_rate=0.5, 
                               num_iterations=10000, random_seed=SEED):
    if not MINISOM_IMPORTED_SUCCESSFULLY:
        print("MiniSom no importado. No se puede inicializar ni entrenar la SOM.")
        return None
    print(f"\nInicializando SOM de {map_x_dim}x{map_y_dim} neuronas...")
    som = MiniSom(x=map_x_dim, y=map_y_dim, input_len=input_len,
                  sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian', 
                  random_seed=random_seed)
    print("Inicializando pesos de la SOM con PCA...")
    som.pca_weights_init(data) 
    print(f"Entrenando SOM por {num_iterations} iteraciones...")
    som.train_random(data, num_iterations, verbose=True) 
    print("Entrenamiento de SOM completado.")
    return som

#%% [markdown]
# ### Descripción de la Función: `visualizar_mapa_distancias_som` (U-Matrix)
# (Sin cambios)
#%%
def visualizar_mapa_distancias_som(som_model):
    if not MINISOM_IMPORTED_SUCCESSFULLY or som_model is None: return
    print("\nVisualizando Mapa de Distancias (U-Matrix)...")
    map_x_dim = som_model.get_weights().shape[0]
    map_y_dim = som_model.get_weights().shape[1]
    plt.figure(figsize=(map_y_dim, map_x_dim)) # Ajustar tamaño
    plt.pcolor(som_model.distance_map().T, cmap='bone_r') 
    plt.colorbar(label='Distancia Promedio a Vecinos')
    plt.title('Mapa de Distancias Unificadas (U-Matrix)')
    plt.xticks(np.arange(map_x_dim) + 0.5, np.arange(map_x_dim))
    plt.yticks(np.arange(map_y_dim) + 0.5, np.arange(map_y_dim))
    plt.gca().invert_yaxis()
    plt.show()

#%% [markdown]
# ### Descripción de la Función: `visualizar_mapa_activaciones_som`
# (Sin cambios)
#%%
def visualizar_mapa_activaciones_som(som_model, data):
    if not MINISOM_IMPORTED_SUCCESSFULLY or som_model is None: return
    print("\nVisualizando Mapa de Activaciones (Frecuencia de BMU)...")
    map_x_dim = som_model.get_weights().shape[0]
    map_y_dim = som_model.get_weights().shape[1]
    activation_map = np.zeros((map_x_dim, map_y_dim))
    for x_sample in data:
        w = som_model.winner(x_sample)
        activation_map[w[0], w[1]] += 1
    plt.figure(figsize=(map_y_dim, map_x_dim)) # Ajustar tamaño
    plt.pcolor(activation_map.T, cmap='viridis') 
    plt.colorbar(label='Frecuencia de Activación (BMU)')
    plt.title('Mapa de Activaciones (Frecuencia de BMU)')
    plt.xticks(np.arange(map_x_dim) + 0.5, np.arange(map_x_dim))
    plt.yticks(np.arange(map_y_dim) + 0.5, np.arange(map_y_dim))
    plt.gca().invert_yaxis()
    plt.show()

#%% [markdown]
# ### Descripción de la Función: `visualizar_mapa_som_con_etiquetas`
# (Adaptada para etiquetas binarias y mejor visualización)
#%%
def visualizar_mapa_som_con_etiquetas(som_model, data, labels_np, target_names_list):
    if not MINISOM_IMPORTED_SUCCESSFULLY or som_model is None:
        print("MiniSom no disponible o modelo no entrenado. No se puede visualizar con etiquetas.")
        return

    print("\nVisualizando Mapa SOM con Etiquetas de Clase (Diagnóstico)...")
    
    map_x_dim = som_model.get_weights().shape[0]
    map_y_dim = som_model.get_weights().shape[1]

    plt.figure(figsize=(map_y_dim + 2, map_x_dim)) # Ajustar tamaño para leyenda
    
    # Fondo con U-Matrix
    plt.pcolor(som_model.distance_map().T, cmap='bone_r', alpha=0.6)
    
    # Colores y marcadores para las clases (maligno, benigno)
    # target_names_list[0] es 'malignant' (label 0), target_names_list[1] es 'benign' (label 1)
    colors = {0: '#FF5733', 1: '#33C4FF'} # Rojo para maligno, Azul para benigno
    markers = {0: 'x', 1: 'o'}
    
    # Mapear cada muestra a su BMU y colocar un marcador
    for i, sample in enumerate(data):
        bmu_coord = som_model.winner(sample) # (x_col, y_row)
        label_class = labels_np[i]
        
        # Añadir un pequeño jitter para que los puntos no se superpongan exactamente
        jitter_x = (np.random.rand() - 0.5) * 0.4
        jitter_y = (np.random.rand() - 0.5) * 0.4
        
        plt.plot(bmu_coord[0] + 0.5 + jitter_x, 
                 bmu_coord[1] + 0.5 + jitter_y,
                 markers[label_class],
                 markerfacecolor='None', # Solo borde del marcador
                 markeredgecolor=colors[label_class],
                 markersize=10, 
                 markeredgewidth=1.5,
                 alpha=0.7)

    plt.title('Mapa SOM con Diagnóstico Superpuesto')
    plt.xticks(np.arange(map_x_dim) + 0.5, np.arange(map_x_dim))
    plt.yticks(np.arange(map_y_dim) + 0.5, np.arange(map_y_dim))
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
    plt.gca().invert_yaxis()
    
    # Crear leyenda manualmente
    handles = [plt.Line2D([0], [0], marker=markers[i], color='w', 
                          markerfacecolor='None', markeredgecolor=colors[i], 
                          markersize=10, markeredgewidth=1.5) for i in range(len(target_names_list))]
    plt.legend(handles, target_names_list, title='Diagnóstico', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para leyenda
    plt.show()

#%% [markdown]
# ### Descripción de la Función: `visualizar_mapa_componentes_som`
# (Adaptada para manejar 30 planos de componentes)
#%%
def visualizar_mapa_componentes_som(som_model, feature_names_list, max_planes_to_show=None):
    if not MINISOM_IMPORTED_SUCCESSFULLY or som_model is None:
        print("MiniSom no disponible o modelo no entrenado. No se puede visualizar los planos de componentes.")
        return

    print("\nVisualizando Planos de Componentes SOM...")
    weights = som_model.get_weights() 
    num_features = weights.shape[2]
    map_x_dim = weights.shape[0]
    map_y_dim = weights.shape[1]

    if feature_names_list is None or len(feature_names_list) != num_features:
        feature_names_list = [f'Feature {i+1}' for i in range(num_features)]

    num_planes_actual = num_features
    if max_planes_to_show is not None and num_features > max_planes_to_show:
        num_planes_actual = max_planes_to_show
        print(f"Mostrando los primeros {max_planes_to_show} de {num_features} planos de componentes.")
    
    cols = 5 # Ajustar número de columnas para 30 características
    rows = (num_planes_actual + cols - 1) // cols 
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3)) # Ajustar tamaño
    axes = axes.flatten() 

    for i in range(num_planes_actual):
        ax = axes[i]
        component_plane = weights[:, :, i]
        im = ax.pcolor(component_plane.T, cmap='viridis') 
        ax.set_title(f"{feature_names_list[i]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label='Peso')

    for j in range(num_planes_actual, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(f"Planos de Componentes SOM (Primeros {num_planes_actual} de {num_features})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()


#%% [markdown]
# ## 3. Desarrollo del Ejercicio: SOM con Dataset Breast Cancer

#%% [markdown]
# ### 3.1. Carga y Preparación de Datos
#
# Cargamos el dataset `Breast Cancer Wisconsin` y escalamos sus 30 características.

#%%
X_cancer_scaled, y_cancer_original, scaler_cancer, cancer_feature_names, cancer_target_names = \
    cargar_y_preparar_datos_cancer_som()

#%% [markdown]
# ### 3.2. Inicialización y Entrenamiento de la SOM
#
# Para el dataset Breast Cancer (569 muestras, 30 características).
# Heurística: `5 * sqrt(569) approx 5 * 23.8 = 119 neuronas`. Un mapa de 10x12 o 11x11.

#%%
som_model_cancer = None
if MINISOM_IMPORTED_SUCCESSFULLY:
    N_SAMPLES_CANCER = X_cancer_scaled.shape[0]
    num_neurons_heuristic_cancer = int(5 * np.sqrt(N_SAMPLES_CANCER))
    MAP_X_DIM_CANCER = int(np.sqrt(num_neurons_heuristic_cancer))
    MAP_Y_DIM_CANCER = (num_neurons_heuristic_cancer // MAP_X_DIM_CANCER) + ((num_neurons_heuristic_cancer % MAP_X_DIM_CANCER) > 0)
    
    print(f"Heurística para tamaño de SOM (Cancer): ~{num_neurons_heuristic_cancer} neuronas. Usando mapa de {MAP_X_DIM_CANCER}x{MAP_Y_DIM_CANCER}.")
    # Podrías fijar un tamaño si prefieres, ej: MAP_X_DIM_CANCER = 10; MAP_Y_DIM_CANCER = 12

    INPUT_LEN_CANCER = X_cancer_scaled.shape[1] # 30 características
    
    SIGMA_INIT_CANCER = 1.8 
    LEARNING_RATE_INIT_CANCER = 0.5
    NUM_ITERATIONS_SOM_CANCER = 20000 # Más iteraciones para datos más complejos

    som_model_cancer = inicializar_y_entrenar_som(
        X_cancer_scaled, MAP_X_DIM_CANCER, MAP_Y_DIM_CANCER, INPUT_LEN_CANCER,
        sigma=SIGMA_INIT_CANCER, learning_rate=LEARNING_RATE_INIT_CANCER,
        num_iterations=NUM_ITERATIONS_SOM_CANCER, random_seed=SEED
    )
else:
    print("Saltando inicialización y entrenamiento de SOM porque MiniSom no se importó.")

#%% [markdown]
# ### 3.3. Visualización de Resultados de la SOM

#%% [markdown]
# #### 3.3.1. Mapa de Distancias (U-Matrix)
# La U-Matrix para el dataset de cáncer.

#%%
if som_model_cancer:
    visualizar_mapa_distancias_som(som_model_cancer)

#%% [markdown]
# #### 3.3.2. Mapa de Activaciones
# Frecuencia de activación de neuronas para las muestras de cáncer.

#%%
if som_model_cancer:
    visualizar_mapa_activaciones_som(som_model_cancer, X_cancer_scaled)

#%% [markdown]
# #### 3.3.3. Mapa SOM con Etiquetas de Diagnóstico
# Superponemos las etiquetas 'malignant' y 'benign' en el mapa.

#%%
if som_model_cancer:
    visualizar_mapa_som_con_etiquetas(som_model_cancer, X_cancer_scaled, y_cancer_original, cancer_target_names)

#%% [markdown]
# #### 3.3.4. Planos de Componentes
# Visualizamos los planos de las 30 características.

#%%
if som_model_cancer:
    visualizar_mapa_componentes_som(som_model_cancer, cancer_feature_names, max_planes_to_show=30) # Mostrar todos


#%% [markdown]
# ## 4. Conclusiones del Ejercicio (SOM con Breast Cancer)
#
# **Resumen de Hallazgos:**
# * Se cargó y preprocesó el dataset `Breast Cancer Wisconsin`, escalando sus 30 características.
# * Se entrenó un Mapa Autoorganizado (SOM) de **[MAP_X_DIM_CANCER x MAP_Y_DIM_CANCER]** neuronas por **[NUM_ITERATIONS_SOM_CANCER]** iteraciones.
# * **Mapa de Distancias (U-Matrix):** La U-Matrix mostró **[Describir: ej., si se observan una o dos regiones principales claras separadas por bordes, o una estructura más gradual. ¿Sugiere dos clusters principales?]**.
# * **Mapa de Activaciones:** Este mapa indicó **[Describir la distribución de activaciones]**.
# * **Mapa SOM con Etiquetas de Diagnóstico:** Al superponer las etiquetas 'malignant' y 'benign', se observó que **[Describir: ej., las muestras malignas y benignas tendieron a agruparse en regiones predominantemente distintas del mapa SOM. ¿Qué tan clara fue la separación? ¿Hubo una zona de transición o mezcla?]**.
# * **Planos de Componentes:** El análisis de los 30 planos de componentes reveló que **[Describir hallazgos clave. ¿Qué características (ej: 'mean radius', 'worst concave points') mostraron patrones claros o gradientes a través del mapa que se correlacionan con las regiones de maligno/benigno? ¿Hubo características que parecían uniformes y por lo tanto menos discriminativas en el mapa SOM?]**.
#
# **Sobre las SOMs con Datos Tabulares como Breast Cancer:**
# * Las SOMs pueden ayudar a visualizar la estructura de separación (o falta de ella) entre clases en un espacio de características de alta dimensión.
# * La U-Matrix y el mapa de etiquetas combinados pueden dar una fuerte indicación de la "dificultad" de separar las clases.
# * Los planos de componentes pueden ofrecer insights sobre qué características son más importantes para la organización topológica del mapa y, por ende, para la distinción entre los grupos que la SOM identifica.
#
# **Aprendizaje General:**
# Este ejercicio demostró el uso de SOMs para la exploración no supervisada del dataset Breast Cancer. Incluso sin conocer las etiquetas de diagnóstico durante el entrenamiento, la SOM fue capaz de organizar los datos en un mapa 2D de tal manera que las clases subyacentes a menudo forman regiones cohesivas. Esto subraya la utilidad de las SOMs para la visualización de datos, la detección de estructuras y la potencial identificación de características relevantes en datasets tabulares complejos.
#
# *(Nota: Las descripciones cualitativas y los hallazgos específicos en los corchetes deben completarse después de ejecutar completamente el notebook y analizar los gráficos generados.)*