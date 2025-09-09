#%% [markdown]
# # Self-Organizing Maps (SOM) for Visualization and Clustering of Breast Cancer Wisconsin Dataset
#
# **Discipline:** Unsupervised Learning, Neural Networks, Kohonen Self-Organizing Maps, Data Visualization, Clustering
#
# **Objective:**
# The objective of this notebook is to implement a Self-Organizing Map (SOM) to analyze the `Breast Cancer Wisconsin (Diagnostic)` dataset. SOMs will be used to produce a low-dimensional representation (2D map) of this tabular data (30 features), with the intention of visualizing the inherent structure, identifying possible clusters of samples (malignant vs. benign) and seeing how different diagnosis types map onto the SOM. The `MiniSom` library will be used.

#%% [markdown]
# ## 1. Library Loading and Initial Configuration
#
# **Purpose of this section:**
# Import all necessary libraries and configure the environment for analysis.
#
# **Key Libraries:**
# * **`numpy`, `pandas`**: For data manipulation.
# * **`matplotlib.pyplot`, `seaborn`**: For visualizations.
# * **`sklearn.datasets`**: To load the `Breast Cancer Wisconsin` dataset.
# * **`sklearn.preprocessing`**: For `StandardScaler`.
# * **`minisom`**: The library to implement the SOM.
#
# **Dependency Note:**
# This example requires the `MiniSom` library. You can install it using pip:
# `pip install MiniSom`

#%%
# IPython magic commands (optional in scripts)
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

#%%
# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
import os
import logging

# Add parent directory to path for logging_config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import setup_logging, get_logger

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Set up logging
logger = setup_logging(log_level=logging.INFO)
module_logger = get_logger(__name__)

# Import MiniSom (make sure it's installed)
MINISOM_IMPORTED_SUCCESSFULLY = False
try:
    from minisom import MiniSom
    module_logger.info("'MiniSom' library imported successfully.")
    MINISOM_IMPORTED_SUCCESSFULLY = True
except ImportError as e:
    module_logger.error(f"Error importing 'MiniSom': {e}")
    module_logger.warning("Please install it with 'pip install MiniSom'")
    module_logger.warning("The script will continue, but SOM sections will likely fail.")

# Configuration for reproducibility
SEED = 42
np.random.seed(SEED) 

# Style and visualization configuration
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 8] 
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

#%% [markdown]
# ## 2. Custom Functions

#%% [markdown]
# ### Function Description: `load_and_prepare_cancer_data_som`
#
# **Main Objective:**
# Load the `Breast Cancer Wisconsin` dataset and preprocess it for use with SOMs, scaling the features.
#
# **Characteristics:**
# * **Processing:**
#     1. Loads the `Breast Cancer` dataset.
#     2. Separates features (X) and target (y).
#     3. Scales features X using `StandardScaler`.
# * **Return Value:**
#     * `X_scaled_np`: Scaled features.
#     * `y_np`: Original labels (0 for malignant, 1 for benign).
#     * `scaler`: The fitted `StandardScaler` object.
#     * `feature_names`, `target_names` (list of strings 'malignant', 'benign').

#%%
def load_and_prepare_cancer_data_som():
    """
    Loads and preprocesses the Breast Cancer Wisconsin dataset for SOM.
    """
    module_logger.info("Loading and preparing Breast Cancer Wisconsin dataset for SOM...")
    cancer_data = load_breast_cancer()
    X_np = cancer_data.data # (569, 30)
    y_np = cancer_data.target # 0: malignant, 1: benign
    feature_names = cancer_data.feature_names
    target_names = list(cancer_data.target_names) # ['malignant', 'benign']

    # Create DataFrame for exploration
    df = pd.DataFrame(X_np, columns=feature_names)
    df['diagnosis_code'] = y_np
    df['diagnosis_name'] = df['diagnosis_code'].map({i: name for i, name in enumerate(target_names)})
    module_logger.info("First rows of Breast Cancer dataset:")
    module_logger.info(f"\n{df.head(3)}")
    module_logger.info(f"Number of features: {X_np.shape[1]}")
    module_logger.info(f"Class distribution:\n{df['diagnosis_name'].value_counts(normalize=True)}")

    # Scaling features is crucial for SOMs
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X_np)
    
    module_logger.info(f"Scaled data dimensions: X_scaled_np: {X_scaled_np.shape}")
    return X_scaled_np, y_np, scaler, feature_names, target_names

#%% [markdown]
# ### Function Description: `initialize_and_train_som`
# (No changes from previous version, it's generic)
#%%
def initialize_and_train_som(data, map_x_dim, map_y_dim, input_len, 
                           sigma=1.0, learning_rate=0.5, 
                           num_iterations=10000, random_seed=SEED):
    if not MINISOM_IMPORTED_SUCCESSFULLY:
        module_logger.error("MiniSom not imported. Cannot initialize or train SOM.")
        return None
    module_logger.info(f"Initializing SOM with {map_x_dim}x{map_y_dim} neurons...")
    som = MiniSom(x=map_x_dim, y=map_y_dim, input_len=input_len,
                  sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian', 
                  random_seed=random_seed)
    module_logger.info("Initializing SOM weights with PCA...")
    som.pca_weights_init(data) 
    module_logger.info(f"Training SOM for {num_iterations} iterations...")
    som.train_random(data, num_iterations, verbose=True) 
    module_logger.info("SOM training completed.")
    return som

#%% [markdown]
# ### Function Description: `visualize_som_distance_map` (U-Matrix)
# (No changes)
#%%
def visualize_som_distance_map(som_model):
    if not MINISOM_IMPORTED_SUCCESSFULLY or som_model is None: return
    module_logger.info("Visualizing Distance Map (U-Matrix)...")
    map_x_dim = som_model.get_weights().shape[0]
    map_y_dim = som_model.get_weights().shape[1]
    plt.figure(figsize=(map_y_dim, map_x_dim)) # Adjust size
    plt.pcolor(som_model.distance_map().T, cmap='bone_r') 
    plt.colorbar(label='Average Distance to Neighbors')
    plt.title('Unified Distance Matrix (U-Matrix)')
    plt.xticks(np.arange(map_x_dim) + 0.5, np.arange(map_x_dim))
    plt.yticks(np.arange(map_y_dim) + 0.5, np.arange(map_y_dim))
    plt.gca().invert_yaxis()
    plt.show()

#%% [markdown]
# ### Function Description: `visualize_som_activation_map`
# (No changes)
#%%
def visualize_som_activation_map(som_model, data):
    if not MINISOM_IMPORTED_SUCCESSFULLY or som_model is None: return
    module_logger.info("Visualizing Activation Map (BMU Frequency)...")
    map_x_dim = som_model.get_weights().shape[0]
    map_y_dim = som_model.get_weights().shape[1]
    activation_map = np.zeros((map_x_dim, map_y_dim))
    for x_sample in data:
        w = som_model.winner(x_sample)
        activation_map[w[0], w[1]] += 1
    plt.figure(figsize=(map_y_dim, map_x_dim)) # Adjust size
    plt.pcolor(activation_map.T, cmap='viridis') 
    plt.colorbar(label='Activation Frequency (BMU)')
    plt.title('Activation Map (BMU Frequency)')
    plt.xticks(np.arange(map_x_dim) + 0.5, np.arange(map_x_dim))
    plt.yticks(np.arange(map_y_dim) + 0.5, np.arange(map_y_dim))
    plt.gca().invert_yaxis()
    plt.show()

#%% [markdown]
# ### Function Description: `visualize_som_map_with_labels`
# (Adapted for binary labels and better visualization)
#%%
def visualize_som_map_with_labels(som_model, data, labels_np, target_names_list):
    if not MINISOM_IMPORTED_SUCCESSFULLY or som_model is None:
        module_logger.error("MiniSom not available or model not trained. Cannot visualize with labels.")
        return

    module_logger.info("Visualizing SOM Map with Class Labels (Diagnosis)...")
    
    map_x_dim = som_model.get_weights().shape[0]
    map_y_dim = som_model.get_weights().shape[1]

    plt.figure(figsize=(map_y_dim + 2, map_x_dim)) # Adjust size for legend
    
    # Background with U-Matrix
    plt.pcolor(som_model.distance_map().T, cmap='bone_r', alpha=0.6)
    
    # Colors and markers for classes (malignant, benign)
    # target_names_list[0] is 'malignant' (label 0), target_names_list[1] is 'benign' (label 1)
    colors = {0: '#FF5733', 1: '#33C4FF'} # Red for malignant, Blue for benign
    markers = {0: 'x', 1: 'o'}
    
    # Map each sample to its BMU and place a marker
    for i, sample in enumerate(data):
        bmu_coord = som_model.winner(sample) # (x_col, y_row)
        label_class = labels_np[i]
        
        # Add small jitter so points don't overlap exactly
        jitter_x = (np.random.rand() - 0.5) * 0.4
        jitter_y = (np.random.rand() - 0.5) * 0.4
        
        plt.plot(bmu_coord[0] + 0.5 + jitter_x, 
                 bmu_coord[1] + 0.5 + jitter_y,
                 markers[label_class],
                 markerfacecolor='None', # Only marker border
                 markeredgecolor=colors[label_class],
                 markersize=10, 
                 markeredgewidth=1.5,
                 alpha=0.7)

    plt.title('SOM Map with Superimposed Diagnosis')
    plt.xticks(np.arange(map_x_dim) + 0.5, np.arange(map_x_dim))
    plt.yticks(np.arange(map_y_dim) + 0.5, np.arange(map_y_dim))
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
    plt.gca().invert_yaxis()
    
    # Create legend manually
    handles = [plt.Line2D([0], [0], marker=markers[i], color='w', 
                          markerfacecolor='None', markeredgecolor=colors[i], 
                          markersize=10, markeredgewidth=1.5) for i in range(len(target_names_list))]
    plt.legend(handles, target_names_list, title='Diagnosis', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.show()

#%% [markdown]
# ### Function Description: `visualize_som_component_map`
# (Adapted to handle 30 component planes)
#%%
def visualize_som_component_map(som_model, feature_names_list, max_planes_to_show=None):
    if not MINISOM_IMPORTED_SUCCESSFULLY or som_model is None:
        module_logger.error("MiniSom not available or model not trained. Cannot visualize component planes.")
        return

    module_logger.info("Visualizing SOM Component Planes...")
    weights = som_model.get_weights() 
    num_features = weights.shape[2]
    map_x_dim = weights.shape[0]
    map_y_dim = weights.shape[1]

    if feature_names_list is None or len(feature_names_list) != num_features:
        feature_names_list = [f'Feature {i+1}' for i in range(num_features)]

    num_planes_actual = num_features
    if max_planes_to_show is not None and num_features > max_planes_to_show:
        num_planes_actual = max_planes_to_show
        module_logger.info(f"Showing first {max_planes_to_show} of {num_features} component planes.")
    
    cols = 5 # Adjust number of columns for 30 features
    rows = (num_planes_actual + cols - 1) // cols 
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3)) # Adjust size
    axes = axes.flatten() 

    for i in range(num_planes_actual):
        ax = axes[i]
        component_plane = weights[:, :, i]
        im = ax.pcolor(component_plane.T, cmap='viridis') 
        ax.set_title(f"{feature_names_list[i]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label='Weight')

    for j in range(num_planes_actual, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(f"SOM Component Planes (First {num_planes_actual} of {num_features})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()


#%% [markdown]
# ## 3. Exercise Development: SOM with Breast Cancer Dataset

#%% [markdown]
# ### 3.1. Data Loading and Preparation
#
# We load the `Breast Cancer Wisconsin` dataset and scale its 30 features.

#%%
X_cancer_scaled, y_cancer_original, scaler_cancer, cancer_feature_names, cancer_target_names = \
    load_and_prepare_cancer_data_som()

#%% [markdown]
# ### 3.2. SOM Initialization and Training
#
# For the Breast Cancer dataset (569 samples, 30 features).
# Heuristic: `5 * sqrt(569) approx 5 * 23.8 = 119 neurons`. A 10x12 or 11x11 map.

#%%
som_model_cancer = None
if MINISOM_IMPORTED_SUCCESSFULLY:
    N_SAMPLES_CANCER = X_cancer_scaled.shape[0]
    num_neurons_heuristic_cancer = int(5 * np.sqrt(N_SAMPLES_CANCER))
    MAP_X_DIM_CANCER = int(np.sqrt(num_neurons_heuristic_cancer))
    MAP_Y_DIM_CANCER = (num_neurons_heuristic_cancer // MAP_X_DIM_CANCER) + ((num_neurons_heuristic_cancer % MAP_X_DIM_CANCER) > 0)
    
    module_logger.info(f"SOM size heuristic (Cancer): ~{num_neurons_heuristic_cancer} neurons. Using {MAP_X_DIM_CANCER}x{MAP_Y_DIM_CANCER} map.")
    # You could fix a size if you prefer, e.g.: MAP_X_DIM_CANCER = 10; MAP_Y_DIM_CANCER = 12

    INPUT_LEN_CANCER = X_cancer_scaled.shape[1] # 30 features
    
    SIGMA_INIT_CANCER = 1.8 
    LEARNING_RATE_INIT_CANCER = 0.5
    NUM_ITERATIONS_SOM_CANCER = 20000 # More iterations for more complex data

    som_model_cancer = initialize_and_train_som(
        X_cancer_scaled, MAP_X_DIM_CANCER, MAP_Y_DIM_CANCER, INPUT_LEN_CANCER,
        sigma=SIGMA_INIT_CANCER, learning_rate=LEARNING_RATE_INIT_CANCER,
        num_iterations=NUM_ITERATIONS_SOM_CANCER, random_seed=SEED
    )
else:
    module_logger.warning("Skipping SOM initialization and training because MiniSom was not imported.")

#%% [markdown]
# ### 3.3. SOM Results Visualization

#%% [markdown]
# #### 3.3.1. Distance Map (U-Matrix)
# The U-Matrix for the cancer dataset.

#%%
if som_model_cancer:
    visualize_som_distance_map(som_model_cancer)

#%% [markdown]
# #### 3.3.2. Activation Map
# Frequency of neuron activation for cancer samples.

#%%
if som_model_cancer:
    visualize_som_activation_map(som_model_cancer, X_cancer_scaled)

#%% [markdown]
# #### 3.3.3. SOM Map with Diagnosis Labels
# We superimpose 'malignant' and 'benign' labels on the map.

#%%
if som_model_cancer:
    visualize_som_map_with_labels(som_model_cancer, X_cancer_scaled, y_cancer_original, cancer_target_names)

#%% [markdown]
# #### 3.3.4. Component Planes
# We visualize the planes of the 30 features.

#%%
if som_model_cancer:
    visualize_som_component_map(som_model_cancer, cancer_feature_names, max_planes_to_show=30) # Show all


#%% [markdown]
# ## 4. Exercise Conclusions (SOM with Breast Cancer)
#
# **Summary of Findings:**
# * The `Breast Cancer Wisconsin` dataset was loaded and preprocessed, scaling its 30 features.
# * A Self-Organizing Map (SOM) of **[MAP_X_DIM_CANCER x MAP_Y_DIM_CANCER]** neurons was trained for **[NUM_ITERATIONS_SOM_CANCER]** iterations.
# * **Distance Map (U-Matrix):** The U-Matrix showed **[Describe: e.g., whether one or two main clear regions separated by borders are observed, or a more gradual structure. Does it suggest two main clusters?]**.
# * **Activation Map:** This map indicated **[Describe the activation distribution]**.
# * **SOM Map with Diagnosis Labels:** When superimposing 'malignant' and 'benign' labels, it was observed that **[Describe: e.g., malignant and benign samples tended to cluster in predominantly distinct regions of the SOM map. How clear was the separation? Was there a transition or mixing zone?]**.
# * **Component Planes:** Analysis of the 30 component planes revealed that **[Describe key findings. Which features (e.g.: 'mean radius', 'worst concave points') showed clear patterns or gradients across the map that correlate with malignant/benign regions? Were there features that appeared uniform and therefore less discriminative in the SOM map?]**.
#
# **About SOMs with Tabular Data like Breast Cancer:**
# * SOMs can help visualize the separation structure (or lack thereof) between classes in a high-dimensional feature space.
# * The U-Matrix and label map combined can give a strong indication of the "difficulty" of separating classes.
# * Component planes can offer insights into which features are most important for the topological organization of the map and, therefore, for the distinction between the groups that the SOM identifies.
#
# **General Learning:**
# This exercise demonstrated the use of SOMs for unsupervised exploration of the Breast Cancer dataset. Even without knowing the diagnosis labels during training, the SOM was able to organize the data in a 2D map such that underlying classes often form cohesive regions. This underscores the utility of SOMs for data visualization, structure detection, and potential identification of relevant features in complex tabular datasets.
#
# *(Note: Qualitative descriptions and specific findings in brackets should be completed after fully executing the notebook and analyzing the generated graphs.)*