import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterSampler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def get_hdbscan_param_grid(embeddings_shape):
    max_components = min(200, embeddings_shape[0] - 1)  # Ensure n < sample count
    base_grid = {
        'min_cluster_size': [10, 15, 20, 25, 50, 100, 150, 200, 250, 300],
        'min_samples': [1, 2, 3, 5],
        'cluster_selection_method': ['eom', 'leaf'],
        'alpha': [0.5, 1.0, 1.5],
        'cluster_selection_epsilon': [0.0, 0.1, 0.2],
    }
    
    # Dynamic component options based on data size
    component_options = [2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400]
    component_options = [x for x in component_options if x < max_components]
    if max_components >= 100:  # Only add higher dims for large datasets
        component_options.extend([100, 150, 200])
    
    base_grid['n_components'] = component_options
    return base_grid

def generate_hdbscan_configs(embeddings_shape):
    return list(ParameterSampler(
        get_hdbscan_param_grid(embeddings_shape), 
        n_iter=80000,
        random_state=42
    ))

def reduce_embedding_dim(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    return UMAP(n_components=n_components, 
               metric='cosine').fit_transform(embeddings)
    
def cluster_embeddings(embeddings: np.ndarray, config):
    if config["reduce_embed_dim"]:
        embeddings = reduce_embedding_dim(embeddings, config["n_components"])
        
    algo = config['clustering_algorithm']
    
    if algo == 'kmeans':
        model = KMeans(
            n_clusters=config['n_clusters'], 
            random_state=42, 
            n_init=10
        )
    elif algo == 'dbscan':
        model = DBSCAN(eps=config['eps'], min_samples=config['min_samples'])
    elif algo == "hdbscan":
        model = HDBSCAN(
            min_cluster_size=config.get('min_cluster_size', 50),
            min_samples=config.get('min_samples', 15),
            cluster_selection_method=config.get('cluster_selection_method', 'eom'),
            metric=config.get('metric', 'cosine'),
            allow_single_cluster=config.get('allow_single_cluster', True),
            **{k: v for k, v in config.items() 
               if k in ['alpha', 'cluster_selection_epsilon']}
        )
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")

    labels = model.fit_predict(embeddings)
    
    # Save cluster info
    output_dir = config['output_dir'] + "/clusters"
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/{algo}_labels.npy", labels)
    return labels, model