import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JOBLIB_START_METHOD'] = 'spawn' 

import json
import yaml
import numpy as np
import multiprocessing as mp

from hdbscan import validity
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from embedding_generator import get_local_embed_path
from clustering import cluster_embeddings, reduce_embedding_dim, generate_hdbscan_configs
from utils.cluster_utils import visualize_hyperparam_search_results, select_best_config, validate_clusters

import warnings
warnings.simplefilter("always")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib.externals.loky")

# Set start method at the top of the script
mp.set_start_method('spawn', force=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def run_single_config(embeddings, config):
    try:
        umap_n_components = config.get("n_components", 50) 
        umap_n_jobs = config.get("umap_n_jobs", -1)
        umap_low_memory = config.get("umap_low_memory", False) 
        umap_metric = config.get("umap_metric", "euclidean") 
        umap_random_state = config.get("umap_random_state", 42)
        umap_n_neighbors = config.get("umap_n_neighbors", 15) 

        reduced_emb = reduce_embedding_dim(
            embeddings,
            n_components=umap_n_components,
            umap_n_jobs=umap_n_jobs,
            umap_low_memory=umap_low_memory,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
            umap_n_neighbors=umap_n_neighbors,
        )
        
        labels, model = cluster_embeddings(reduced_emb, config, False)
        metrics = validate_clusters(reduced_emb, labels, model)
        
        return {
            **config,
            **metrics,
            'labels': labels
        }
    except Exception as e:
        print(f"Error in config {config}: {str(e)}")
        return {}  # Return empty result to avoid crashing the pool

def tune_hdbscan(embeddings, n_iter=20):
    base_config = {
        'clustering_algorithm': 'hdbscan',
        'reduce_embed_dim': True,
        'output_dir': 'outputs',
        'metric': 'euclidean',
    }
    
    configs = [
        {**base_config, **params} 
        for params in generate_hdbscan_configs(embeddings.shape)
    ]

    print(f"There are {len(configs)} configs to try")
    
    with Parallel(n_jobs=-1, backend='loky', verbose=10) as parallel:  
        results = parallel(
            delayed(run_single_config)(embeddings, config)
            for config in configs
        )

    best = select_best_config(
        results,
        max_noise=0.5, 
        min_clusters=None 
    )

    visualize_hyperparam_search_results(results, embeddings.shape[0], best)

    with open('best_hdbscan.json', 'w') as f:
        json.dump(best, f, cls=NumpyEncoder)
    
    return best

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    embed_dir, embed_path = get_local_embed_path(config)
    
    assert os.path.exists(embed_path), "If loading embeddings from local dir, embeddings should exist in a local dir"
    print("Loading embeddings from", embed_path)
    embeddings = np.load(embed_path)
    
    best_config = tune_hdbscan(embeddings)
    print(best_config)