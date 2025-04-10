import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
import json
import yaml
import numpy as np

from tqdm import tqdm
from hdbscan import validity
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from embedding_generator import get_local_embed_path
from clustering import cluster_embeddings, reduce_embedding_dim, generate_hdbscan_configs
from utils.cluster_utils import visualize_hyperparam_search_results, select_best_config, validate_clusters

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
    # Always reduce dimensions for fair comparison
    reduced_emb = reduce_embedding_dim(embeddings, config["n_components"]) if config['reduce_embed_dim'] else embeddings
    
    labels, model = cluster_embeddings(reduced_emb, config)
    metrics = validate_clusters(reduced_emb, labels, model)
    
    return {
        **config,
        **metrics,
        'labels': labels
    }

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
    
    results = Parallel(n_jobs=-1)(
        delayed(run_single_config)(embeddings, config)
        for config in tqdm(configs, desc="Tuning HDBSCAN")
    )

    best = select_best_config(
        results,
        max_noise=0.4, 
        min_clusters=None 
    )

    visualize_hyperparam_search_results(results, embeddings.shape[0], best)

    print("Best result is ", best)
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

    # Normalize the embeddings
    embeddings = StandardScaler().fit_transform(embeddings)
    
    best_config = tune_hdbscan(embeddings)
