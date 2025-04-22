import umap
import yaml
import json
import torch
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel

from clustering import cluster_embeddings
from embedding_generator import generate_or_load_embeddings
from utils.loader_utils import load_data, load_model_tokenizer, load_best_hdbscan_config
from utils.cluster_utils import visualize_clusters
        
def main(config_path: str ="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    hdbscan_config = load_best_hdbscan_config("")     
    config = {
        **config,  # Base config
        **{k: v for k, v in hdbscan_config.items() 
           if k not in ['silhouette', 'dbcv', 'labels']}
    }
    
    print("Loaded optimized HDBSCAN configuration")
        
    print("Loading data...")
    df = load_data(config["file_path"], sample_size=config["sample_size"])

    print("Loading model and tokenizer")
    model, tokenizer = load_model_tokenizer(config["model_name"], device=config["device"])

    assert model is not None
    assert tokenizer is not None

    embeddings = generate_or_load_embeddings(model, tokenizer, df['text'].tolist(), config)
    embeddings = StandardScaler().fit_transform(embeddings)
    
    cluster_labels, cluster_model = cluster_embeddings(embeddings, config)

    print(cluster_labels)

    print("Visualizing clusters...")
    visualize_clusters(embeddings, cluster_labels, config)

    

if __name__ == "__main__":
    # args = parse_args()
    main()