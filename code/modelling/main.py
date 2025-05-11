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
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel

from labelling import ClusterLabelGenerator
from clustering import cluster_embeddings, calculate_cluster_sizes, calculate_temporal_trends, analyze_trends
from embedding_generator import generate_or_load_embeddings
from utils.loader_utils import load_data, load_model_tokenizer, load_best_hdbscan_config
from utils.cluster_utils import visualize_clusters, plot_temporal_trends
        
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
    if config["sample"]:
        df = load_data(config["sample_file_path"])
    else:
        df = load_data(config["file_path"])

    print("Loading model and tokenizer")
    model, tokenizer = load_model_tokenizer(config["model_name"], device=config["device"])

    assert model is not None

    embeddings = generate_or_load_embeddings(model, tokenizer, df['text'].tolist(), config)

    print(f"Embedding shape is {embeddings.shape}")

    if embeddings.shape[0] > 100000:
        config['hdbscan_use_approximate_predict'] = True
    
    cluster_labels, cluster_model = cluster_embeddings(embeddings, config, verbose=True, sample_percentage=0.1)
    cluster_labels = cluster_labels.tolist()
    cluster_sizes = calculate_cluster_sizes(cluster_labels, config)

    label_generator = ClusterLabelGenerator(
        port=30000, 
        host= "10.100.30.241"
    )
    
    # Generate cluster labels
    cluster_text_labels = label_generator.generate_cluster_labels(
        embeddings=embeddings,
        texts=df['text'].tolist(),
        cluster_labels=cluster_labels,
        max_texts_per_cluster=40, 
        max_label_length=1024 
    )
    
    recent_years = 5
    trends_pivot = calculate_temporal_trends(df, cluster_labels, "year")
    analysis = analyze_trends(trends_pivot, recent_years=recent_years, bottom_x=0.01, top_y=0.01)
    plot_temporal_trends(trends_pivot, analysis, cluster_text_labels, recent_years=recent_years)

    print("Visualizing clusters...")
    visualize_clusters(embeddings, cluster_labels, cluster_text_labels, config)

if __name__ == "__main__":
    main()