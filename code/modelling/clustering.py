import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import hdbscan
import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from utils.cluster_utils import validate_clusters

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value .* overridden to 1 by setting random_state.")

def get_hdbscan_param_grid(embeddings_shape):
    base_grid = {
        'min_cluster_size': [20, 25, 50, 100],
        'min_samples': [3, 5, 7, 10, 15, 20],
        'cluster_selection_method': ['eom'],
        'hdbscan_use_approximate_predict': [False],
        'alpha': [1.0, 1.5],
        'cluster_selection_epsilon': [0.0],
        'n_components': [2, 5, 10, 20, 30, 40], 
        'umap_n_neighbors': [5, 10, 15, 20]
    }
    
    return base_grid

def generate_hdbscan_configs(embeddings_shape):
    return list(ParameterGrid(get_hdbscan_param_grid(embeddings_shape)))

def reduce_embedding_dim(embeddings: np.ndarray, 
                         n_components: int, 
                         umap_n_jobs: int = -1, 
                         umap_low_memory: bool = False, 
                         umap_random_state: int = 42,
                         umap_metric: str = 'cosine',
                         umap_n_neighbors: int = 15,
                        ) -> np.ndarray:
    """
    Reduces embedding dimensions using UMAP.
    """
    reducer = UMAP(
        n_components=n_components,
        metric=umap_metric,
        random_state=umap_random_state,
        n_neighbors=umap_n_neighbors,      
        n_jobs=umap_n_jobs,         
        low_memory=umap_low_memory  
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings
    
def cluster_embeddings(embeddings: np.ndarray, config, flag: bool = True, verbose: bool = False, sample_percentage: float = 0.5):
    """
    Performs clustering on embeddings, with UMAP dimensionality reduction
    and optional HDBSCAN approximate prediction.
    """
    
    # --- Dimensionality Reduction (UMAP) ---
    if config["reduce_embed_dim"] and flag:
        if verbose:
            print(f"Original embedding shape: {embeddings.shape}")
        
        umap_n_components = config.get("n_components", 50) 
        umap_n_jobs = config.get("umap_n_jobs", 60)
        umap_low_memory = config.get("umap_low_memory", False) 
        umap_metric = config.get("umap_metric", "euclidean") 
        umap_random_state = config.get("umap_random_state", 42)
        umap_n_neighbors = config.get("umap_n_neighbors", 15)     

        embeddings_for_clustering = reduce_embedding_dim(
            embeddings,
            n_components=umap_n_components,
            umap_n_jobs=umap_n_jobs,
            umap_low_memory=umap_low_memory,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
            umap_n_neighbors=umap_n_neighbors,
        )
        if verbose:
            print(f"Reduced embedding shape for clustering: {embeddings_for_clustering.shape}")
    else:
        embeddings_for_clustering = embeddings 
        if verbose:
            print(f"Using original embeddings for clustering. Shape: {embeddings_for_clustering.shape}")
    
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
        hdbscan_base_params = {
            'min_cluster_size': config.get('min_cluster_size', 15), 
            'min_samples': config.get('min_samples', 15),      
            'cluster_selection_method': config.get('cluster_selection_method', 'eom'),
            'metric': config.get('metric', 'euclidean'),      
            'prediction_data': True,
            'allow_single_cluster': config.get('allow_single_cluster', True),
            'approx_min_span_tree': config.get('approx_min_span_tree', True),
            'gen_min_span_tree': config.get('gen_min_span_tree', False), 
            'core_dist_n_jobs': config.get('core_dist_n_jobs', -1),    
            'alpha': config.get('alpha', 1.0), # Default for HDBSCAN*
            'cluster_selection_epsilon': config.get('cluster_selection_epsilon', 0.0)
        }
        
        hdbscan_params = {k: v for k, v in hdbscan_base_params.items() if v is not None}
        
        for key, value in config.items():
            if key.startswith("hdbscan_opt_") and key[12:] not in hdbscan_params:
                hdbscan_params[key[12:]] = value
            elif key in HDBSCAN().get_params().keys() and key not in hdbscan_params:
                 if key not in ["min_cluster_size", "min_samples", "metric", "alpha",
                                "cluster_selection_method", "allow_single_cluster", 
                                "approx_min_span_tree", "gen_min_span_tree", "core_dist_n_jobs",
                                "cluster_selection_epsilon"]:
                    hdbscan_params[key] = value

        current_hdbscan_model = HDBSCAN(**hdbscan_params)
        
        use_approximate = config.get('hdbscan_use_approximate_predict', False)
        sample_size = int(embeddings_for_clustering.shape[0]*sample_percentage)

        if use_approximate:
            if sample_size <= 0:
                raise ValueError("hdbscan_approximate_predict_sample_size must be positive.")
            actual_sample_size = min(sample_size, embeddings_for_clustering.shape[0])

            if verbose:
                print(f"Fetching {actual_sample_size} random samples")

            sample_indices = np.random.choice(embeddings_for_clustering.shape[0], actual_sample_size, replace=False)
            sampled_embeddings = embeddings_for_clustering[sample_indices]

            if verbose:
                print("Fitting model to sampled embeddings")
            
            current_hdbscan_model.fit(sampled_embeddings)
            model = current_hdbscan_model 

            if verbose:
                print("Predicting labels for the full dataset using approximate_predict...")
            labels_tuple = hdbscan.approximate_predict(model, embeddings_for_clustering)
            
            if isinstance(labels_tuple, tuple) and len(labels_tuple) == 2:
                 labels = labels_tuple[0] 
            else:
                 # This case should ideally not happen if approximate_predict succeeds
                if verbose:
                     print("Warning: approximate_predict did not return the expected (labels, probabilities) tuple.")
                labels = labels_tuple

            # print metrics
            metrics = validate_clusters(embeddings_for_clustering, labels, model)
            if verbose:
                print(metrics)
            

        else:
            model = current_hdbscan_model
            labels = model.fit_predict(embeddings_for_clustering)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")

    if verbose:
        print("Model fitting/prediction complete.")
    
    # Save cluster info
    output_dir = config['output_dir'] + "/clusters"
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/{algo}_labels.npy", labels)
    return labels, model


# ======================= CLUSTER ANALYSIS =====================================
def calculate_cluster_sizes(cluster_assignments, config):
    """
    Calculate the size of each cluster based on assignments.
    
    Args:
        cluster_assignments (list): List of cluster IDs (e.g., [4, 2, -1, ...]).
        
    Returns:
        pd.DataFrame: DataFrame with columns 'Cluster' and 'Size', sorted by Size.
    """
    # Convert assignments to a Series and count occurrences
    cluster_counts = pd.Series(cluster_assignments).value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Size']
    # Sort by size in descending order
    cluster_counts = cluster_counts.sort_values(by='Size', ascending=False)
    cluster_counts.to_csv(f"outputs/cluster_sizes_{config["sample_size"]}.csv")
    return cluster_counts

def calculate_temporal_trends(df, cluster_assignments, time_period='year'):
    """
    Calculate temporal trends of paper counts per cluster over time.
    
    Args:
        df (pd.DataFrame): Dataset with 'Update Date' column.
        cluster_assignments (list): List of cluster IDs.
        time_period (str): Aggregation period ('year' or 'month').
        
    Returns:
        pd.DataFrame: Pivot table with clusters as columns, time periods as rows, and paper counts as values.
    """
    # Add cluster assignments to the DataFrame
    df = df.copy()  # Avoid modifying the original
    df['Cluster'] = cluster_assignments
    
    # Extract time period
    if time_period == 'year':
        df['Time_Period'] = df['update_date'].dt.year
    elif time_period == 'month':
        df['Time_Period'] = df['update_date'].dt.to_period('M').dt.to_timestamp()
    else:
        raise ValueError("time_period must be 'year' or 'month'")
    
    # Group by Time_Period and Cluster, count papers
    trends = df.groupby(['Time_Period', 'Cluster']).size().reset_index(name='Count')
    
    # Pivot to get clusters as columns
    trends_pivot = trends.pivot(index='Time_Period', columns='Cluster', values='Count').fillna(0)
    
    return trends_pivot


def analyze_trends(trends_pivot, recent_years=5, low_count_threshold=0.035, min_papers=3, bottom_x=0.02, top_y=0.02):
    """
    Analyze trends to identify underrepresented, slowing, and leading clusters.
    
    Args:
        trends_pivot (pd.DataFrame): Pivot table from calculate_temporal_trends.
        recent_years (int): Number of recent years to consider for activity.
        low_count_threshold (float): Fraction of total papers for underrepresentation.
        min_papers (int): Minimum papers per year to avoid noise.
        
    Returns:
        pd.DataFrame: Analysis with total papers, growth rate, recent activity, and gap flags.
    """
    total_papers = trends_pivot.sum().sum()
    low_count_absolute = total_papers * low_count_threshold
    
    analysis = pd.DataFrame({
        'Total_Papers': trends_pivot.sum(),
        'Mean_Annual_Papers': trends_pivot.mean(),
        'Recent_Papers': trends_pivot.tail(recent_years).sum()
    })
    
    # Calculate growth rate (average annual change, normalized by mean papers)
    growth_rates = []
    for cluster in trends_pivot.columns:
        counts = trends_pivot[cluster]
        if counts.mean() > 0:
            annual_change = counts.diff().mean() / counts.mean()
        else:
            annual_change = 0
        growth_rates.append(annual_change)
    analysis['Growth_Rate'] = growth_rates

    # Flag underrepresented (bottom x% by growth rate AND low mean annual papers)
    sorted_growth = analysis['Growth_Rate'].sort_values()
    bottom_clusters = sorted_growth.head(int(len(sorted_growth) * bottom_x)).index
    analysis['Underrepresented'] = (analysis.index.isin(bottom_clusters))
    top_clusters = analysis['Growth_Rate'].sort_values(ascending=False).head(int(len(sorted_growth) * top_y)).index
    analysis['Leading'] = analysis.index.isin(top_clusters)
    analysis['Slowing'] = (analysis['Total_Papers'] > low_count_absolute) & (analysis['Recent_Papers'] < analysis['Total_Papers'] * 0.2)    
    
    return analysis.reset_index()