import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterSampler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value .* overridden to 1 by setting random_state.")

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
               metric='cosine',
               random_state=42).fit_transform(embeddings)
    
def cluster_embeddings(embeddings: np.ndarray, config, flag: bool = True):
    if config["reduce_embed_dim"] and flag:
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


# ======================= CLUSTER ANALYSIS =====================================
def calculate_cluster_sizes(cluster_assignments):
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


def analyze_trends(trends_pivot, recent_years=5, low_count_threshold=0.03, min_papers=5):
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
    
    # Flag underrepresented (low total papers) and slowing (high early, low recent)
    analysis['Underrepresented'] = (analysis['Total_Papers'] < low_count_absolute) & (analysis['Mean_Annual_Papers'] < min_papers)
    analysis['Slowing'] = (analysis['Total_Papers'] > low_count_absolute) & (analysis['Recent_Papers'] < analysis['Total_Papers'] * 0.2)
    analysis['Leading'] = (analysis['Total_Papers'] > low_count_absolute) & (analysis['Growth_Rate'] > 0.05)
    
    return analysis.reset_index()