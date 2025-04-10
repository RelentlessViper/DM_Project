import umap
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.metrics import silhouette_score

def visualize_clusters(embeddings: np.ndarray, cluster_labels, config):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    df_plot = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "cluster": cluster_labels
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="cluster",
        title="Interactive Cluster Visualization",
        labels={"cluster": "Cluster"},
        hover_data=["cluster"],  
        color_continuous_scale=px.colors.sequential.Plasma 
    )

    fig.write_html(f"{config['output_dir']}/interactive_plot_{config['clustering_algorithm']}_{config['sample_size']}.html")
    fig.show()


def visualize_hyperparam_search_results(results, sample_size: int, best_config):
    df = pd.DataFrame(results)
    
    df['config_id'] = df.apply(lambda row: (
        f"Sil={row['silhouette']:.2f}_"
        f"DBCV={row['dbcv']:.2f}_"
        f"Noise={row['noise_ratio']:.2f}_"
        f"min_clust={row['min_cluster_size']}_"
        f"min_samp={row['min_samples']}_"
        f"eps={row.get('cluster_selection_epsilon', 0):.1f}"
    ), axis=1)

    # Plot 1: Silhouette vs DBCV
    fig1 = px.scatter(
        df,
        x='silhouette',
        y='dbcv',
        color='n_clusters',
        hover_name='config_id',
        title='Silhouette Score vs DBCV',
        labels={'silhouette': 'Silhouette Score', 'dbcv': 'DBCV Score'},
        size_max=15
    )
    if best_config:
        fig1.add_annotation(
            x=best_config['silhouette'],
            y=best_config['dbcv'],
            text="Best Config",
            showarrow=True,
            arrowhead=1
        )
    fig1.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    fig1.write_html(f"silhouette_vs_dbcv_{sample_size}.html")
    
    # Plot 2: Silhouette vs Noise Ratio
    fig2 = px.scatter(
        df,
        x='silhouette',
        y='noise_ratio',
        color='n_clusters',
        hover_name='config_id',
        title='Silhouette Score vs Noise Ratio',
        labels={'silhouette': 'Silhouette Score', 'noise_ratio': 'Noise Ratio'},
        size_max=15
    )
    if best_config:
        fig2.add_annotation(
            x=best_config['silhouette'],
            y=best_config['noise_ratio'],
            text="Best Config",
            showarrow=True,
            arrowhead=1
        )
    fig2.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    fig2.write_html(f"silhouette_vs_noise_{sample_size}.html")


def select_best_config(results, max_noise=0.3, min_clusters=None):
    """
    Selects best config based on:
    - noise_ratio < max_noise
    - n_clusters > min_clusters (auto-calculated if None)
    - maximum silhouette score among remaining
    
    Args:
        results: List of configuration results from tuning
        max_noise: Maximum allowed noise ratio (0-1)
        min_clusters: Minimum cluster threshold. If None, auto-calculates.
    """
    df = pd.DataFrame(results)
    
    # Auto-calculate min_clusters if not provided
    if min_clusters is None:
        min_clusters = calculate_min_cluster_threshold(df)
    
    # Filter valid configurations
    valid_configs = df[
        (df['noise_ratio'] < max_noise) &
        (df['n_clusters'] >= min_clusters)
    ]
    
    if len(valid_configs) == 0:
        print("Warning: No configurations met criteria!")
        print("Relaxing noise constraint...")
        valid_configs = df[df['n_clusters'] >= min_clusters]
        
    # Select config with highest silhouette score
    best_idx = valid_configs['silhouette'].idxmax()
    return valid_configs.loc[best_idx].to_dict()

def calculate_min_cluster_threshold(df):
    """
    Heuristic to determine minimum cluster threshold based on:
    - Number of data points
    - Distribution of cluster counts in results
    
    Args:
        df: Results DataFrame
        min_points_per_cluster: Minimum points per cluster to consider valid
    """
    # Get approximate data size from first config's labels
    data_size = len(df.iloc[0]['labels'])
    
    # Logarithmic rule
    log_rule = int(np.log(data_size)**2)  # ln(1000)^2 ≈ 49
    
    # Sturges' formula (for normally distributed data)
    sturges = int(1 + 3.322 * np.log10(data_size))  # ≈ 11 for 1000 points
    
    return min(sturges, log_rule)


def validate_clusters(embeddings, labels, model):
    """Calculate DBCV and silhouette score"""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    
    if len(np.unique(labels)) > 1:  # Silhouette requires >1 cluster
        sil_score = silhouette_score(embeddings, labels)
    else:
        sil_score = -1  # Invalid
        
    # DBCV (native to HDBSCAN)
    if hasattr(model, 'relative_validity_'):
        # print(model.relative_validity_)
        dbcv_score = model.relative_validity_
    else:
        dbcv_score = -2
        
    return {
        'silhouette': sil_score,
        'dbcv': dbcv_score,
        'n_clusters': len(np.unique(labels)),
        'noise_ratio': (labels == -1).mean()
    }