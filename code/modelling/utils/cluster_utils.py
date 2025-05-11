import os
import umap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import List
from sklearn.metrics import silhouette_score

def visualize_clusters(embeddings: np.ndarray, cluster_labels, cluster_text_labels, config):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Map numeric cluster labels to text labels
    text_labels = [cluster_text_labels.get(label, str(label)) for label in cluster_labels]
    
    df_plot = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "cluster": cluster_labels,
        "cluster_text": text_labels
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="cluster",
        title="Interactive Cluster Visualization",
        labels={"cluster": "Cluster"},
        hover_data=["cluster_text"],  # Show text labels on hover
        color_continuous_scale=px.colors.sequential.Plasma 
    )
    
    # Update hover template to show text labels more prominently
    fig.update_traces(
        hovertemplate="<b>Cluster Text:</b> %{customdata[0]}<br>" +
                     "<b>X:</b> %{x}<br>" +
                     "<b>Y:</b> %{y}<extra></extra>"
    )

    fig.write_html(f"{config['output_dir']}/interactive_plot_{config['clustering_algorithm']}_{config['sample_size']}.html")

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
    fig1.write_html(f"outputs/silhouette_vs_dbcv_{sample_size}.html")
    
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
    fig2.write_html(f"outputs/silhouette_vs_noise_{sample_size}.html")


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
    labels = np.asarray(labels, dtype=np.float64)
    
    if len(np.unique(labels)) > 1:  # Silhouette requires >1 cluster
        sil_score = silhouette_score(embeddings, labels)
    else:
        sil_score = -1  # Invalid
        
    # DBCV (native to HDBSCAN)
    if hasattr(model, 'relative_validity_'):
        dbcv_score = model.relative_validity_
    else:
        dbcv_score = -2
        
    return {
        'silhouette': sil_score,
        'dbcv': dbcv_score,
        'n_clusters': len(np.unique(labels)),
        'noise_ratio': (labels == -1).mean()
    }

def assign_text_labels_to_df(df: pd.DataFrame, cluster_labels: np.ndarray, cluster_text_labels: dict) -> pd.DataFrame:
    """
    Assign text labels to the dataframe based on cluster assignments.
    
    Args:
        df: pd.DataFrame, original dataframe with 'text' column
        cluster_labels: np.ndarray, cluster labels from HDBSCAN
        cluster_text_labels: dict, mapping of cluster ID to text label
    
    Returns:
        pd.DataFrame: Updated dataframe with 'cluster_label' column
    """
    df = df.copy()
    df['cluster_label'] = [cluster_text_labels.get(label, f"Cluster {label}") for label in cluster_labels]
    return df


def plot_temporal_trends(trends_pivot, analysis, cluster_text_labels, output_dir: str ='outputs', recent_years: int = 5):
    """
    Plot temporal trends using Plotly with cluster text labels, saving as PNG and HTML.
    
    Args:
        trends_pivot (pd.DataFrame): Pivot table from calculate_temporal_trends.
        analysis (pd.DataFrame): Analysis from analyze_trends.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig = go.Figure()
    
    for cluster in trends_pivot.columns:
        cluster_id = int(cluster) if isinstance(cluster, (int, float)) else cluster
        label = cluster_text_labels.get(cluster_id, f'Cluster {cluster}')
        # short_label = shorten_label(label)
        
        # Determine color based on cluster status
        if cluster in analysis[analysis['Underrepresented']]['Cluster'].values:
            color = 'red'
        elif cluster in analysis[analysis['Slowing']]['Cluster'].values:
            color = 'blue'
        elif cluster in analysis[analysis['Leading']]['Cluster'].values:
            color = 'green'
        else:
            color = 'grey'
        
        # Get analysis metrics for tooltip
        cluster_analysis = analysis[analysis['Cluster'] == cluster]
        total_papers = cluster_analysis['Total_Papers'].iloc[0] if not cluster_analysis.empty else 0
        growth_rate = cluster_analysis['Growth_Rate'].iloc[0] if not cluster_analysis.empty else 0
        recent_papers = cluster_analysis['Recent_Papers'].iloc[0] if not cluster_analysis.empty else 0
        mean_annual_papers = cluster_analysis['Mean_Annual_Papers'].iloc[0] if not cluster_analysis.empty else 0
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=trends_pivot.index,
            y=trends_pivot[cluster],
            name=label,
            line=dict(color=color),
            hovertemplate=(
                f'<b>{label}</b><br>' +
                'Year: %{x}<br>' +
                'Papers: %{y}<br>' +
                f'Total Papers: {total_papers:.0f}<br>' +
                f'Growth Rate: {growth_rate:.3f}<br>' +
                f'Recent Papers ({recent_years}y): {recent_papers:.0f}<br>' +
                f'Mean Annual Papers: {mean_annual_papers:.0f}<br>'
            )
        ))
    
    # Update layout
    fig.update_layout(
        title='Temporal Trends of Paper Counts by Research Topic',
        xaxis_title='Year',
        yaxis_title='Number of Papers',
        legend=dict(
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top',
            font=dict(size=10)
        ),
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )
    
    # Save HTML
    html_path = os.path.join(output_dir, 'temporal_trends.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"Interactive trends plot saved to {html_path}")