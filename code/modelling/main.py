import umap
import torch
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import BertTokenizer, BertModel

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def load_data(file_path, sample_size=None):
    """Load the dataset. To load a small sample fo the dataset, set the sample size"""
    df = pd.read_csv(file_path)
    df['text'] = df['title'] + " " + df['abstract']
    
    if sample_size:
        df = df.sample(sample_size, random_state=42)
        
    return df


def get_embeddings(texts, batch_size=32):
    """Batch the input (text + abstract) and generate the embeddings"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval() 

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                          truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


def cluster_embeddings(embeddings, n_clusters: int):
    """Cluster embeddings"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    return cluster_labels, kmeans

def visualize_clusters(embeddings, cluster_labels):
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

    fig.write_html("interactive_plot.html")
    fig.show()

def main(file_path, sample_size: int, n_clusters: int):
    print("Loading data...")
    df = load_data(file_path, sample_size=sample_size)
    
    print("Generating BERT embeddings...")
    embeddings = get_embeddings(df['text'].tolist())
    
    print("Clustering...")
    cluster_labels, kmeans = cluster_embeddings(embeddings, n_clusters=n_clusters)
    
    print("Visualizing...")
    visualize_clusters(embeddings, cluster_labels)
    
    df.to_csv("clustered_data.csv", index=False)
    print("Results saved to 'clustered_data.csv'")

def parse_args():
    parser = argparse.ArgumentParser(description="Read args for clustering")

    parser.add_argument("--file_path", type=str, default="../ignore_folder/data/docs_df.csv",
                        help="Path to .csv")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size for experimentations")
    parser.add_argument("--num_clusters", type=int, default=20,
                        help="Number of embedding clusters")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.file_path, args.sample_size, args.num_clusters)