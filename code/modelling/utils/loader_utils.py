import os
import json
import pandas as pd

from datetime import datetime
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, AutoModelForCausalLM, AutoTokenizer, AutoModel

def load_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load the dataset. To load a small sample of the dataset, set the sample size"""
    
    df = pd.read_csv(file_path, low_memory=False)
    df['update_date'] = pd.to_datetime(df['update_date'])
    df['text'] = df['title'] + " " + df['abstract']
    
    if sample_size:
        df = df.sample(sample_size, random_state=42)
        df.to_csv(f'../../ignore_folder/data/sample_data_{sample_size}.csv', index=False)
        
    return df

def load_model_tokenizer(model_name: str, device: str) -> AutoModelForCausalLM: 
    "Load the model and tokenizer from HuggingFace"
    
    if "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
    elif "gte" in model_name.lower() or "qwen" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    elif "sentence-transformers" in model_name.lower():
        model = SentenceTransformer(model_name).to(device)
        return model, None 
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
        
    # Set model to eval mode
    model.eval()

    return model, tokenizer

def load_best_hdbscan_config(config_dir: str = "") -> Dict[str, Any]:
    """Loads the best HDBSCAN config from JSON file"""
    config_path = os.path.join(config_dir, "best_hdbscan.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""HDBSCAN config not found at {config_path}. 
            Run tuning first or check path."""
        )
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in HDBSCAN config file")