import pandas as pd

from typing import Optional
from transformers import BertTokenizer, BertModel, AutoModelForCausalLM, AutoTokenizer

def load_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load the dataset. To load a small sample of the dataset, set the sample size"""
    
    df = pd.read_csv(file_path)
    df['text'] = df['title'] + " " + df['abstract']
    
    if sample_size:
        df = df.sample(sample_size, random_state=42)
        
    return df
