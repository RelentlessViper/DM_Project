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

def load_model_tokenizer(model_name: str, device: str) -> AutoModelForCausalLM: 
    "Load the model and tokenizer from HuggingFace"
    
    if "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
    # Set model to eval mode
    model.eval()

    return model, tokenizer