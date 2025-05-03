import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.loader_utils import load_data, load_model_tokenizer

def get_local_embed_path(config) -> Tuple[str, str]:
    embed_dir = config["output_dir"] + "/embeddings"
    embed_path = f"{embed_dir}/embeddings_{config['sample_size']}.npy"
    return embed_dir, embed_path

def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
def generate_or_load_embeddings(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    texts: List[str],  
    config  
) -> np.ndarray:
    embed_dir, embed_path = get_local_embed_path(config)

    if config["load_from_local"]:
        assert os.path.exists(embed_path), "If loading embeddings from local dir, embeddings should exist in a local dir"
        print("Loading embeddings from", embed_path)
        return np.load(embed_path)

    embeddings = []
    for i in tqdm(range(0, len(texts), config['batch_size']), desc="Embedding"):
        batch_input = texts[i:i + config['batch_size']]
        inputs = tokenizer(batch_input, return_tensors='pt', padding=True, truncation=True, 
                          max_length=8192).to(config["device"])
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            norm_batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
            embeddings.append(norm_batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    
    if config['save_to_local']:
        os.makedirs(embed_dir, exist_ok=True)
        np.save(embed_path, embeddings)
    return embeddings

def main(config_path: str ="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loading data...")
    if config["sample"]:
        df = load_data(config["sample_file_path"])
    else:
        df = load_data(config["file_path"])

    print("Loading model and tokenizer")
    model, tokenizer = load_model_tokenizer(config["model_name"], device=config["device"])

    assert model is not None
    assert tokenizer is not None
    config["save_to_local"] = True
    
    print("Generating embeddings")
    embeddings = generate_or_load_embeddings(model, tokenizer, df['text'].tolist(), config)

if __name__ == "__main__":
    main()