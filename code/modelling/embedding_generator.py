import os
import torch
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_local_embed_path(output_dir: str) -> Tuple[str, str]:
    embed_dir = output_dir + "/embeddings"
    embed_path = f"{embed_dir}/embeddings.npy"
    return embed_dir, embed_path

def generate_or_load_embeddings(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    texts: List[str],  
    config  
) -> np.ndarray:
    embed_dir, embed_path = get_local_embed_path(config["output_dir"])

    if config["load_from_local"]:
        assert os.path.exists(embed_path), "If loading embeddings from local dir, embeddings should exist in a local dir"
        print("Loading embeddings from", embed_path)
        return np.load(embed_path)

    embeddings = []
    for i in tqdm(range(0, len(texts), config['batch_size']), desc="Embedding"):
        batch_input = texts[i:i + config['batch_size']]
        inputs = tokenizer(batch_input, return_tensors='pt', padding=True, truncation=True, 
                          max_length=512).to(config["device"])
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

    embeddings = np.vstack(embeddings)
    os.makedirs(embed_dir, exist_ok=True)
    np.save(embed_path, embeddings)
    return embeddings