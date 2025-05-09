import numpy as np
from typing import List, Dict, Optional
import requests
from scipy.spatial.distance import cdist
from transformers import AutoTokenizer

class ClusterLabelGenerator:
    def __init__(
        self,
        port: int = 30000,
        host: str = "localhost",
        default_prompt_config: Optional[dict] = None
    ):
        """
        Initialize the ClusterLabelGenerator with SGLANG server connection details.
        
        Args:
            port: Port where SGLANG server is running
            host: Host address of SGLANG server
            default_prompt_config: Default prompt configuration
        """
        self.port = port
        self.host = host
        self.base_url = f"http://{host}:{port}"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        
        # Default prompt configuration
        self.default_prompt_config = default_prompt_config or {
            "system_prompt": (
                """
                    You are a helpful clustering assistant. Your task is to generate concise, informative labels for text clusters.
                    You have been provided with a bunch of papers to generate labels for. Follow these rules:\n
                    - Summarize all the papers and extract the unifying ideas from all these papers.
                    - Return 2-3 max phrases that describes ALL the papers in the list.\n
                    - Don't be too specific and don't be too generic, but the phrases should describe all the papers.\n
                    - Return results as a comma separated list.
                    - Do not forget to keep the description to a maximum of 2-3 phrases
                """
            ),
            "user_prompt_template": (
                """
                    Generate a concise label for these related texts:\n{texts_str}\n
                    Label:
                """
            ),
            "sampling_params": {
                "max_new_tokens": 256,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0
            }
        }

    def _send_request(self, endpoint: str, data: dict) -> dict:
        """
        Send a request to the SGLANG server.
        
        Args:
            endpoint: API endpoint (e.g., '/generate')
            data: Payload to send
            
        Returns:
            dict: Response from server
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    def _format_texts_for_prompt(self, texts: List[str]) -> str:
        """
        Format the list of texts into a string suitable for the prompt.
        
        Args:
            texts: List of texts to format
            
        Returns:
            str: Formatted text string
        """
        return "\n".join(f"- {text}" for text in texts)  # Truncate long texts
        
    def _generate_label_prompt(
        self,
        texts: List[str],
        prompt_config: Optional[dict] = None,
        max_label_length: int = 50
    ) -> dict:
        """
        Generate the complete prompt in chat format for label generation.
        """
        config = prompt_config or self.default_prompt_config
        
        # Format the texts for inclusion in prompt
        texts_str = self._format_texts_for_prompt(texts)
        
        # Create chat-style messages
        messages = [
            {
                "role": "system", 
                "content": config["system_prompt"].format(max_label_length=max_label_length)
            },
            {
                "role": "user",
                "content": config["user_prompt_template"].format(texts_str=texts_str)
            }
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        # return text
        
        return {
            "text": text,
            "sampling_params": config["sampling_params"],
        }


    def generate_cluster_labels(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        cluster_labels: List[int],
        max_texts_per_cluster: int = 30,
        max_label_length: int = 50,
        prompt_config: Optional[dict] = None,
        metric: str = "cosine"
    ) -> Dict[int, str]:
        """
        Generate representative text labels for each cluster using SGLANG.
        
        Args:
            embeddings: Embedding vectors for texts
            texts: List of original texts
            cluster_labels: Cluster assignments for each text
            max_texts_per_cluster: Max texts to use for label generation
            max_label_length: Maximum length of generated labels
            prompt_config: Custom prompt configuration
            metric: Distance metric for centroid calculation
            
        Returns:
            dict: Mapping of cluster IDs to their labels
        """
        cluster_text_labels = {}
        unique_clusters = np.unique(cluster_labels)

        print(f"Length of texts is  {len(texts)}")
        
        for cluster_id in unique_clusters:
            # Get cluster members
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            
            # Find centroid and closest texts
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cdist([centroid], cluster_embeddings, metric=metric)
            closest_indices = cluster_indices[np.argsort(distances[0])[:max_texts_per_cluster]]
            representative_texts = [texts[idx] for idx in closest_indices]
            
            # Generate label using LLM
            prompt_data = self._generate_label_prompt(
                representative_texts,
                prompt_config,
                max_label_length
            )

            
            try:
                response = self._send_request("/generate", prompt_data)
                label = response.get("text", "").strip()
                
                # Fallback if label is empty
                if not label:
                    label = f"Cluster {cluster_id}"
            except Exception as e:
                print(f"Error generating label for cluster {cluster_id}: {e}")
                label = f"Cluster {cluster_id}"
            
            cluster_text_labels[int(cluster_id)] = label
        
        # Handle noise points
        if -1 in cluster_labels:
            cluster_text_labels[-1] = "Noise/Outliers"
        
        return cluster_text_labels

    @staticmethod
    def assign_labels_to_data(
        data: List[dict],
        cluster_labels: List[int],
        cluster_text_labels: Dict[int, str],
        text_field: str = "text",
        label_field: str = "cluster_label"
    ) -> List[dict]:
        """
        Assign generated cluster labels to the original data.
        
        Args:
            data: Original data records
            cluster_labels: Cluster assignments
            cluster_text_labels: Generated labels
            text_field: Field containing the text
            label_field: Field to store cluster label
            
        Returns:
            List[dict]: Updated data with cluster labels
        """
        return [
            {**record, label_field: cluster_text_labels.get(cluster_label, f"Cluster {cluster_label}")}
            for record, cluster_label in zip(data, cluster_labels)
        ]