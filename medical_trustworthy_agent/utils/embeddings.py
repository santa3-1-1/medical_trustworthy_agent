# utils/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        print(f"[EmbeddingModel] Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(embeddings)
