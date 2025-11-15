import os
import faiss
import numpy as np
from utils.embeddings import EmbeddingModel
from utils.data_loader import DataLoader
from langchain.docstore.document import Document

class Retriever:
    def __init__(self, knowledge_dir="knowledge_dir", embedding_model=None, top_k=3):
        self.knowledge_dir = knowledge_dir
        self.embedding_model = embedding_model or EmbeddingModel()
        self.top_k = top_k
        self.docs = []
        self.index = None

        # 如果目录不存在，创建并放入示例文档
        if not os.path.exists(knowledge_dir):
            os.makedirs(knowledge_dir)
            sample_path = os.path.join(knowledge_dir, "sample_case.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("感冒通常可以通过多休息和多喝水缓解。常用药物包括对乙酰氨基酚和布洛芬。")
            print(f"[Retriever] Created empty knowledge base folder with a sample file: {knowledge_dir}")

        # 使用 DataLoader 加载文档
        self._load_documents()
        self._build_index()

    def _load_documents(self):
        loader = DataLoader(self.knowledge_dir)
        all_docs = loader.load_all()
        self.docs = [{"file": doc.metadata["source"], "text": doc.page_content} for doc in all_docs]
        print(f"[Retriever] Loaded {len(self.docs)} documents from {self.knowledge_dir}")

    def _build_index(self):
        if not self.docs:
            print("[Retriever] Warning: No documents found.")
            return
        texts = [d["text"] for d in self.docs]
        vectors = self.embedding_model.encode(texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        print(f"[Retriever] FAISS index built with {len(vectors)} documents")

    def retrieve(self, query):
        if self.index is None:
            print("[Retriever] No index available.")
            return []
        query_vec = self.embedding_model.encode(query)
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, 0)
        D, I = self.index.search(query_vec, self.top_k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.docs):
                results.append(self.docs[idx])
        return results
