"""
FAISS dense retriever.
Encodes query → searches FAISS index → returns top-k chunks with scores.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger


class FAISSRetriever:
    """Dense semantic retriever backed by a FAISS flat-IP index."""

    def __init__(self, config: Dict[str, Any]):
        self.top_k = config["retrieval"]["faiss_top_k"]
        self.index_path = Path(config["retrieval"]["faiss_index_path"])
        self.model_name = config["retrieval"]["embedding_model"]

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path / "index.faiss"))

        with open(self.index_path / "chunk_ids.pkl", "rb") as f:
            self.chunk_ids: List[str] = pickle.load(f)

        with open(self.index_path / "chunk_store.pkl", "rb") as f:
            self.chunk_store: Dict[str, Any] = pickle.load(f)

        logger.info(f"FAISS retriever ready: {self.index.ntotal} vectors, top_k={self.top_k}")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks for a query.
        Returns list of dicts: {chunk_id, text, metadata, score, rank}.
        """
        k = top_k or self.top_k
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunk_store[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "doc_id": chunk["doc_id"],
                "score": float(score),
                "rank": rank,
                "retriever": "faiss",
            })

        return results
