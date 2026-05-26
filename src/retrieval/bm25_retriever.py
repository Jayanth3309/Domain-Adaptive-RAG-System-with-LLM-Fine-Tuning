"""
BM25 sparse retriever.
Keyword-based retrieval to complement dense FAISS search.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from loguru import logger
from tqdm import tqdm


class BM25Retriever:
    """Sparse keyword retriever using BM25Okapi."""

    def __init__(self, config: Dict[str, Any]):
        self.top_k = config["retrieval"]["bm25_top_k"]
        self.index_path = Path(config["retrieval"]["faiss_index_path"])
        self.bm25_path = self.index_path / "bm25_index.pkl"

        if self.bm25_path.exists():
            self._load()
        else:
            logger.warning("BM25 index not found. Call build_from_file() first.")
            self.bm25 = None
            self.chunk_ids = []
            self.chunk_store = {}

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build_from_file(self, chunks_file: str):
        """Build BM25 index from chunked JSONL file."""
        chunks = []
        with open(chunks_file) as f:
            for line in tqdm(f, desc="Loading chunks for BM25"):
                if line.strip():
                    chunks.append(json.loads(line))

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_store = {
            c["chunk_id"]: {"text": c["text"], "metadata": c["metadata"], "doc_id": c["doc_id"]}
            for c in chunks
        }

        tokenized = [self._tokenize(c["text"]) for c in chunks]
        logger.info(f"Building BM25 index over {len(tokenized)} chunks...")
        self.bm25 = BM25Okapi(tokenized)

        self._save()
        logger.info("BM25 index built and saved")

    def _save(self):
        self.index_path.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids,
                "chunk_store": self.chunk_store,
            }, f)

    def _load(self):
        logger.info(f"Loading BM25 index from {self.bm25_path}")
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.chunk_store = data["chunk_store"]
        logger.info(f"BM25 index loaded: {len(self.chunk_ids)} chunks")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks by BM25 score."""
        k = top_k or self.top_k
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunk_store[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "doc_id": chunk["doc_id"],
                "score": float(scores[idx]),
                "rank": rank,
                "retriever": "bm25",
            })

        return results
