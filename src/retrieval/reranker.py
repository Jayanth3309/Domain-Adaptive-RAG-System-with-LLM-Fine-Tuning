"""
Cross-encoder reranker.
Takes merged candidates from FAISS + BM25, scores each (query, chunk) pair
with a cross-encoder, and returns the top-k by joint relevance score.
"""

from typing import List, Dict, Any

from sentence_transformers import CrossEncoder
from loguru import logger


class CrossEncoderReranker:
    """
    Reranks a merged candidate pool from dense + sparse retrievers
    using a cross-encoder model for more accurate relevance scoring.
    """

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config["retrieval"]["reranker_model"]
        self.top_k = config["retrieval"]["reranker_top_k"]

        logger.info(f"Loading cross-encoder: {self.model_name}")
        self.model = CrossEncoder(self.model_name, max_length=512)
        logger.info("Cross-encoder reranker ready")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Score all (query, chunk_text) pairs and return top-k.
        Deduplicates by chunk_id before scoring.
        """
        k = top_k or self.top_k

        # Deduplicate by chunk_id, keeping best score per chunk
        seen = {}
        for c in candidates:
            cid = c["chunk_id"]
            if cid not in seen or c["score"] > seen[cid]["score"]:
                seen[cid] = c
        unique = list(seen.values())

        if not unique:
            return []

        pairs = [(query, c["text"]) for c in unique]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for c, score in zip(unique, scores):
            c["reranker_score"] = float(score)

        reranked = sorted(unique, key=lambda x: x["reranker_score"], reverse=True)

        for rank, c in enumerate(reranked[:k]):
            c["final_rank"] = rank

        logger.debug(f"Reranked {len(unique)} candidates → top {k} selected")
        return reranked[:k]


class HybridRetriever:
    """
    Full hybrid retrieval pipeline:
    FAISS dense + BM25 sparse → cross-encoder reranker.
    """

    def __init__(self, faiss_retriever, bm25_retriever, reranker):
        self.faiss = faiss_retriever
        self.bm25 = bm25_retriever
        self.reranker = reranker

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        dense_results = self.faiss.retrieve(query)
        sparse_results = self.bm25.retrieve(query)
        candidates = dense_results + sparse_results
        return self.reranker.rerank(query, candidates)
