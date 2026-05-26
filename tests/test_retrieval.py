"""
Unit and integration tests for the retrieval pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


MOCK_CONFIG = {
    "retrieval": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "faiss_top_k": 5,
        "bm25_top_k": 5,
        "reranker_top_k": 3,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "faiss_index_path": "tests/fixtures/faiss_index",
        "chunk_size": 512,
        "chunk_overlap": 64,
    }
}

MOCK_CHUNKS = [
    {"chunk_id": "doc1_chunk_0000", "doc_id": "doc1", "text": "Machine learning is a subset of artificial intelligence.", "metadata": {"source": "doc1"}},
    {"chunk_id": "doc1_chunk_0001", "doc_id": "doc1", "text": "Deep learning uses neural networks with many layers.", "metadata": {"source": "doc1"}},
    {"chunk_id": "doc2_chunk_0000", "doc_id": "doc2", "text": "RAG combines retrieval with language model generation.", "metadata": {"source": "doc2"}},
    {"chunk_id": "doc2_chunk_0001", "doc_id": "doc2", "text": "FAISS is a library for efficient similarity search.", "metadata": {"source": "doc2"}},
    {"chunk_id": "doc3_chunk_0000", "doc_id": "doc3", "text": "Fine-tuning adapts a pre-trained model to a specific domain.", "metadata": {"source": "doc3"}},
]


class TestDocumentChunker:
    def test_chunk_single_document(self):
        from src.ingestion.chunker import DocumentChunker
        config = {"retrieval": {"chunk_size": 100, "chunk_overlap": 20}}
        chunker = DocumentChunker(config)
        doc = {
            "doc_id": "test_doc",
            "text": "This is a test document. " * 20,
            "metadata": {"source": "test"},
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.doc_id == "test_doc"
            assert len(chunk.text) <= 150  # chunk_size + some tolerance
            assert chunk.chunk_id.startswith("test_doc_chunk_")

    def test_chunk_preserves_metadata(self):
        from src.ingestion.chunker import DocumentChunker
        config = {"retrieval": {"chunk_size": 200, "chunk_overlap": 20}}
        chunker = DocumentChunker(config)
        doc = {
            "doc_id": "meta_test",
            "text": "Some text content. " * 10,
            "metadata": {"author": "Test Author", "url": "https://example.com"},
        }
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert "author" in chunk.metadata
            assert chunk.metadata["author"] == "Test Author"

    def test_chunk_id_uniqueness(self):
        from src.ingestion.chunker import DocumentChunker
        config = {"retrieval": {"chunk_size": 50, "chunk_overlap": 10}}
        chunker = DocumentChunker(config)
        doc = {"doc_id": "unique_test", "text": "word " * 100, "metadata": {}}
        chunks = chunker.chunk_document(doc)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"


class TestBM25Retriever:
    def test_retrieve_returns_ranked_results(self):
        from src.retrieval.bm25_retriever import BM25Retriever
        ret = BM25Retriever.__new__(BM25Retriever)
        ret.top_k = 3
        from rank_bm25 import BM25Okapi
        texts = [c["text"] for c in MOCK_CHUNKS]
        ret.bm25 = BM25Okapi([t.lower().split() for t in texts])
        ret.chunk_ids = [c["chunk_id"] for c in MOCK_CHUNKS]
        ret.chunk_store = {c["chunk_id"]: c for c in MOCK_CHUNKS}

        results = ret.retrieve("machine learning neural networks", top_k=3)
        assert len(results) == 3
        assert all("chunk_id" in r for r in results)
        assert all("score" in r for r in results)
        # Verify descending score order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_query_relevance(self):
        from src.retrieval.bm25_retriever import BM25Retriever
        ret = BM25Retriever.__new__(BM25Retriever)
        ret.top_k = 5
        from rank_bm25 import BM25Okapi
        texts = [c["text"] for c in MOCK_CHUNKS]
        ret.bm25 = BM25Okapi([t.lower().split() for t in texts])
        ret.chunk_ids = [c["chunk_id"] for c in MOCK_CHUNKS]
        ret.chunk_store = {c["chunk_id"]: c for c in MOCK_CHUNKS}

        results = ret.retrieve("FAISS similarity search retrieval", top_k=3)
        top_chunk_id = results[0]["chunk_id"]
        # FAISS chunk should rank highest
        assert "faiss" in results[0]["text"].lower() or "retrieval" in results[0]["text"].lower()


class TestCrossEncoderReranker:
    def test_deduplication(self):
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.top_k = 3
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5]
        reranker.model = mock_model

        # Duplicate chunk_ids
        candidates = [
            {"chunk_id": "c1", "text": "text1", "metadata": {}, "doc_id": "d1", "score": 0.8, "retriever": "faiss"},
            {"chunk_id": "c1", "text": "text1", "metadata": {}, "doc_id": "d1", "score": 0.6, "retriever": "bm25"},
            {"chunk_id": "c2", "text": "text2", "metadata": {}, "doc_id": "d2", "score": 0.7, "retriever": "faiss"},
            {"chunk_id": "c3", "text": "text3", "metadata": {}, "doc_id": "d3", "score": 0.5, "retriever": "bm25"},
        ]
        results = reranker.rerank("test query", candidates, top_k=3)
        result_ids = [r["chunk_id"] for r in results]
        assert len(result_ids) == len(set(result_ids)), "Reranker must deduplicate"

    def test_rerank_respects_top_k(self):
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.top_k = 2
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]
        reranker.model = mock_model

        candidates = [
            {"chunk_id": f"c{i}", "text": f"text{i}", "metadata": {}, "doc_id": f"d{i}", "score": 0.5, "retriever": "faiss"}
            for i in range(5)
        ]
        results = reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2


class TestIntegration:
    """Integration test: full retrieval pipeline on mock data."""

    def test_hybrid_retriever_pipeline(self):
        mock_faiss = MagicMock()
        mock_faiss.retrieve.return_value = [
            {"chunk_id": "c1", "text": "RAG retrieval text", "metadata": {}, "doc_id": "d1", "score": 0.85, "retriever": "faiss"},
            {"chunk_id": "c2", "text": "LLM generation text", "metadata": {}, "doc_id": "d2", "score": 0.72, "retriever": "faiss"},
        ]
        mock_bm25 = MagicMock()
        mock_bm25.retrieve.return_value = [
            {"chunk_id": "c2", "text": "LLM generation text", "metadata": {}, "doc_id": "d2", "score": 12.3, "retriever": "bm25"},
            {"chunk_id": "c3", "text": "Fine-tuning domain adaptation", "metadata": {}, "doc_id": "d3", "score": 9.1, "retriever": "bm25"},
        ]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"chunk_id": "c1", "text": "RAG retrieval text", "metadata": {}, "doc_id": "d1", "score": 0.85, "reranker_score": 0.95, "retriever": "faiss", "final_rank": 0},
            {"chunk_id": "c3", "text": "Fine-tuning domain adaptation", "metadata": {}, "doc_id": "d3", "score": 9.1, "reranker_score": 0.88, "retriever": "bm25", "final_rank": 1},
        ]

        from src.retrieval.reranker import HybridRetriever
        hybrid = HybridRetriever(mock_faiss, mock_bm25, mock_reranker)
        results = hybrid.retrieve("RAG system fine-tuning")

        assert len(results) == 2
        mock_faiss.retrieve.assert_called_once_with("RAG system fine-tuning")
        mock_bm25.retrieve.assert_called_once_with("RAG system fine-tuning")
        mock_reranker.rerank.assert_called_once()
