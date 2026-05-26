"""
Embedding generation pipeline.
Encodes chunked documents using a sentence-transformer model,
builds a FAISS index, and persists both index and chunk metadata to disk.
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger
import yaml


class DocumentEmbedder:
    """
    Encodes document chunks into dense vectors and builds a FAISS flat index.
    Also persists a chunk metadata store (chunk_id → text + metadata).
    """

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config["retrieval"]["embedding_model"]
        self.index_path = Path(config["retrieval"]["faiss_index_path"])
        self.batch_size = 128

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.embedding_dim}")

    def embed_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode a list of chunk dicts.
        Returns (embeddings array [N, D], ordered list of chunk_ids).
        """
        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        logger.info(f"Embedding {len(texts)} chunks in batches of {self.batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity via inner product
        )
        return np.array(embeddings, dtype=np.float32), chunk_ids

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build a flat inner-product (cosine) FAISS index."""
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        logger.info(f"FAISS index built: {index.ntotal} vectors")
        return index

    def save(
        self,
        index: faiss.Index,
        chunk_ids: List[str],
        chunk_store: Dict[str, Dict[str, Any]],
    ):
        """Persist FAISS index, chunk_ids list, and chunk metadata store."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(self.index_path / "index.faiss"))
        logger.info(f"FAISS index saved → {self.index_path / 'index.faiss'}")

        with open(self.index_path / "chunk_ids.pkl", "wb") as f:
            pickle.dump(chunk_ids, f)

        with open(self.index_path / "chunk_store.pkl", "wb") as f:
            pickle.dump(chunk_store, f)

        logger.info(f"Chunk store saved ({len(chunk_store)} entries)")

    def load(self) -> Tuple[faiss.Index, List[str], Dict[str, Any]]:
        """Load persisted FAISS index, chunk_ids, and chunk store."""
        index = faiss.read_index(str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunk_ids.pkl", "rb") as f:
            chunk_ids = pickle.load(f)
        with open(self.index_path / "chunk_store.pkl", "rb") as f:
            chunk_store = pickle.load(f)
        logger.info(f"Loaded index with {index.ntotal} vectors")
        return index, chunk_ids, chunk_store

    def build_from_file(self, chunks_file: str):
        """End-to-end: read chunks JSONL → embed → build index → save."""
        chunks = []
        with open(chunks_file) as f:
            for line in tqdm(f, desc="Reading chunks"):
                if line.strip():
                    chunks.append(json.loads(line))

        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")

        # Build chunk store: chunk_id → {text, metadata, doc_id}
        chunk_store = {
            c["chunk_id"]: {
                "text": c["text"],
                "metadata": c["metadata"],
                "doc_id": c["doc_id"],
            }
            for c in chunks
        }

        embeddings, chunk_ids = self.embed_chunks(chunks)
        index = self.build_faiss_index(embeddings)
        self.save(index, chunk_ids, chunk_store)
        logger.info("Embedding pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--chunks", required=True, help="Chunked JSONL path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    embedder = DocumentEmbedder(config)
    embedder.build_from_file(args.chunks)


if __name__ == "__main__":
    main()
