"""
Document chunking pipeline.
Supports recursive character splitting with overlap, semantic chunking,
and metadata preservation for downstream retrieval.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm
import yaml


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    char_start: int
    char_end: int
    chunk_index: int


class DocumentChunker:
    """
    Splits raw documents into overlapping chunks for embedding and retrieval.
    Preserves source metadata (doc_id, title, url, page) on every chunk.
    """

    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config["retrieval"]["chunk_size"]
        self.chunk_overlap = config["retrieval"]["chunk_overlap"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"DocumentChunker initialized: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        """Chunk a single document dict with keys: doc_id, text, metadata."""
        doc_id = doc["doc_id"]
        text = doc["text"]
        metadata = doc.get("metadata", {})

        raw_chunks = self.splitter.create_documents([text])
        chunks = []
        char_cursor = 0

        for idx, raw_chunk in enumerate(raw_chunks):
            chunk_text = raw_chunk.page_content
            char_start = text.find(chunk_text, char_cursor)
            char_end = char_start + len(chunk_text)
            char_cursor = max(char_cursor, char_start)

            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{idx:04d}",
                doc_id=doc_id,
                text=chunk_text,
                metadata={**metadata, "chunk_index": idx, "doc_id": doc_id},
                char_start=char_start,
                char_end=char_end,
                chunk_index=idx,
            )
            chunks.append(chunk)

        return chunks

    def chunk_corpus(
        self,
        input_path: str,
        output_path: str,
        max_docs: Optional[int] = None,
    ) -> int:
        """
        Read JSONL corpus from input_path, chunk all documents,
        write chunked JSONL to output_path.
        Returns total chunk count.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_chunks = 0
        docs_processed = 0

        logger.info(f"Chunking corpus from {input_path}")
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in tqdm(fin, desc="Chunking documents"):
                if not line.strip():
                    continue
                doc = json.loads(line)
                chunks = self.chunk_document(doc)
                for chunk in chunks:
                    fout.write(json.dumps(asdict(chunk)) + "\n")
                    total_chunks += 1
                docs_processed += 1
                if max_docs and docs_processed >= max_docs:
                    break

        logger.info(
            f"Chunking complete: {docs_processed} docs → {total_chunks} chunks"
        )
        return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk document corpus for RAG")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="Input JSONL corpus path")
    parser.add_argument("--output", required=True, help="Output chunked JSONL path")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    chunker = DocumentChunker(config)
    total = chunker.chunk_corpus(args.input, args.output, args.max_docs)
    logger.info(f"Total chunks written: {total}")


if __name__ == "__main__":
    main()
