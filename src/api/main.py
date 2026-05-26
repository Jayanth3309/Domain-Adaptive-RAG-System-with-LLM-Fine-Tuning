"""
FastAPI serving layer for the RAG system.
Exposes /query endpoint that runs the full retrieval + generation pipeline.
Includes health check, metrics endpoint, and request logging.
"""

import time
from typing import List, Optional
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker, HybridRetriever
from src.generation.inference import RAGGenerator


# ── Globals ──────────────────────────────────────────────────────────────────
config: dict = {}
retriever: HybridRetriever = None
generator: RAGGenerator = None
request_count: int = 0
total_latency_ms: float = 0.0


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, retriever, generator

    logger.info("Loading config...")
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing retrieval pipeline...")
    faiss_ret = FAISSRetriever(config)
    bm25_ret = BM25Retriever(config)
    reranker = CrossEncoderReranker(config)
    retriever = HybridRetriever(faiss_ret, bm25_ret, reranker)

    logger.info("Initializing generator...")
    generator = RAGGenerator(config)

    logger.info("RAG system ready to serve requests")
    yield
    logger.info("Shutting down RAG system")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Domain-Adaptive RAG API",
    description="Production RAG system with Mistral-7B fine-tuned via QLoRA",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    max_new_tokens: Optional[int] = Field(default=512, ge=64, le=1024)


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    doc_id: str
    retriever: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_chunks: List[ChunkResult]
    latency_ms: float
    num_chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    total_requests: int
    avg_latency_ms: float


# ── Middleware ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{elapsed:.1f}ms]")
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    avg = (total_latency_ms / request_count) if request_count > 0 else 0.0
    return HealthResponse(
        status="healthy",
        total_requests=request_count,
        avg_latency_ms=round(avg, 2),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global request_count, total_latency_ms

    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    start = time.perf_counter()

    try:
        # Retrieve
        chunks = retriever.retrieve(req.question)
        top_chunks = chunks[:req.top_k]

        # Generate
        gen_result = generator.generate(
            query=req.question,
            chunks=top_chunks,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000
    request_count += 1
    total_latency_ms += latency_ms

    return QueryResponse(
        answer=gen_result["answer"],
        sources=gen_result["sources"],
        retrieved_chunks=[
            ChunkResult(
                chunk_id=c["chunk_id"],
                text=c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                score=round(c.get("reranker_score", c.get("score", 0.0)), 4),
                doc_id=c["doc_id"],
                retriever=c.get("retriever", "hybrid"),
            )
            for c in top_chunks
        ],
        latency_ms=round(latency_ms, 2),
        num_chunks_retrieved=len(top_chunks),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
