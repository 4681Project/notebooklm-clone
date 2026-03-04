"""
retrieval.py
------------
Retrieval-Augmented Generation (RAG) module.

Supported retrieval techniques:
  1. naive          — Basic top-k cosine similarity
  2. mmr            — Maximal Marginal Relevance (diversity-aware)
  3. hyde           — Hypothetical Document Embeddings (HyDE)
  4. rerank         — Retrieve wide, then rerank with cross-encoder

Each technique returns a list of RetrievedChunk objects containing
the text and source metadata needed for citation.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from backend import storage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))   # 0=max diversity, 1=max relevance

RetrievalTechnique = Literal["naive", "mmr", "hyde", "rerank"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float
    technique: str
    metadata: dict = field(default_factory=dict)

    def citation_label(self) -> str:
        return f"[{self.source}, chunk {self.chunk_index}]"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_openai_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _embed(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _get_collection(username: str, notebook_id: str):
    chroma_path = storage.get_chroma_path(username, notebook_id)
    client = chromadb.PersistentClient(path=chroma_path)
    ef = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name="sources",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _chroma_results_to_chunks(results: dict, technique: str) -> list[RetrievedChunk]:
    chunks = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return chunks
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[0.0] * len(ids)])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(RetrievedChunk(
            text=doc,
            source=meta.get("source", "unknown"),
            chunk_index=meta.get("chunk_index", 0),
            score=1.0 - dist,  # cosine similarity
            technique=technique,
            metadata=meta,
        ))
    return chunks


# ---------------------------------------------------------------------------
# Retrieval technique 1: Naive top-k
# ---------------------------------------------------------------------------

def retrieve_naive(
    username: str,
    notebook_id: str,
    query: str,
    k: int = TOP_K,
    enabled_sources: Optional[list[str]] = None,
) -> tuple[list[RetrievedChunk], float]:
    """Basic dense retrieval — fastest, simplest baseline."""
    t0 = time.perf_counter()
    collection = _get_collection(username, notebook_id)
    where = {"source": {"$in": enabled_sources}} if enabled_sources else None
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    elapsed = time.perf_counter() - t0
    return _chroma_results_to_chunks(results, "naive"), elapsed


# ---------------------------------------------------------------------------
# Retrieval technique 2: Maximal Marginal Relevance (MMR)
# ---------------------------------------------------------------------------

def _cosine_sim(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_mmr(
    username: str,
    notebook_id: str,
    query: str,
    k: int = TOP_K,
    fetch_k: int = 20,
    lam: float = MMR_LAMBDA,
    enabled_sources: Optional[list[str]] = None,
) -> tuple[list[RetrievedChunk], float]:
    """
    MMR: retrieve fetch_k candidates, then greedily pick k that balance
    relevance to query and diversity from already-selected chunks.
    """
    t0 = time.perf_counter()
    collection = _get_collection(username, notebook_id)
    where = {"source": {"$in": enabled_sources}} if enabled_sources else None

    # Get candidates with embeddings
    results = collection.query(
        query_texts=[query],
        n_results=min(fetch_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    if not results["ids"][0]:
        return [], time.perf_counter() - t0

    candidates = list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["embeddings"][0],
    ))

    # Embed query
    q_emb = _embed([query])[0]

    selected_idx = []
    selected_embs = []

    for _ in range(min(k, len(candidates))):
        best_score, best_i = -1e9, -1
        for i, (doc, meta, dist, emb) in enumerate(candidates):
            if i in selected_idx:
                continue
            rel = 1.0 - dist  # relevance
            if selected_embs:
                redundancy = max(_cosine_sim(emb, s) for s in selected_embs)
            else:
                redundancy = 0.0
            score = lam * rel - (1 - lam) * redundancy
            if score > best_score:
                best_score, best_i = score, i

        if best_i == -1:
            break
        selected_idx.append(best_i)
        selected_embs.append(candidates[best_i][3])

    chunks = []
    for i in selected_idx:
        doc, meta, dist, _ = candidates[i]
        chunks.append(RetrievedChunk(
            text=doc,
            source=meta.get("source", "unknown"),
            chunk_index=meta.get("chunk_index", 0),
            score=1.0 - dist,
            technique="mmr",
            metadata=meta,
        ))

    return chunks, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Retrieval technique 3: HyDE (Hypothetical Document Embeddings)
# ---------------------------------------------------------------------------

def retrieve_hyde(
    username: str,
    notebook_id: str,
    query: str,
    k: int = TOP_K,
    enabled_sources: Optional[list[str]] = None,
) -> tuple[list[RetrievedChunk], float]:
    """
    HyDE: ask the LLM to generate a hypothetical answer, then use THAT
    as the query embedding.  Often finds more semantically relevant chunks.
    """
    t0 = time.perf_counter()
    client = _get_client()

    # Generate a short hypothetical passage
    hypo_prompt = (
        f"Write a concise, factual passage that would directly answer the question:\n"
        f'"{query}"\n'
        f"The passage should be 3-5 sentences, written as if it came from a reference document."
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": hypo_prompt}],
        max_tokens=200,
        temperature=0.0,
    )
    hypothetical_doc = resp.choices[0].message.content.strip()

    # Retrieve using hypothetical document as query
    collection = _get_collection(username, notebook_id)
    where = {"source": {"$in": enabled_sources}} if enabled_sources else None
    results = collection.query(
        query_texts=[hypothetical_doc],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    elapsed = time.perf_counter() - t0
    chunks = _chroma_results_to_chunks(results, "hyde")
    return chunks, elapsed


# ---------------------------------------------------------------------------
# Retrieval technique 4: Rerank (wide retrieval + cross-encoder scoring)
# ---------------------------------------------------------------------------

def retrieve_rerank(
    username: str,
    notebook_id: str,
    query: str,
    k: int = TOP_K,
    fetch_k: int = 20,
    enabled_sources: Optional[list[str]] = None,
) -> tuple[list[RetrievedChunk], float]:
    """
    Two-stage retrieval: fetch_k candidates with dense search,
    then rerank using an LLM-based cross-encoder scoring prompt.
    Falls back gracefully if cross-encoder is slow.
    """
    t0 = time.perf_counter()
    collection = _get_collection(username, notebook_id)
    where = {"source": {"$in": enabled_sources}} if enabled_sources else None

    results = collection.query(
        query_texts=[query],
        n_results=min(fetch_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    candidates = _chroma_results_to_chunks(results, "rerank")
    if not candidates:
        return [], time.perf_counter() - t0

    # Score each candidate with LLM relevance prompt
    client = _get_client()
    scored = []
    for chunk in candidates:
        score_prompt = (
            f"On a scale of 0-10, how relevant is this passage to the query?\n"
            f"Query: {query}\n"
            f"Passage: {chunk.text[:400]}\n"
            f"Respond with ONLY a number 0-10."
        )
        try:
            sr = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": score_prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            score_str = sr.choices[0].message.content.strip()
            llm_score = float(score_str.split()[0])
        except Exception:
            llm_score = chunk.score * 10  # fallback to cosine

        chunk.score = llm_score / 10.0
        scored.append(chunk)

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k], time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Unified retrieval entry point
# ---------------------------------------------------------------------------

def retrieve(
    username: str,
    notebook_id: str,
    query: str,
    technique: RetrievalTechnique = "recursive",
    k: int = TOP_K,
    enabled_sources: Optional[list[str]] = None,
) -> tuple[list[RetrievedChunk], float]:
    """
    Dispatch to the requested technique.
    Returns (chunks, elapsed_seconds).
    """
    if technique == "naive":
        return retrieve_naive(username, notebook_id, query, k, enabled_sources)
    elif technique == "mmr":
        return retrieve_mmr(username, notebook_id, query, k, enabled_sources=enabled_sources)
    elif technique == "hyde":
        return retrieve_hyde(username, notebook_id, query, k, enabled_sources)
    elif technique == "rerank":
        return retrieve_rerank(username, notebook_id, query, k, enabled_sources=enabled_sources)
    else:
        return retrieve_naive(username, notebook_id, query, k, enabled_sources)


def build_context_string(chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    """
    Format retrieved chunks into a context string for the LLM,
    and return a list of citation labels.
    """
    context_parts = []
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[Source {i}: {chunk.source}]"
        context_parts.append(f"{label}\n{chunk.text}")
        citations.append(label)
    return "\n\n---\n\n".join(context_parts), citations