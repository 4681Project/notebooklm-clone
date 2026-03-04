"""
chat.py
-------
RAG-powered chat with citation support.

Flow:
  1. Retrieve relevant chunks via `retrieval.retrieve()`
  2. Build a system prompt grounding the LLM in those chunks
  3. Call the LLM with full conversation history
  4. Parse citations from the response
  5. Persist the exchange to storage
"""

import os
import re
from typing import Generator, Optional

from openai import OpenAI

from backend import storage
from backend.retrieval import RetrievalTechnique, build_context_string, retrieve

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a helpful research assistant. The user has provided source documents for context.
Use the source excerpts below to inform your answers. When you reference specific information
from a source, cite it inline using its label exactly as shown, e.g. [Source 1: filename.pdf].

You may use your own knowledge to analyze, interpret, or give advice about the source content.
For example, if asked to improve a resume, analyze what is in the sources and provide actionable advice.

If the sources contain no relevant information at all, say so clearly.

--- SOURCE EXCERPTS ---
{context}
--- END OF SOURCES ---

Guidelines:
- Be helpful, specific, and actionable.
- Cite sources when referencing specific content from them.
- For analytical or improvement questions, use the source content as input and apply your expertise.
"""


def _build_system_prompt(context: str) -> str:
    return _SYSTEM_TEMPLATE.format(context=context)


# ---------------------------------------------------------------------------
# Chat function (non-streaming)
# ---------------------------------------------------------------------------

def chat(
    username: str,
    notebook_id: str,
    user_message: str,
    technique: RetrievalTechnique = "recursive",
    enabled_sources: Optional[list[str]] = None,
    k: int = 5,
) -> dict:
    """
    Send a message and get a cited response.

    Returns:
        {
          "answer": str,
          "citations": list[str],
          "chunks": list[RetrievedChunk],
          "retrieval_time_s": float,
          "technique": str,
        }
    """
    # 1. Retrieve
    chunks, retrieval_time = retrieve(
        username, notebook_id, user_message, technique=technique,
        k=k, enabled_sources=enabled_sources
    )
    context, citation_labels = build_context_string(chunks)

    # 2. Load history (last N turns)
    history = storage.load_chat_history(username, notebook_id)
    recent = history[-(MAX_HISTORY_TURNS * 2):]  # each turn = 2 messages

    messages = [{"role": "system", "content": _build_system_prompt(context)}]
    for msg in recent:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # 3. LLM call
    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()

    # 4. Extract cited source labels mentioned in the answer
    cited = list(set(re.findall(r'\[Source \d+:[^\]]+\]', answer)))

    # 5. Persist
    storage.append_message(username, notebook_id, "user", user_message)
    storage.append_message(
        username, notebook_id, "assistant", answer,
        metadata={"citations": cited, "technique": technique, "retrieval_time_s": retrieval_time}
    )

    return {
        "answer": answer,
        "citations": cited,
        "chunks": chunks,
        "retrieval_time_s": retrieval_time,
        "technique": technique,
    }


# ---------------------------------------------------------------------------
# Streaming version
# ---------------------------------------------------------------------------

def chat_stream(
    username: str,
    notebook_id: str,
    user_message: str,
    technique: RetrievalTechnique = "recursive",
    enabled_sources: Optional[list[str]] = None,
    k: int = 5,
) -> Generator[str, None, dict]:
    """
    Generator that yields answer tokens one by one for live streaming.
    After exhausting the generator, call .send(None) or just iterate —
    the final yielded item is a sentinel dict with metadata.
    """
    chunks, retrieval_time = retrieve(
        username, notebook_id, user_message, technique=technique,
        k=k, enabled_sources=enabled_sources
    )
    context, _ = build_context_string(chunks)

    history = storage.load_chat_history(username, notebook_id)
    recent = history[-(MAX_HISTORY_TURNS * 2):]

    messages = [{"role": "system", "content": _build_system_prompt(context)}]
    for msg in recent:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    client = _get_client()
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_answer += delta
        yield delta

    cited = list(set(re.findall(r'\[Source \d+:[^\]]+\]', full_answer)))
    storage.append_message(username, notebook_id, "user", user_message)
    storage.append_message(
        username, notebook_id, "assistant", full_answer,
        metadata={"citations": cited, "technique": technique, "retrieval_time_s": retrieval_time}
    )

    # Final metadata yield
    yield {
        "done": True,
        "citations": cited,
        "chunks": chunks,
        "retrieval_time_s": retrieval_time,
    }


# ---------------------------------------------------------------------------
# Gradio-compatible history formatter
# ---------------------------------------------------------------------------

def load_gradio_history(username: str, notebook_id: str) -> list[dict]:
    """
    Convert stored JSONL messages into Gradio 6.0 chatbot format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    messages = storage.load_chat_history(username, notebook_id)
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg["role"] in ("user", "assistant")
    ]