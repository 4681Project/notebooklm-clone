import os
from dotenv import load_dotenv
from openai import OpenAI
from backend.retrieval import get_retriever

load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from environment automatically


def format_history_for_openai(history: list) -> list:
    """
    Convert Gradio chat history [[user, assistant], ...] 
    into OpenAI message format.
    """
    messages = []
    for pair in history:
        if pair[0]:
            messages.append({"role": "user", "content": pair[0]})
        if pair[1]:
            messages.append({"role": "assistant", "content": pair[1]})
    return messages


def chat_with_sources(message: str, nb_id: str, username: str, history: list) -> str:
    """
    Retrieve relevant chunks from ChromaDB, then ask GPT-4o 
    to answer using only those chunks. Returns answer with citations.
    """
    try:
        # Retrieve top-k relevant chunks
        retrieve = get_retriever(nb_id, username=username, top_k=5)
        chunks, metas = retrieve(message)

        # Build context string with source labels
        context_parts = []
        for chunk, meta in zip(chunks, metas):
            source = meta.get("source", "unknown")
            chunk_num = meta.get("chunk", "?")
            context_parts.append(f"[Source: {source} | Chunk: {chunk_num}]\n{chunk}")
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful research assistant. "
            "Answer the user's question using ONLY the context provided below. "
            "After each claim, cite the source inline using the format: "
            "[Source: filename | Chunk: N]. "
            "If the context does not contain enough information to answer, "
            "say so clearly rather than making things up.\n\n"
            f"Context:\n{context}"
        )

        # Build message history for multi-turn conversation
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages += format_history_for_openai(history)
        openai_messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        # Append a deduplicated source list at the bottom
        seen_sources = set()
        source_list = []
        for meta in metas:
            src = meta.get("source", "unknown")
            if src not in seen_sources:
                seen_sources.add(src)
                source_list.append(f"- {src}")

        if source_list:
            answer += "\n\n**Sources consulted:**\n" + "\n".join(source_list)

        return answer

    except Exception as e:
        return f"Error generating response: {str(e)}"