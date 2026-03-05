import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from backend.storage import notebook_dir

load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from environment automatically


def get_embedding(text: str) -> list:
    """Get an embedding vector for a query string."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding


def get_retriever(nb_id: str, username: str = "default", top_k: int = 5):
    """
    Returns a retrieval function for the given notebook.
    The returned function takes a query string and returns (chunks, metadatas).
    """
    def retrieve(query: str):
        try:
            chroma_path = str(notebook_dir(username, nb_id) / "chroma")
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            collection = chroma_client.get_or_create_collection(
                name="sources",
                metadata={"hnsw:space": "cosine"},
            )

            # Return empty if nothing has been ingested yet
            if collection.count() == 0:
                return ["No sources have been ingested yet."], [{"source": "none", "chunk": 0}]

            query_embedding = get_embedding(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
            )

            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            return chunks, metadatas

        except Exception as e:
            return [f"Retrieval error: {str(e)}"], [{"source": "error", "chunk": 0}]

    return retrieve