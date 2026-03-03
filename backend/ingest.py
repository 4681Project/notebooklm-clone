import uuid
from pathlib import Path

# Uncomment these once you run: pip install -r requirements.txt
# from PyPDF2 import PdfReader
# from pptx import Presentation
# import requests
# from bs4 import BeautifulSoup
# import chromadb

DATA_DIR = Path("data/users")


def get_notebook_dir(nb_id: str) -> Path:
    path = DATA_DIR / "default" / "notebooks" / nb_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_and_store(chunks: list, source_name: str, nb_id: str):
    """Embed chunks and store in ChromaDB."""
    # TODO: Uncomment when ChromaDB + OpenAI are configured
    # client = chromadb.PersistentClient(path=str(get_notebook_dir(nb_id) / "chroma"))
    # collection = client.get_or_create_collection("sources")
    # collection.add(
    #     documents=chunks,
    #     metadatas=[{"source": source_name, "chunk": i} for i in range(len(chunks))],
    #     ids=[str(uuid.uuid4()) for _ in chunks],
    # )
    print(f"[STUB] Would embed {len(chunks)} chunks from {source_name}")


def extract_pdf(filepath: str) -> str:
    """Extract text from a PDF file."""
    # reader = PdfReader(filepath)
    # return "\n".join(page.extract_text() or "" for page in reader.pages)
    return f"[STUB] Extracted text from PDF: {filepath}"


def extract_pptx(filepath: str) -> str:
    """Extract text from a PPTX file."""
    # prs = Presentation(filepath)
    # texts = [shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")]
    # return "\n".join(texts)
    return f"[STUB] Extracted text from PPTX: {filepath}"


def extract_txt(filepath: str) -> str:
    """Read plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def ingest_file(filepath: str, nb_id: str) -> str:
    """Ingest a file into the notebook vector store."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        text = extract_pdf(filepath)
    elif ext == ".pptx":
        text = extract_pptx(filepath)
    elif ext == ".txt":
        text = extract_txt(filepath)
    else:
        return f"Unsupported file type: {ext}"

    chunks = chunk_text(text)
    embed_and_store(chunks, Path(filepath).name, nb_id)
    return f"Ingested {Path(filepath).name} ({len(chunks)} chunks)"


def ingest_url(url: str, nb_id: str) -> str:
    """Fetch and ingest a web page."""
    # response = requests.get(url, timeout=10)
    # soup = BeautifulSoup(response.text, "html.parser")
    # text = soup.get_text(separator=" ", strip=True)
    text = f"[STUB] Fetched content from {url}"
    chunks = chunk_text(text)
    embed_and_store(chunks, url, nb_id)
    return f"Ingested URL: {url} ({len(chunks)} chunks)"