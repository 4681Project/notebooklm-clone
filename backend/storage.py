"""
storage.py
----------
Storage abstraction layer for per-user, per-notebook data isolation.

Directory layout:
/data/users/<username>/notebooks/
    index.json                     <- list of all notebooks
    <notebook-uuid>/
        files_raw/                 <- original uploads
        files_extracted/           <- plain-text extractions
        chroma/                    <- ChromaDB vector store
        chat/messages.jsonl        <- chat history
        artifacts/
            reports/
            quizzes/
            podcasts/
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Root data directory — writable on HF Spaces
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _user_root(username: str) -> Path:
    return DATA_ROOT / "users" / username / "notebooks"


def _nb_root(username: str, notebook_id: str) -> Path:
    return _user_root(username) / notebook_id


def _nb_index_path(username: str) -> Path:
    return _user_root(username) / "index.json"


# ---------------------------------------------------------------------------
# Notebook index helpers
# ---------------------------------------------------------------------------

def _load_index(username: str) -> list[dict]:
    path = _nb_index_path(username)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_index(username: str, index: list[dict]) -> None:
    path = _nb_index_path(username)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_notebooks(username: str) -> list[dict]:
    """Return list of notebook metadata dicts for a user."""
    return _load_index(username)


def create_notebook(username: str, name: str) -> dict:
    """Create a new notebook and return its metadata."""
    nb_id = str(uuid.uuid4())
    nb_path = _nb_root(username, nb_id)

    # Create all sub-directories
    for subdir in [
        "files_raw",
        "files_extracted",
        "chroma",
        "chat",
        "artifacts/reports",
        "artifacts/quizzes",
        "artifacts/podcasts",
    ]:
        (nb_path / subdir).mkdir(parents=True, exist_ok=True)

    entry = {
        "id": nb_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    index = _load_index(username)
    index.append(entry)
    _save_index(username, index)
    return entry


def rename_notebook(username: str, notebook_id: str, new_name: str) -> bool:
    index = _load_index(username)
    for nb in index:
        if nb["id"] == notebook_id:
            nb["name"] = new_name
            nb["updated_at"] = datetime.utcnow().isoformat()
            _save_index(username, index)
            return True
    return False


def delete_notebook(username: str, notebook_id: str) -> bool:
    index = _load_index(username)
    new_index = [nb for nb in index if nb["id"] != notebook_id]
    if len(new_index) == len(index):
        return False
    shutil.rmtree(_nb_root(username, notebook_id), ignore_errors=True)
    _save_index(username, new_index)
    return True


def get_notebook(username: str, notebook_id: str) -> Optional[dict]:
    for nb in _load_index(username):
        if nb["id"] == notebook_id:
            return nb
    return None


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def save_raw_file(username: str, notebook_id: str, filename: str, data: bytes) -> Path:
    dest = _nb_root(username, notebook_id) / "files_raw" / filename
    dest.write_bytes(data)
    _touch_notebook(username, notebook_id)
    return dest


def save_extracted_text(username: str, notebook_id: str, source_name: str, text: str) -> Path:
    stem = Path(source_name).stem
    dest = _nb_root(username, notebook_id) / "files_extracted" / f"{stem}.txt"
    dest.write_text(text, encoding="utf-8")
    return dest


def list_sources(username: str, notebook_id: str) -> list[str]:
    raw_dir = _nb_root(username, notebook_id) / "files_raw"
    if not raw_dir.exists():
        return []
    return [f.name for f in raw_dir.iterdir() if f.is_file()]


def list_extracted(username: str, notebook_id: str) -> list[Path]:
    ext_dir = _nb_root(username, notebook_id) / "files_extracted"
    if not ext_dir.exists():
        return []
    return list(ext_dir.glob("*.txt"))


def get_chroma_path(username: str, notebook_id: str) -> str:
    return str(_nb_root(username, notebook_id) / "chroma")


# ---------------------------------------------------------------------------
# Chat persistence
# ---------------------------------------------------------------------------

def _chat_path(username: str, notebook_id: str) -> Path:
    return _nb_root(username, notebook_id) / "chat" / "messages.jsonl"


def append_message(username: str, notebook_id: str, role: str, content: str, metadata: Optional[dict] = None) -> None:
    path = _chat_path(username, notebook_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
        **(metadata or {}),
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_chat_history(username: str, notebook_id: str) -> list[dict]:
    path = _chat_path(username, notebook_id)
    if not path.exists():
        return []
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def clear_chat(username: str, notebook_id: str) -> None:
    path = _chat_path(username, notebook_id)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def _artifact_dir(username: str, notebook_id: str, kind: str) -> Path:
    return _nb_root(username, notebook_id) / "artifacts" / kind


def save_artifact(username: str, notebook_id: str, kind: str, content: bytes | str, ext: str) -> Path:
    """Save an artifact and return its path. kind: reports | quizzes | podcasts"""
    art_dir = _artifact_dir(username, notebook_id, kind)
    art_dir.mkdir(parents=True, exist_ok=True)
    existing = list(art_dir.glob(f"*.{ext}"))
    idx = len(existing) + 1
    dest = art_dir / f"{kind[:-1]}_{idx}.{ext}"
    if isinstance(content, str):
        dest.write_text(content, encoding="utf-8")
    else:
        dest.write_bytes(content)
    _touch_notebook(username, notebook_id)
    return dest


def list_artifacts(username: str, notebook_id: str, kind: str) -> list[Path]:
    art_dir = _artifact_dir(username, notebook_id, kind)
    if not art_dir.exists():
        return []
    return sorted(art_dir.iterdir())


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _touch_notebook(username: str, notebook_id: str) -> None:
    index = _load_index(username)
    for nb in index:
        if nb["id"] == notebook_id:
            nb["updated_at"] = datetime.utcnow().isoformat()
    _save_index(username, index)