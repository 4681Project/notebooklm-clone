"""
storage.py — handles all reading/writing to the /data folder.

Structure:
    data/users/<username>/notebooks/index.json
    data/users/<username>/notebooks/<nb-id>/chat/messages.jsonl
    data/users/<username>/notebooks/<nb-id>/artifacts/reports/
    data/users/<username>/notebooks/<nb-id>/artifacts/quizzes/
    data/users/<username>/notebooks/<nb-id>/artifacts/podcasts/
    data/users/<username>/notebooks/<nb-id>/files_raw/
    data/users/<username>/notebooks/<nb-id>/files_extracted/
    data/users/<username>/notebooks/<nb-id>/chroma/
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path("data/users")

# Path helpers

def user_dir(username: str) -> Path:
    p = DATA_ROOT / username / "notebooks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def notebook_dir(username: str, nb_id: str) -> Path:
    p = user_dir(username) / nb_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def index_path(username: str) -> Path:
    return user_dir(username) / "index.json"


# Notebook CRUD

def get_user_notebooks(username: str) -> list:
    """Return list of notebook dicts for a user."""
    path = index_path(username)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_index(username: str, notebooks: list):
    with open(index_path(username), "w") as f:
        json.dump(notebooks, f, indent=2)


def create_notebook_for_user(username: str, name: str) -> str:
    """Create a new notebook and return its ID."""
    nb_id = str(uuid.uuid4())
    notebooks = get_user_notebooks(username)
    notebooks.append({
        "id": nb_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
    })
    _save_index(username, notebooks)

    # Pre-create subdirectories
    nb = notebook_dir(username, nb_id)
    for sub in ["files_raw", "files_extracted", "chroma", "chat",
                "artifacts/reports", "artifacts/quizzes", "artifacts/podcasts"]:
        (nb / sub).mkdir(parents=True, exist_ok=True)

    return nb_id


def rename_notebook(username: str, nb_id: str, new_name: str):
    notebooks = get_user_notebooks(username)
    for nb in notebooks:
        if nb["id"] == nb_id:
            nb["name"] = new_name
    _save_index(username, notebooks)


def delete_notebook(username: str, nb_id: str):
    import shutil
    notebooks = [nb for nb in get_user_notebooks(username) if nb["id"] != nb_id]
    _save_index(username, notebooks)
    nb_path = notebook_dir(username, nb_id)
    if nb_path.exists():
        shutil.rmtree(nb_path)


# Chat persistence

def _chat_path(username: str, nb_id: str) -> Path:
    p = notebook_dir(username, nb_id) / "chat"
    p.mkdir(parents=True, exist_ok=True)
    return p / "messages.jsonl"


def save_message(username: str, nb_id: str, role: str, content: str):
    """Append a single message to the chat history file."""
    path = _chat_path(username, nb_id)
    with open(path, "a") as f:
        f.write(json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }) + "\n")


def load_chat_history(username: str, nb_id: str) -> list:
    """Return list of {role, content} dicts."""
    path = _chat_path(username, nb_id)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# Artifact helpers

def save_artifact(username: str, nb_id: str, artifact_type: str,
                  content: str, extension: str = "md") -> Path:
    """
    Save an artifact and return its path.
    artifact_type: 'reports' | 'quizzes' | 'podcasts'
    """
    folder = notebook_dir(username, nb_id) / "artifacts" / artifact_type
    folder.mkdir(parents=True, exist_ok=True)
    existing = list(folder.glob(f"*.{extension}"))
    index = len(existing) + 1
    path = folder / f"{artifact_type[:-1]}_{index}.{extension}"
    path.write_text(content, encoding="utf-8")
    return path


def list_artifacts(username: str, nb_id: str, artifact_type: str) -> list:
    """Return list of artifact filenames for a given type."""
    folder = notebook_dir(username, nb_id) / "artifacts" / artifact_type
    if not folder.exists():
        return []
    return [p.name for p in sorted(folder.iterdir())]


# Raw file storage

def save_raw_file(username: str, nb_id: str, filename: str, data: bytes) -> Path:
    folder = notebook_dir(username, nb_id) / "files_raw"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_bytes(data)
    return path


def save_extracted_text(username: str, nb_id: str, filename: str, text: str) -> Path:
    folder = notebook_dir(username, nb_id) / "files_extracted"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (filename + ".txt")
    path.write_text(text, encoding="utf-8")
    return path