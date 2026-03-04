# 📓 NotebookLM Clone

A full-stack AI research assistant inspired by Google's NotebookLM. Upload documents, chat using RAG with citations, and generate study artifacts (reports, quizzes, podcasts).

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Auth** | Username/password login — per-user data isolation |
| **Notebooks** | Create / rename / delete multiple notebooks per user |
| **Ingestion** | PDF, PPTX, TXT, Markdown, Web URLs |
| **RAG Chat** | 4 retrieval techniques with inline citations |
| **Report** | AI-generated Markdown research report |
| **Quiz** | 10-question multiple-choice quiz + answer key |
| **Podcast** | Transcript + MP3 audio (single or two-host) |
| **Persistence** | Chat history and artifacts survive across sessions |
| **CI** | GitHub Actions — lint + test on every push/PR |

---

## 🏗️ Architecture

```
notebooklm-clone/
├── app.py                  ← Gradio UI + all callbacks (entry point)
├── backend/
│   ├── storage.py          ← Per-user/per-notebook file & metadata management
│   ├── ingestion.py        ← Text extraction + chunking + ChromaDB upsert
│   ├── retrieval.py        ← 4 RAG retrieval techniques
│   ├── chat.py             ← RAG chat with citation parsing + history
│   └── artifacts.py        ← Report / Quiz / Podcast generation + TTS
├── .github/workflows/
│   └── ci.yml              ← Lint + test on push/PR
├── data/                   ← Auto-created at runtime by storage.py
│   └── users/
│       └── <username>/
│           └── notebooks/
│               ├── index.json               ← list of all notebooks
│               └── <notebook-uuid>/
│                   ├── files_raw/           ← original uploads
│                   ├── files_extracted/     ← plain text extractions
│                   ├── chroma/              ← ChromaDB vector store
│                   ├── chat/
│                   │   └── messages.jsonl   ← persistent chat history
│                   └── artifacts/
│                       ├── reports/         ← report_1.md ...
│                       ├── quizzes/         ← quiz_1.md ...
│                       └── podcasts/        ← podcast_1.md, podcast_1.mp3 ...
├── .env.example
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

> **Note:** The `data/` folder and all subdirectories are created automatically by `storage.py` when the app first runs. You do not need to create anything manually.

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/<your-repo>
cd notebooklm-clone
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and USERS
```

### 3. Run

```bash
python app.py
# Open http://localhost:7860
```

---

## 🔐 Authentication

Users are defined via the `USERS` environment variable:

```
USERS="alice:password1,bob:password2"
```

Each user gets a completely isolated data directory. Add this to your `.env` (local) or as a repository secret (GitHub Actions / any deployment).

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `USERS` | ✅ | `admin:changeme` | Comma-separated `user:pass` pairs |
| `LLM_MODEL` | | `gpt-4o-mini` | Chat/artifact model |
| `EMBED_MODEL` | | `text-embedding-3-small` | Embedding model |
| `TTS_MODEL` | | `tts-1` | Text-to-speech model |
| `CHUNK_SIZE` | | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | | `150` | Overlap between chunks |
| `TOP_K` | | `5` | Chunks retrieved per query |
| `DATA_ROOT` | | `./data` | Root directory for all user data |
| `PORT` | | `7860` | Server port |

---

## 📊 RAG Techniques

See `ARCHITECTURE.md` for the full analysis.

| Technique | Speed | Best For |
|---|---|---|
| **Naive** | ⚡⚡⚡ | Quick factual lookups |
| **MMR** | ⚡⚡ | Broad summaries (diverse chunks) |
| **HyDE** | ⚡ | Abstract/indirect questions |
| **Rerank** | 🐢 | Complex, high-stakes queries |