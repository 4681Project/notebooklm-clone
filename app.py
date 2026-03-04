"""
app.py
------
Main Gradio application for the NotebookLM Clone.

Authentication: simple username/password pairs loaded from the USERS env var.
Format: USERS="alice:password1,bob:password2"

Features:
- Username/password login (per-user data isolation)
- Notebook manager (create / rename / delete / switch)
- Source ingestion (PDF, PPTX, TXT, URL)
- RAG chat with inline citations and technique selector
- Artifact generation: report, quiz, podcast (single + two-host)
- Artifact download and audio playback
- Persistent chat history across sessions
"""

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from backend import artifacts, chat, ingestion, storage

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _load_users() -> list[tuple[str, str]]:
    """
    Parse USERS env var into (username, password) tuples.
    Format: "alice:pass1,bob:pass2"
    Falls back to admin:changeme for local dev.
    """
    raw = os.getenv("USERS", "admin:changeme")
    users = []
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            username, password = entry.split(":", 1)
            users.append((username.strip(), password.strip()))
    return users


AUTH_USERS = _load_users()


def _authenticate(username: str, password: str) -> bool:
    for u, p in AUTH_USERS:
        if u == username and p == password:
            return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _username(request: gr.Request) -> str:
    if request and hasattr(request, "username") and request.username:
        return request.username
    return "anonymous"


def _nb_choices(username: str) -> list[str]:
    nbs = storage.list_notebooks(username)
    return [f"{nb['name']} ({nb['id'][:8]})" for nb in nbs]


def _parse_short_id(choice: str) -> Optional[str]:
    if not choice:
        return None
    m = re.search(r'\(([a-f0-9\-]+)\)', choice)
    return m.group(1) if m else None


def _resolve_nb(username: str, choice: str) -> Optional[str]:
    short = _parse_short_id(choice)
    if not short:
        return None
    for nb in storage.list_notebooks(username):
        if nb["id"].startswith(short):
            return nb["id"]
    return None


# ---------------------------------------------------------------------------
# Notebook callbacks
# ---------------------------------------------------------------------------

def create_nb(name: str, request: gr.Request):
    username = _username(request)
    if not name.strip():
        return gr.update(), gr.update(), "Please enter a notebook name."
    nb = storage.create_notebook(username, name.strip())
    choices = _nb_choices(username)
    new_val = f"{nb['name']} ({nb['id'][:8]})"
    return gr.update(choices=choices, value=new_val), gr.update(choices=choices, value=new_val), f"Created: {nb['name']}"


def rename_nb(choice: str, new_name: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, choice)
    if not nb_id:
        return gr.update(), gr.update(), "No notebook selected."
    if not new_name.strip():
        return gr.update(), gr.update(), "New name cannot be empty."
    storage.rename_notebook(username, nb_id, new_name.strip())
    choices = _nb_choices(username)
    new_val = f"{new_name.strip()} ({nb_id[:8]})"
    return gr.update(choices=choices, value=new_val), gr.update(choices=choices, value=new_val), f"Renamed to: {new_name.strip()}"


def delete_nb(choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, choice)
    if not nb_id:
        return gr.update(), gr.update(), "No notebook selected."
    storage.delete_notebook(username, nb_id)
    choices = _nb_choices(username)
    val = choices[0] if choices else None
    return gr.update(choices=choices, value=val), gr.update(choices=choices, value=val), "Notebook deleted."


def refresh_nbs(request: gr.Request):
    username = _username(request)
    choices = _nb_choices(username)
    val = choices[0] if choices else None
    return gr.update(choices=choices, value=val), gr.update(choices=choices, value=val)


# ---------------------------------------------------------------------------
# Ingestion callbacks
# ---------------------------------------------------------------------------

def ingest_files(files, url: str, strategy: str, nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if not nb_id:
        return "Please select or create a notebook first.", gr.update()

    messages = []
    if files:
        for f in files:
            try:
                p = Path(f.name)
                result = ingestion.ingest_source(username, nb_id, p, chunk_strategy=strategy, raw_bytes=p.read_bytes())
                messages.append(f"✅ {result['source_name']} — {result['chunk_count']} chunks ({strategy})")
            except Exception as e:
                messages.append(f"❌ {Path(f.name).name}: {e}")

    if url.strip():
        try:
            result = ingestion.ingest_source(username, nb_id, url.strip(), chunk_strategy=strategy)
            messages.append(f"✅ {result['source_name']} — {result['chunk_count']} chunks ({strategy})")
        except Exception as e:
            messages.append(f"❌ URL: {e}")

    if not messages:
        return "Nothing to ingest.", gr.update()

    sources = ingestion.list_indexed_sources(username, nb_id)
    return "\n".join(messages), gr.update(choices=sources, value=sources)


def load_sources(nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if not nb_id:
        return gr.update(choices=[], value=[])
    sources = ingestion.list_indexed_sources(username, nb_id)
    return gr.update(choices=sources, value=sources)


def delete_source_cb(source: str, nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if not nb_id or not source:
        return "Nothing to delete.", gr.update()
    n = ingestion.delete_source(username, nb_id, source)
    sources = ingestion.list_indexed_sources(username, nb_id)
    return f"Removed {n} chunks for '{source}'.", gr.update(choices=sources, value=sources)


# ---------------------------------------------------------------------------
# Chat callbacks
# ---------------------------------------------------------------------------

def load_history(nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if not nb_id:
        return []
    return chat.load_gradio_history(username, nb_id)


def send_message(user_msg: str, history: list, technique: str, enabled_sources: list, nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if not nb_id:
        return history, user_msg, "❌ Please select a notebook first."
    if not user_msg.strip():
        return history, "", ""

    try:
        result = chat.chat(username, nb_id, user_msg.strip(), technique=technique, enabled_sources=enabled_sources or None)
        answer = result["answer"]
        info = f"Technique: **{technique}** | ⏱ {result['retrieval_time_s']:.2f}s | {len(result['citations'])} citation(s)"
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": answer})
        return history, "", info
    except Exception as e:
        return history, user_msg, f"❌ Error: {e}"


def clear_chat_cb(nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if nb_id:
        storage.clear_chat(username, nb_id)
    return [], ""


# ---------------------------------------------------------------------------
# Artifact callbacks
# ---------------------------------------------------------------------------

def gen_report(extra: str, enabled_sources: list, nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if not nb_id:
        return "No notebook selected.", None, gr.update()
    try:
        path = artifacts.generate_report(username, nb_id, extra, enabled_sources or None)
        arts = [str(p) for p in storage.list_artifacts(username, nb_id, "reports")]
        return path.read_text(), str(path), gr.update(choices=arts)
    except Exception as e:
        return f"Error: {e}", None, gr.update()


def gen_quiz(extra: str, enabled_sources: list, nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if not nb_id:
        return "No notebook selected.", None, gr.update()
    try:
        path = artifacts.generate_quiz(username, nb_id, extra, enabled_sources or None)
        arts = [str(p) for p in storage.list_artifacts(username, nb_id, "quizzes")]
        return path.read_text(), str(path), gr.update(choices=arts)
    except Exception as e:
        return f"Error: {e}", None, gr.update()


def gen_podcast(extra: str, two_hosts: bool, duration: int, gen_audio: bool, enabled_sources: list, nb_choice: str, request: gr.Request):
    username = _username(request)
    nb_id = _resolve_nb(username, nb_choice)
    if not nb_id:
        return "❌ No notebook selected.", "", None, None, gr.update()
    try:
        tx_path = artifacts.generate_podcast_transcript(username, nb_id, duration_minutes=duration, two_hosts=two_hosts, extra_instructions=extra, enabled_sources=enabled_sources or None)
        transcript = tx_path.read_text(encoding="utf-8", errors="ignore")
        audio_path = None
        status = "✅ Transcript generated."
        if gen_audio:
            status = "✅ Transcript generated. Generating audio (this may take 30–60 seconds)…"
            audio_path = str(artifacts.generate_podcast_audio(username, nb_id, tx_path, two_hosts=two_hosts))
            status = "✅ Done — transcript and audio ready."
        arts = [str(p) for p in storage.list_artifacts(username, nb_id, "podcasts")]
        return status, transcript, str(tx_path), audio_path, gr.update(choices=arts)
    except Exception as e:
        return f"❌ Error: {e}", "", None, None, gr.update()


def load_artifact(path: str):
    if not path:
        return "", None
    p = Path(path)
    if not p.exists():
        return "File not found.", None
    return (p.read_text(), str(p)) if p.suffix == ".md" else ("", str(p))


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="NotebookLM Clone") as demo:
        gr.Markdown("# 📓 NotebookLM Clone\nUpload sources, chat with RAG, and generate study artifacts.")

        with gr.Row():
            # LEFT — notebooks + sources
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 📚 Notebooks")
                nb_dropdown = gr.Dropdown(label="Active Notebook", choices=[], interactive=True)
                with gr.Row():
                    nb_name_input = gr.Textbox(placeholder="New notebook name…", show_label=False, scale=3)
                    create_btn = gr.Button("➕ Create", scale=1, variant="primary")
                with gr.Row():
                    rename_input = gr.Textbox(placeholder="Rename to…", show_label=False, scale=3)
                    rename_btn = gr.Button("✏️ Rename", scale=1)
                delete_btn = gr.Button("🗑️ Delete Notebook", variant="stop")
                nb_status = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("### 📁 Sources")
                file_upload = gr.File(label="Upload (PDF, PPTX, TXT)", file_types=[".pdf", ".pptx", ".ppt", ".txt", ".md"], file_count="multiple")
                url_input = gr.Textbox(label="Or enter a URL", placeholder="https://…")
                chunk_strategy = gr.Radio(choices=["recursive", "sentence", "fixed"], value="recursive", label="Chunking Strategy")
                ingest_btn = gr.Button("⬆️ Ingest", variant="primary")
                ingest_status = gr.Markdown("")
                sources_list = gr.CheckboxGroup(label="Active Sources (uncheck to disable for RAG)", choices=[], value=[], interactive=True)
                with gr.Row():
                    source_to_delete = gr.Textbox(placeholder="Source name to remove…", show_label=False, scale=3)
                    del_src_btn = gr.Button("🗑️ Remove", scale=1, variant="stop")
                del_src_status = gr.Markdown("")

            # RIGHT — chat + artifacts
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("💬 Chat"):
                        retrieve_technique = gr.Radio(choices=["naive", "mmr", "hyde", "rerank"], value="naive", label="Retrieval Technique", info="naive=fast | mmr=diverse | hyde=hypothetical | rerank=LLM scoring")
                        chatbot = gr.Chatbot(elem_id="chat-panel", label="Chat", height=420)
                        chat_status = gr.Markdown("")
                        with gr.Row():
                            chat_input = gr.Textbox(placeholder="Ask anything about your sources…", show_label=False, scale=5)
                            send_btn = gr.Button("Send ➤", variant="primary", scale=1)
                        clear_btn = gr.Button("🧹 Clear Chat History")

                    with gr.TabItem("📄 Report"):
                        report_extra = gr.Textbox(label="Optional focus instructions", placeholder="e.g. Focus on section 3 and the conclusion")
                        gen_report_btn = gr.Button("📝 Generate Report", variant="primary")
                        report_output = gr.Markdown(label="Generated Report")
                        report_file = gr.File(label="Download Report")
                        report_list = gr.Dropdown(label="Past Reports", choices=[], interactive=True)
                        load_report_btn = gr.Button("📂 Load Selected")

                    with gr.TabItem("🧠 Quiz"):
                        quiz_extra = gr.Textbox(label="Optional focus instructions", placeholder="e.g. Focus on key definitions")
                        gen_quiz_btn = gr.Button("🎯 Generate Quiz", variant="primary")
                        quiz_output = gr.Markdown(label="Generated Quiz")
                        quiz_file = gr.File(label="Download Quiz")
                        quiz_list = gr.Dropdown(label="Past Quizzes", choices=[], interactive=True)
                        load_quiz_btn = gr.Button("📂 Load Selected")

                    with gr.TabItem("🎙️ Podcast"):
                        podcast_extra = gr.Textbox(label="Optional focus instructions", placeholder="e.g. Keep it conversational")
                        with gr.Row():
                            podcast_duration = gr.Slider(2, 15, value=5, step=1, label="Duration (minutes)")
                            two_hosts_cb = gr.Checkbox(label="Two-host dialogue (Alex + Jordan)", value=False)
                            gen_audio_cb = gr.Checkbox(label="Generate MP3 audio", value=False)
                        gen_podcast_btn = gr.Button("🎙️ Generate Podcast", variant="primary")
                        podcast_status = gr.Markdown("")
                        podcast_transcript = gr.Textbox(label="Transcript", lines=15, interactive=False)
                        podcast_tx_file = gr.File(label="Download Transcript")
                        podcast_audio = gr.Audio(label="🎧 Listen", type="filepath")
                        podcast_list = gr.Dropdown(label="Past Podcasts", choices=[], interactive=True)
                        load_podcast_btn = gr.Button("📂 Load Selected")

        # Wire events
        demo.load(refresh_nbs, outputs=[nb_dropdown, nb_dropdown])

        create_btn.click(create_nb, [nb_name_input], [nb_dropdown, nb_dropdown, nb_status])
        rename_btn.click(rename_nb, [nb_dropdown, rename_input], [nb_dropdown, nb_dropdown, nb_status])
        delete_btn.click(delete_nb, [nb_dropdown], [nb_dropdown, nb_dropdown, nb_status])

        nb_dropdown.change(load_sources, [nb_dropdown], [sources_list])
        nb_dropdown.change(load_history, [nb_dropdown], [chatbot])

        ingest_btn.click(ingest_files, [file_upload, url_input, chunk_strategy, nb_dropdown], [ingest_status, sources_list])
        del_src_btn.click(delete_source_cb, [source_to_delete, nb_dropdown], [del_src_status, sources_list])

        send_btn.click(send_message, [chat_input, chatbot, retrieve_technique, sources_list, nb_dropdown], [chatbot, chat_input, chat_status])
        chat_input.submit(send_message, [chat_input, chatbot, retrieve_technique, sources_list, nb_dropdown], [chatbot, chat_input, chat_status])
        clear_btn.click(clear_chat_cb, [nb_dropdown], [chatbot, chat_status])

        gen_report_btn.click(gen_report, [report_extra, sources_list, nb_dropdown], [report_output, report_file, report_list])
        load_report_btn.click(load_artifact, [report_list], [report_output, report_file])

        gen_quiz_btn.click(gen_quiz, [quiz_extra, sources_list, nb_dropdown], [quiz_output, quiz_file, quiz_list])
        load_quiz_btn.click(load_artifact, [quiz_list], [quiz_output, quiz_file])

        gen_podcast_btn.click(gen_podcast, [podcast_extra, two_hosts_cb, podcast_duration, gen_audio_cb, sources_list, nb_dropdown], [podcast_status, podcast_transcript, podcast_tx_file, podcast_audio, podcast_list])
        load_podcast_btn.click(load_artifact, [podcast_list], [podcast_transcript, podcast_tx_file])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", 7860)),
        auth=_authenticate,
        auth_message="Sign in to NotebookLM Clone.",
        share=False,
        theme=gr.themes.Soft(),
        css="#chat-panel{min-height:420px}",
    )