import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from backend.ingest import ingest_file, ingest_url
from backend.chat import chat_with_sources
from backend.artifacts import generate_report, generate_quiz, generate_podcast
from backend.storage import (
    create_notebook_for_user,
    get_user_notebooks,
    load_chat_history,
    save_message,
)

# Helpers — all data is now keyed by HF username

def get_username(profile: gr.OAuthProfile | None) -> str:
    """Return HF username, or 'guest' if not logged in."""
    if profile is None:
        return "guest"
    return profile.username


# Notebook actions

def create_notebook(name: str, profile: gr.OAuthProfile | None):
    username = get_username(profile)
    nb_id = create_notebook_for_user(username, name)
    notebooks = get_user_notebooks(username)
    choices = [nb["name"] for nb in notebooks]
    return gr.update(choices=choices, value=name), f"Created notebook: {name}"


def load_notebooks(profile: gr.OAuthProfile | None):
    """Populate notebook dropdown when user logs in."""
    username = get_username(profile)
    notebooks = get_user_notebooks(username)
    choices = [nb["name"] for nb in notebooks]
    return gr.update(choices=choices)


def switch_notebook(name: str, profile: gr.OAuthProfile | None):
    username = get_username(profile)
    notebooks = get_user_notebooks(username)
    for nb in notebooks:
        if nb["name"] == name:
            history = load_chat_history(username, nb["id"])
            # Convert stored messages to Gradio chatbot format [[user, assistant], ...]
            pairs = []
            for i in range(0, len(history) - 1, 2):
                user_msg = history[i]["content"]
                asst_msg = history[i + 1]["content"] if i + 1 < len(history) else ""
                pairs.append([user_msg, asst_msg])
            return nb["id"], f"Switched to: {name}", pairs
    return None, "Notebook not found.", []

# Ingestion

def handle_file_upload(files, nb_id: str, profile: gr.OAuthProfile | None):
    username = get_username(profile)
    if not nb_id:
        return "Please create or select a notebook first."
    results = []
    for f in files:
        result = ingest_file(f.name, nb_id, username)
        results.append(result)
    return "\n".join(results)


def handle_url_ingest(url: str, nb_id: str, profile: gr.OAuthProfile | None):
    username = get_username(profile)
    if not nb_id:
        return "Please create or select a notebook first."
    return ingest_url(url, nb_id, username)


# Chat

def handle_chat(message: str, history: list, nb_id: str, profile: gr.OAuthProfile | None):
    username = get_username(profile)
    if not nb_id:
        return history + [[message, "Please create or select a notebook first."]], ""
    response = chat_with_sources(message, nb_id, username, history)
    save_message(username, nb_id, "user", message)
    save_message(username, nb_id, "assistant", response)
    history.append([message, response])
    return history, ""


# Artifacts

def handle_generate_report(nb_id: str, profile: gr.OAuthProfile | None):
    username = get_username(profile)
    if not nb_id:
        return "Please create or select a notebook first."
    return generate_report(nb_id, username)


def handle_generate_quiz(nb_id: str, profile: gr.OAuthProfile | None):
    username = get_username(profile)
    if not nb_id:
        return "Please create or select a notebook first."
    return generate_quiz(nb_id, username)


def handle_generate_podcast(nb_id: str, profile: gr.OAuthProfile | None):
    username = get_username(profile)
    if not nb_id:
        return "Please create or select a notebook first.", None
    return generate_podcast(nb_id, username)


# Gradio UI

with gr.Blocks(title="NotebookLM Clone") as demo:
    # Stores the currently selected notebook ID in browser state
    current_nb_id = gr.State(None)

    with gr.Row():
        gr.Markdown("# NotebookLM Clone")
        # HF OAuth login button — appears automatically on HF Spaces
        login_btn = gr.LoginButton()

    with gr.Row():
        # ── Left sidebar ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Notebooks")
            notebook_dropdown = gr.Dropdown(
                label="Select Notebook", choices=[], interactive=True
            )
            new_notebook_name = gr.Textbox(
                label="New Notebook Name", placeholder="My Notebook"
            )
            create_btn = gr.Button("+ New Notebook")
            notebook_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Sources")
            file_upload = gr.File(
                label="Upload Files (.pdf .pptx .txt)", file_count="multiple"
            )
            upload_btn = gr.Button("Upload")
            url_input = gr.Textbox(label="Ingest URL", placeholder="https://...")
            url_btn = gr.Button("Add URL")
            ingest_status = gr.Textbox(label="Ingestion Status", interactive=False)

        # ── Main panel ────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(label="Chat with your sources", height=400)
                    chat_input = gr.Textbox(label="Ask a question...")
                    chat_btn = gr.Button("Send")

                with gr.Tab("Artifacts"):
                    with gr.Tabs():
                        with gr.Tab("Report"):
                            report_btn = gr.Button("Generate Report")
                            report_output = gr.Markdown()
                        with gr.Tab("Quiz"):
                            quiz_btn = gr.Button("Generate Quiz")
                            quiz_output = gr.Markdown()
                        with gr.Tab("Podcast"):
                            podcast_btn = gr.Button("Generate Podcast")
                            podcast_transcript = gr.Markdown()
                            podcast_audio = gr.Audio(label="Podcast Audio")

    # ── Event wiring ──────────────────────────────────────────────

    # Load user's notebooks after login
    demo.load(load_notebooks, inputs=None, outputs=[notebook_dropdown])

    create_btn.click(
        create_notebook,
        inputs=[new_notebook_name],
        outputs=[notebook_dropdown, notebook_status],
    )
    notebook_dropdown.change(
        switch_notebook,
        inputs=[notebook_dropdown],
        outputs=[current_nb_id, notebook_status, chatbot],
    )
    upload_btn.click(
        handle_file_upload,
        inputs=[file_upload, current_nb_id],
        outputs=[ingest_status],
    )
    url_btn.click(
        handle_url_ingest,
        inputs=[url_input, current_nb_id],
        outputs=[ingest_status],
    )
    chat_btn.click(
        handle_chat,
        inputs=[chat_input, chatbot, current_nb_id],
        outputs=[chatbot, chat_input],
    )
    report_btn.click(
        handle_generate_report, inputs=[current_nb_id], outputs=[report_output]
    )
    quiz_btn.click(
        handle_generate_quiz, inputs=[current_nb_id], outputs=[quiz_output]
    )
    podcast_btn.click(
        handle_generate_podcast,
        inputs=[current_nb_id],
        outputs=[podcast_transcript, podcast_audio],
    )

if __name__ == "__main__":
    demo.launch()