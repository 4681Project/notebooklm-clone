import gradio as gr
from backend.ingest import ingest_file, ingest_url
from backend.chat import chat_with_sources
from backend.artifacts import generate_report, generate_quiz, generate_podcast
import uuid

notebooks = {}
current_notebook = {"id": None}

def create_notebook(name):
    nb_id = str(uuid.uuid4())
    notebooks[nb_id] = {"name": name, "sources": []}
    current_notebook["id"] = nb_id
    choices = [nb["name"] for nb in notebooks.values()]
    return gr.update(choices=choices, value=name), f"Created notebook: {name}"

def switch_notebook(name):
    for nb_id, nb in notebooks.items():
        if nb["name"] == name:
            current_notebook["id"] = nb_id
            sources = nb.get("sources", [])
            return f"Switched to: {name}", chr(10).join(sources) or "No sources yet."
    return "Notebook not found.", ""

def handle_file_upload(files):
    nb_id = current_notebook["id"]
    if not nb_id:
        return "Please create or select a notebook first."
    results = []
    for f in files:
        result = ingest_file(f.name, nb_id)
        notebooks[nb_id]["sources"].append(f.name)
        results.append(result)
    return chr(10).join(results)

def handle_url_ingest(url):
    nb_id = current_notebook["id"]
    if not nb_id:
        return "Please create or select a notebook first."
    result = ingest_url(url, nb_id)
    notebooks[nb_id]["sources"].append(url)
    return result

def handle_chat(message, history):
    nb_id = current_notebook["id"]
    if not nb_id:
        return history + [[message, "Please create or select a notebook first."]]
    response = chat_with_sources(message, nb_id, history)
    history.append([message, response])
    return history

def handle_generate_report():
    nb_id = current_notebook["id"]
    if not nb_id:
        return "Please create or select a notebook first."
    return generate_report(nb_id)

def handle_generate_quiz():
    nb_id = current_notebook["id"]
    if not nb_id:
        return "Please create or select a notebook first."
    return generate_quiz(nb_id)

def handle_generate_podcast():
    nb_id = current_notebook["id"]
    if not nb_id:
        return "Please create or select a notebook first.", None
    return generate_podcast(nb_id)

with gr.Blocks(title="NotebookLM Clone") as demo:
    gr.Markdown("# NotebookLM Clone")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Notebooks")
            notebook_dropdown = gr.Dropdown(label="Select Notebook", choices=[], interactive=True)
            new_notebook_name = gr.Textbox(label="New Notebook Name", placeholder="My Notebook")
            create_btn = gr.Button("+ New Notebook")
            notebook_status = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("### Sources")
            file_upload = gr.File(label="Upload Files (.pdf .pptx .txt)", file_count="multiple")
            upload_btn = gr.Button("Upload")
            url_input = gr.Textbox(label="Ingest URL", placeholder="https://...")
            url_btn = gr.Button("Add URL")
            ingest_status = gr.Textbox(label="Ingestion Status", interactive=False)
            sources_display = gr.Textbox(label="Ingested Sources", interactive=False, lines=5)
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

    create_btn.click(create_notebook, inputs=[new_notebook_name], outputs=[notebook_dropdown, notebook_status])
    notebook_dropdown.change(switch_notebook, inputs=[notebook_dropdown], outputs=[notebook_status, sources_display])
    upload_btn.click(handle_file_upload, inputs=[file_upload], outputs=[ingest_status])
    url_btn.click(handle_url_ingest, inputs=[url_input], outputs=[ingest_status])
    chat_btn.click(handle_chat, inputs=[chat_input, chatbot], outputs=[chatbot])
    report_btn.click(handle_generate_report, outputs=[report_output])
    quiz_btn.click(handle_generate_quiz, outputs=[quiz_output])
    podcast_btn.click(handle_generate_podcast, outputs=[podcast_transcript, podcast_audio])

if __name__ == "__main__":
    demo.launch()