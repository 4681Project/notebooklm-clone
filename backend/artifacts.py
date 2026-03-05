import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from backend.retrieval import get_retriever
from backend.storage import save_artifact, notebook_dir

load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from environment automatically


def _get_context(nb_id: str, username: str, query: str = "summarize all content") -> str:
    """Retrieve top chunks and format them as a context string."""
    retrieve = get_retriever(nb_id, username=username, top_k=10)
    chunks, metas = retrieve(query)
    parts = []
    for chunk, meta in zip(chunks, metas):
        source = meta.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{chunk}")
    return "\n\n---\n\n".join(parts)


def _call_llm(prompt: str, temperature: float = 0.4) -> str:
    """Call GPT-4o and return the response text."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(nb_id: str, username: str) -> str:
    """Generate a comprehensive markdown study report from notebook sources."""
    try:
        context = _get_context(nb_id, username, query="summarize all key information")

        prompt = (
            "Using ONLY the source material provided below, write a comprehensive "
            "study report in Markdown format. Structure it with:\n"
            "1. A clear title (# heading)\n"
            "2. An executive summary\n"
            "3. Key concepts and themes (## headings)\n"
            "4. Important details and supporting evidence\n"
            "5. Conclusions\n\n"
            "Cite sources inline using [Source: filename] format.\n\n"
            f"Source Material:\n{context}"
        )

        report = _call_llm(prompt)
        save_artifact(username, nb_id, "reports", report, extension="md")
        return report

    except Exception as e:
        return f"Error generating report: {str(e)}"


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

def generate_quiz(nb_id: str, username: str) -> str:
    """Generate a 10-question multiple-choice quiz with answer key."""
    try:
        context = _get_context(nb_id, username, query="key facts, concepts, and details")

        prompt = (
            "Using ONLY the source material provided below, create a 10-question "
            "multiple-choice quiz in Markdown format.\n\n"
            "Format each question exactly like this:\n"
            "**Q1.** Question text here?\n"
            "- A) Option one\n"
            "- B) Option two\n"
            "- C) Option three\n"
            "- D) Option four\n\n"
            "After all 10 questions, include a section:\n"
            "## Answer Key\n"
            "1. A, 2. C, ... etc, with a one-sentence explanation for each answer.\n\n"
            f"Source Material:\n{context}"
        )

        quiz = _call_llm(prompt)
        save_artifact(username, nb_id, "quizzes", quiz, extension="md")
        return quiz

    except Exception as e:
        return f"Error generating quiz: {str(e)}"


# ---------------------------------------------------------------------------
# Podcast
# ---------------------------------------------------------------------------

def generate_podcast(nb_id: str, username: str):
    """
    Generate a two-host podcast transcript and convert it to audio.
    Returns (transcript_markdown, audio_file_path).
    """
    try:
        context = _get_context(nb_id, username, query="main topics and interesting insights")

        # Step 1: Generate transcript
        transcript_prompt = (
            "Using ONLY the source material below, write an engaging podcast script "
            "as a conversation between two hosts: Alex and Jordan.\n\n"
            "Guidelines:\n"
            "- Alex introduces topics and asks questions\n"
            "- Jordan provides deeper explanations and insights\n"
            "- Make it conversational, informative, and around 5-7 minutes when read aloud\n"
            "- Format each line as: **Alex:** text or **Jordan:** text\n"
            "- Start with a brief intro and end with a summary and sign-off\n\n"
            f"Source Material:\n{context}"
        )

        transcript = _call_llm(transcript_prompt, temperature=0.6)

        # Save transcript as markdown
        save_artifact(username, nb_id, "podcasts", transcript, extension="md")

        # Step 2: Convert transcript to audio using OpenAI TTS
        # Strip markdown formatting for cleaner audio
        clean_transcript = transcript.replace("**Alex:**", "Alex:").replace("**Jordan:**", "Jordan:")

        audio_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=clean_transcript,
        )

        # Save audio file
        podcasts_dir = notebook_dir(username, nb_id) / "artifacts" / "podcasts"
        podcasts_dir.mkdir(parents=True, exist_ok=True)
        existing = list(podcasts_dir.glob("*.mp3"))
        audio_path = str(podcasts_dir / f"podcast_{len(existing) + 1}.mp3")
        audio_response.stream_to_file(audio_path)

        return transcript, audio_path

    except Exception as e:
        error_msg = f"Error generating podcast: {str(e)}"
        return error_msg, None