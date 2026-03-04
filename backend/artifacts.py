"""
artifacts.py
------------
Artifact generation: report (.md), quiz (.md), podcast transcript + audio (.mp3).

Each generator:
  1. Retrieves the full text corpus for the notebook
  2. Prompts the LLM to produce the artifact
  3. Saves the result via storage and returns the file path
"""

import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from backend import storage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Shared: build a combined corpus from extracted text files
# ---------------------------------------------------------------------------

def _build_corpus(username: str, notebook_id: str, enabled_sources: Optional[list[str]] = None) -> str:
    """Concatenate all extracted text files (optionally filtered) into one corpus."""
    extracted = storage.list_extracted(username, notebook_id)
    parts = []
    for path in extracted:
        if enabled_sources is not None:
            # match by stem or full name
            if not any(path.stem in src or path.name in src for src in enabled_sources):
                continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        parts.append(f"=== {path.stem} ===\n{text}")
    return "\n\n".join(parts)


def _llm(prompt: str, system: str = "", max_tokens: int = 2048, temperature: float = 0.4) -> str:
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = """\
You are an expert research analyst. Write clear, well-structured Markdown reports
grounded in the provided source material. Use headings, bullet points, and
conclude with a summary. Do not invent information not in the sources."""

_REPORT_PROMPT = """\
Based on the following source material, write a comprehensive research report.
{extra_instructions}

--- SOURCES ---
{corpus}
--- END SOURCES ---

The report should include:
- Executive Summary
- Key Findings (with section headings)
- Conclusion

Use Markdown formatting."""


def generate_report(
    username: str,
    notebook_id: str,
    extra_instructions: str = "",
    enabled_sources: Optional[list[str]] = None,
) -> Path:
    corpus = _build_corpus(username, notebook_id, enabled_sources)
    if not corpus.strip():
        raise ValueError("No source content available to generate a report.")

    extra = f"\nAdditional focus: {extra_instructions}" if extra_instructions else ""
    prompt = _REPORT_PROMPT.format(corpus=corpus[:12000], extra_instructions=extra)
    md = _llm(prompt, system=_REPORT_SYSTEM, max_tokens=2500)
    path = storage.save_artifact(username, notebook_id, "reports", md, "md")
    return path


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

_QUIZ_SYSTEM = """\
You are an expert educator. Create rigorous, thoughtful multiple-choice quizzes
from academic source material. Each question must have exactly 4 options (A-D)
and one correct answer. Include an answer key at the end."""

_QUIZ_PROMPT = """\
Based on the following source material, create a 10-question multiple-choice quiz.
{extra_instructions}

--- SOURCES ---
{corpus}
--- END SOURCES ---

Format each question as:
**Q1. [Question text]**
A) [Option]
B) [Option]
C) [Option]
D) [Option]

After all questions, add:
## Answer Key
Q1: [Letter] - [Brief explanation]
...

Use Markdown formatting."""


def generate_quiz(
    username: str,
    notebook_id: str,
    extra_instructions: str = "",
    enabled_sources: Optional[list[str]] = None,
) -> Path:
    corpus = _build_corpus(username, notebook_id, enabled_sources)
    if not corpus.strip():
        raise ValueError("No source content available to generate a quiz.")

    extra = f"\nFocus: {extra_instructions}" if extra_instructions else ""
    prompt = _QUIZ_PROMPT.format(corpus=corpus[:12000], extra_instructions=extra)
    md = _llm(prompt, system=_QUIZ_SYSTEM, max_tokens=2500)
    path = storage.save_artifact(username, notebook_id, "quizzes", md, "md")
    return path


# ---------------------------------------------------------------------------
# Podcast transcript (single host or two-host dialogue)
# ---------------------------------------------------------------------------

_PODCAST_SYSTEM_SINGLE = """\
You are a scriptwriter for an educational podcast. Write engaging, conversational
scripts in the voice of a single knowledgeable host who explains complex topics
clearly to a general audience."""

_PODCAST_SYSTEM_DUAL = """\
You are a scriptwriter for an educational podcast with two hosts:
- HOST_A: Alex — enthusiastic, asks clarifying questions
- HOST_B: Jordan — the expert, provides detailed explanations

Write natural dialogue where they discuss the topic together."""

_PODCAST_PROMPT = """\
Write a {duration}-minute podcast {style} about the following source material.
{extra_instructions}

--- SOURCES ---
{corpus}
--- END SOURCES ---

The podcast should:
- Have a clear intro, main discussion, and outro
- Be engaging and accessible to a general audience
- Cover the most important points from the sources
- Sound natural when read aloud

{format_hint}"""


def generate_podcast_transcript(
    username: str,
    notebook_id: str,
    duration_minutes: int = 5,
    two_hosts: bool = False,
    extra_instructions: str = "",
    enabled_sources: Optional[list[str]] = None,
) -> Path:
    corpus = _build_corpus(username, notebook_id, enabled_sources)
    if not corpus.strip():
        raise ValueError("No source content available to generate a podcast.")

    style = "dialogue between two hosts (Alex and Jordan)" if two_hosts else "monologue"
    fmt_hint = (
        "Format each line as:\nALEX: [speech]\nJORDAN: [speech]"
        if two_hosts
        else "Write as continuous narrative prose."
    )
    system = _PODCAST_SYSTEM_DUAL if two_hosts else _PODCAST_SYSTEM_SINGLE
    extra = f"\nAdditional focus: {extra_instructions}" if extra_instructions else ""

    prompt = _PODCAST_PROMPT.format(
        duration=duration_minutes,
        style=style,
        corpus=corpus[:10000],
        extra_instructions=extra,
        format_hint=fmt_hint,
    )
    transcript = _llm(prompt, system=system, max_tokens=3000, temperature=0.6)
    path = storage.save_artifact(username, notebook_id, "podcasts", transcript, "md")
    return path


# ---------------------------------------------------------------------------
# Podcast audio generation (TTS)
# ---------------------------------------------------------------------------

def generate_podcast_audio(
    username: str,
    notebook_id: str,
    transcript_path: Path,
    two_hosts: bool = False,
) -> Path:
    """
    Convert a podcast transcript to MP3 audio using OpenAI TTS.
    For two-host scripts, alternate voices per speaker line.
    Single-host scripts use one voice for the full text.
    """
    client = _get_client()
    transcript = transcript_path.read_text(encoding="utf-8", errors="ignore")

    if two_hosts:
        audio_segments = _tts_two_hosts(client, transcript)
    else:
        audio_segments = [_tts_single(client, transcript, voice="alloy")]

    # Concatenate all PCM/MP3 segments
    combined = b"".join(audio_segments)
    path = storage.save_artifact(username, notebook_id, "podcasts", combined, "mp3")
    return path


def _clean_for_tts(text: str) -> str:
    """Strip markdown formatting that sounds bad when spoken."""
    text = re.sub(r'\*+', '', text)        # bold/italic
    text = re.sub(r'#+\s*', '', text)      # headings
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # links
    text = re.sub(r'`+', '', text)         # code ticks
    text = re.sub(r'\n{2,}', '\n', text)   # extra newlines
    return text.strip()


def _tts_single(client: OpenAI, text: str, voice: str = "alloy") -> bytes:
    clean = _clean_for_tts(text)
    # OpenAI TTS has a ~4096 char limit per request — chunk if needed
    chunks = [clean[i:i+4000] for i in range(0, len(clean), 4000)]
    audio = b""
    for chunk in chunks:
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=chunk,
            response_format="mp3",
        )
        audio += resp.content
    return audio


def _tts_two_hosts(client: OpenAI, transcript: str) -> list[bytes]:
    """Parse ALEX/JORDAN lines and assign different TTS voices."""
    lines = transcript.splitlines()
    segments = []
    voice_map = {"ALEX": "alloy", "JORDAN": "nova"}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        for speaker, voice in voice_map.items():
            if line.upper().startswith(f"{speaker}:"):
                speech = line[len(speaker) + 1:].strip()
                if speech:
                    segments.append(_tts_single(client, speech, voice=voice))
                break
        else:
            # Narrator or untagged line — use default voice
            if line:
                segments.append(_tts_single(client, line, voice="alloy"))

    return segments