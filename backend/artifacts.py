from backend.retrieval import get_retriever

# from openai import OpenAI
# client = OpenAI()

def _get_context(nb_id: str, query: str = "summarize all content") -> str:
    retrieve = get_retriever(nb_id, top_k=10)
    chunks, _ = retrieve(query)
    return "\n\n".join(chunks)


def generate_report(nb_id: str) -> str:
    context = _get_context(nb_id)

    # TODO: Replace stub with real OpenAI call
    # prompt = (
    #     "Using the source material below, write a comprehensive study report in Markdown. "
    #     "Include: title, summary, key concepts, and conclusions. Cite sources inline.\n\n"
    #     + context
    # )
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content

    return f"## [STUB] Report\n\nGenerated from notebook `{nb_id}`\n\n{context}"


def generate_quiz(nb_id: str) -> str:
    context = _get_context(nb_id)

    # TODO: Replace stub with real OpenAI call
    # prompt = (
    #     "Using the source material below, create a 10-question multiple-choice quiz in Markdown. "
    #     "Include an answer key at the end.\n\n" + context
    # )
    # response = client.chat.completions.create(...)
    # return response.choices[0].message.content

    return (
        f"## [STUB] Quiz\n\n"
        f"**Q1.** What is the main topic?\n"
        f"- A) Topic A\n- B) Topic B\n- C) Topic C\n\n"
        f"**Answer Key:** 1-A"
    )


def generate_podcast(nb_id: str):
    context = _get_context(nb_id)

    # TODO: Replace stub with real OpenAI TTS
    # prompt = (
    #     "Using the source material below, write a podcast script as a conversation "
    #     "between two hosts: Alex and Jordan. Make it engaging and educational.\n\n"
    #     + context
    # )
    # response = client.chat.completions.create(...)
    # transcript = response.choices[0].message.content
    #
    # audio_response = client.audio.speech.create(
    #     model="tts-1", voice="alloy", input=transcript
    # )
    # audio_path = f"data/users/default/notebooks/{nb_id}/artifacts/podcasts/podcast.mp3"
    # Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    # audio_response.stream_to_file(audio_path)
    # return transcript, audio_path

    transcript = (
        "## [STUB] Podcast Transcript\n\n"
        "**Alex:** Welcome to the show! Today we are covering the key topics from your sources.\n\n"
        "**Jordan:** Great! Let us dive right in..."
    )
    audio_path = None
    return transcript, audio_path