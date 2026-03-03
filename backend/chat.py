from backend.retrieval import get_retriever

# from openai import OpenAI
# client = OpenAI()

def chat_with_sources(message: str, nb_id: str, history: list) -> str:
    retrieve = get_retriever(nb_id)
    chunks, metas = retrieve(message)
    context = "\n\n".join(chunks)
    citations = ", ".join(
        f"[Source: {m.get('source','?')} chunk {m.get('chunk','?')}]" for m in metas
    )

    # TODO: Replace stub with real OpenAI call
    # system_prompt = (
    #     "You are a helpful research assistant. Answer using ONLY the context below. "
    #     "Cite sources inline.\n\nContext:\n" + context
    # )
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         *[{"role": "user" if i%2==0 else "assistant", "content": c}
    #           for i, (_, c) in enumerate(history)],
    #         {"role": "user", "content": message},
    #     ]
    # )
    # return response.choices[0].message.content + "\n\n" + citations

    return f"[STUB] Answer to: '{message}'\n\nSources: {citations}"