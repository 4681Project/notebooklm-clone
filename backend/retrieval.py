def get_retriever(nb_id, top_k=5):
    def retrieve(query):
        return ["[STUB] chunk for: " + query], [{"source": "stub.pdf", "chunk": 0}]
    return retrieve