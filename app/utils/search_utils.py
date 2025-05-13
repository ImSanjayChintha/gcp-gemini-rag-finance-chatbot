from utils.embedding_utils import embed_text_chunks
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_similar_chunks(query, chunks, vectors, model, top_k=3):
    # Embed the query (returns 2-tuple; we only need the embedding)
    _, query_vecs = embed_text_chunks([query], model)
    query_vec = query_vecs[0].reshape(1, -1)

    # Cosine similarity
    similarities = cosine_similarity(query_vec, vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]