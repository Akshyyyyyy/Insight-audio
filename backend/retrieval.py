import json
import numpy as np
import faiss

from backend.models import embedding_model
from backend.generator import generate_answer


def search_query(query_text, top_k=3):

    index = faiss.read_index("data/faiss_index/podcast.index")

    with open("data/embeddings/chunks_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    query_embedding = embedding_model.encode([query_text])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    top_chunks = []

    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            chunk_data = chunks[idx].copy()  # important: copy to avoid mutation
            chunk_data["score"] = float(distances[0][i])
            top_chunks.append(chunk_data)

    if not top_chunks:
        return None

    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    final_answer = generate_answer(query_text, context)

    best = top_chunks[0]
    score = best["score"]

    if score > 0.4:
        confidence = "High Confidence"
    elif score > 0.25:
        confidence = "Medium Confidence"
    else:
        confidence = "Low Confidence"

    # 👇 RETURN structured data instead of printing
    return {
        "answer": final_answer,
        "similarity_score": round(score, 4),
        "confidence": confidence,
        "start": best["start"],
        "end": best["end"],
        "source_text": best["text"]
    }
