from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

# Load model once when module is imported
from backend.models import embedding_model




def generate_embeddings(chunks):
    """
    Generate embeddings for chunk texts.
    """

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = embedding_model.encode(texts)

    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    return embeddings

def save_embeddings_and_metadata(embeddings, chunks):
    """
    Save embeddings and metadata to disk.
    """

    os.makedirs("data/embeddings", exist_ok=True)

    np.save("data/embeddings/embeddings.npy", embeddings)

    with open("data/embeddings/chunks_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

    print("Embeddings and metadata saved.")