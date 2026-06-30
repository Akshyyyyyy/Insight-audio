import whisper
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def build_faiss_index(embeddings):
    """
    Build FAISS index using Inner Product (cosine similarity on normalized vectors).
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS index built.")
    print("Total vectors in index:", index.ntotal)

    return index

def save_faiss_index(index):
    os.makedirs("data/faiss_index", exist_ok=True)
    faiss.write_index(index, "data/faiss_index/podcast.index")
    print("FAISS index saved.")