# #LOADING WHISPER TO TRANSCRIBE THE AUDIO FILE TO TEXT TRANSCRIBING
# import whisper
# import json
# import os
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss


# # Load Whisper model globally (so it doesn't reload every call)
# model = whisper.load_model("medium")  # change to "small" if slow
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# def transcribe_audio(audio_path, output_path):
#     """
#     Transcribes audio file and saves transcript as JSON with timestamps.
#     """

#     print("Transcribing audio...")

#     result = model.transcribe(audio_path)

#     transcript_data = []

#     for segment in result["segments"]:
#         transcript_data.append({
#             "segment_id": segment["id"],
#             "start": segment["start"],
#             "end": segment["end"],
#             "text": segment["text"].strip()
#         })

#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # Save transcript JSON
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(transcript_data, f, indent=4)

#     print("Transcription saved successfully.")
#     return transcript_data


# #CHUNKING THE TRANSCRIBED DATA

# def create_chunks(transcript_data, max_words=150):
#     """
#     Create chunks from timestamped transcript segments.
#     Each chunk will preserve start and end timestamps.
#     """
    
#     chunks = []
#     current_chunk_text = []
#     current_start_time = None
#     word_count = 0
#     chunk_id = 0

#     for segment in transcript_data:
#         segment_text = segment["text"].strip()
#         segment_words = segment_text.split()
#         segment_word_count = len(segment_words)

#         # If this is the first segment in a new chunk
#         if current_start_time is None:
#             current_start_time = segment["start"]

#         current_chunk_text.append(segment_text)
#         word_count += segment_word_count

#         # If chunk reached max word limit
#         if word_count >= max_words:
#             chunk_id += 1

#             chunks.append({
#                 "chunk_id": chunk_id,
#                 "start": current_start_time,
#                 "end": segment["end"],
#                 "text": " ".join(current_chunk_text)
#             })

#             # Reset for next chunk
#             current_chunk_text = []
#             current_start_time = None
#             word_count = 0

#     # Add remaining text as final chunk
#     if current_chunk_text:
#         chunk_id += 1
#         chunks.append({
#             "chunk_id": chunk_id,
#             "start": current_start_time,
#             "end": transcript_data[-1]["end"],
#             "text": " ".join(current_chunk_text)
#         })

#     return chunks

# #EMBEDDING THE CHUNKED DATA

# def generate_embeddings(chunks):
#     """
#     Generate embeddings for chunk texts.
#     """

#     texts = [chunk["text"] for chunk in chunks]

#     print("Generating embeddings...")

#     embeddings = embedding_model.encode(texts)

#     embeddings = np.array(embeddings).astype("float32")

#     # Normalize for cosine similarity
#     faiss.normalize_L2(embeddings)

#     return embeddings

# def save_embeddings_and_metadata(embeddings, chunks):
#     """
#     Save embeddings and metadata to disk.
#     """

#     os.makedirs("data/embeddings", exist_ok=True)

#     np.save("data/embeddings/embeddings.npy", embeddings)

#     with open("data/embeddings/chunks_metadata.json", "w", encoding="utf-8") as f:
#         json.dump(chunks, f, indent=4)

#     print("Embeddings and metadata saved.")



# #FAISS VECTOR DATABASE

# def build_faiss_index(embeddings):
#     """
#     Build FAISS index using Inner Product (cosine similarity on normalized vectors).
#     """

#     dimension = embeddings.shape[1]

#     index = faiss.IndexFlatIP(dimension)

#     index.add(embeddings)

#     print("FAISS index built.")
#     print("Total vectors in index:", index.ntotal)

#     return index

# def save_faiss_index(index):
#     os.makedirs("data/faiss_index", exist_ok=True)
#     faiss.write_index(index, "data/faiss_index/podcast.index")
#     print("FAISS index saved.")


# #SIMILARITY SEARCH

# def search_query(query_text, top_k=3):
#     # Load FAISS index
#     index = faiss.read_index("data/faiss_index/podcast.index")

#     # Load metadata
#     with open("data/embeddings/chunks_metadata.json", "r", encoding="utf-8") as f:
#         chunks = json.load(f)

#     # Embed query
#     query_embedding = embedding_model.encode([query_text])
#     query_embedding = np.array(query_embedding).astype("float32")

#     faiss.normalize_L2(query_embedding)

#     # Search
#     distances, indices = index.search(query_embedding, top_k)

#     print("\nTop matches:")
#     for i, idx in enumerate(indices[0]):
#         print(f"\nRank {i+1}")
#         print("Score:", distances[0][i])
#         print("Start:", chunks[idx]["start"])
#         print("End:", chunks[idx]["end"])
#         print("Text:", chunks[idx]["text"])





