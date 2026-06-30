import whisper
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def create_chunks(transcript_data, max_words=150):
    """
    Create chunks from timestamped transcript segments.
    Each chunk will preserve start and end timestamps.
    """
    
    chunks = []
    current_chunk_text = []
    current_start_time = None
    word_count = 0
    chunk_id = 0

    for segment in transcript_data:
        segment_text = segment["text"].strip()
        segment_words = segment_text.split()
        segment_word_count = len(segment_words)

        # If this is the first segment in a new chunk
        if current_start_time is None:
            current_start_time = segment["start"]

        current_chunk_text.append(segment_text)
        word_count += segment_word_count

        # If chunk reached max word limit
        if word_count >= max_words:
            chunk_id += 1

            chunks.append({
                "chunk_id": chunk_id,
                "start": current_start_time,
                "end": segment["end"],
                "text": " ".join(current_chunk_text)
            })

            # Reset for next chunk
            current_chunk_text = []
            current_start_time = None
            word_count = 0

    # Add remaining text as final chunk
    if current_chunk_text:
        chunk_id += 1
        chunks.append({
            "chunk_id": chunk_id,
            "start": current_start_time,
            "end": transcript_data[-1]["end"],
            "text": " ".join(current_chunk_text)
        })

    return chunks