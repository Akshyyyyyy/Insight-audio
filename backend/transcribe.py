#LOADING WHISPER TO TRANSCRIBE THE AUDIO FILE TO TEXT TRANSCRIBING
import whisper
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# Load Whisper model globally (so it doesn't reload every call)
model = whisper.load_model("small")  # Changed to "small" for faster processing

def transcribe_audio(audio_path, output_path):
    """
    Transcribes audio file and saves transcript as JSON with timestamps.
    """

    print("Transcribing audio...")

    result = model.transcribe(audio_path)

    transcript_data = []

    for segment in result["segments"]:
        transcript_data.append({
            "segment_id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save transcript JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4)

    print("Transcription saved successfully.")
    return transcript_data