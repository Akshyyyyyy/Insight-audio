import json
import ollama


def summarize_transcript():

    transcript_path = "data/transcripts/test_transcript.json"

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except FileNotFoundError:
        print("No transcript found. Please index audio first.")
        return "No transcript found. Please process an audio file first."

    # Handle both list-of-dicts (whisper) and list-of-strings cases if any
    if isinstance(segments, list) and len(segments) > 0:
        if isinstance(segments[0], dict) and "text" in segments[0]:
            full_text = " ".join([segment["text"] for segment in segments])
        else:
            full_text = " ".join(str(s) for s in segments)
    else:
        full_text = str(segments)

    # Truncate for context window if needed (rough char count)
    if len(full_text) > 15000:
        full_text = full_text[:15000] + "..."

    prompt = f"""
You are an expert podcast summarizer.

Provide a clear and concise summary of the following transcript.
Use 5-8 sentences.
Do not add outside knowledge.

Transcript:
{full_text}

Summary:
"""

    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "top_p": 1
            }
        )
        summary = response["message"]["content"].strip()
        print("\nPodcast Summary:\n")
        print(summary)
        return summary
        
    except Exception as e:
        print(f"Summarization error: {e}")
        return f"Error generating summary: {str(e)}"
