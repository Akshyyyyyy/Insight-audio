# InSight Audio: Podcast AI System 🎙️🧠

An Explainable AI platform that transforms passive podcast listening into active knowledge. This CLI-based application makes unstructured audio searchable, summarizable, and interactive using a local Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Key Features

* **Local Transcription:** Converts `.wav`, `.mp3`, and `.m4a` files into highly accurate timestamped text using OpenAI's Whisper model.
* **Explainable RAG (Smart Search):** Ask questions about the podcast and get exact answers powered by FAISS vector search, with confidence scores to prevent hallucinations.
* **Automated Summarization:** Instantly generates concise, 5-8 sentence summaries of entire episodes using local LLMs.
* **Active Learning (Quiz Mode):** Automatically reads the transcript and generates interactive multiple-choice questions to test your knowledge retention.
* **Fully Local & Private:** Utilizes Ollama and SentenceTransformers for all generation and embedding tasks, ensuring no sensitive data leaves your machine.

## 🛠️ Tech Stack

* **Language:** Python
* **LLM Engine:** Ollama (`qwen2.5:7b`)
* **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Inner Product / Cosine Similarity)
* **Transcription:** Whisper (`small` model)

## 📁 Repository Structure

```text
├── data/
│   ├── transcripts/      # Raw transcribed JSON files
│   ├── embeddings/       # Saved embeddings and chunk metadata
│   └── faiss_index/      # Compiled FAISS vector indexes
├── backend/
│   ├── transcribe.py     # Whisper integration for audio processing
│   ├── chunking.py       # Timestamp-aware transcript segmenting
│   ├── embedding.py      # Vector generation using MiniLM
│   ├── faiss_index.py    # Vector database building and storage
│   ├── retrieval.py      # Semantic search and context retrieval
│   ├── generator.py      # LLM prompting for strict RAG answers
│   ├── summarizer.py     # Full transcript summarization logic
│   ├── quiz.py           # MCQ generation based on transcript context
│   └── models.py         # Centralized model loading
└── main.py               # Interactive CLI entry point