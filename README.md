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

## ⚙️ Installation & Setup

Follow these steps to set up and run the Podcast AI System on your local machine.

### Prerequisites
* **Python 3.10+**
* **FFmpeg:** Required by OpenAI's Whisper model for audio processing[cite: 11]. 
  * *Windows:* Install via Chocolatey (`choco install ffmpeg`) or download directly and add to your System PATH.
  * *Mac:* `brew install ffmpeg`

### 1. Clone the Repository
```bash
git clone [https://github.com/Akshyyyyyy/Insight-audio.git](https://github.com/Akshyyyyyy/Insight-audio.git)
cd Insight-audio

2. Set Up a Virtual Environment & Dependencies

Create and activate the virtual environment, then install the packages listed in your requirements file:
# Create the environment
python -m venv podcast_env

# Activate the environment
# On Windows (PowerShell):
.\podcast_env\Scripts\Activate.ps1
# On Mac/Linux:
source podcast_env/bin/activate

# Install required packages
pip install -r requirements.txt

3. Set Up Local AI Models
This system runs entirely locally for privacy and cost-efficiency.  
Ollama (LLM Engine): Download and install Ollama. Once running, pull the Qwen model used for generating answers and summaries: 

ollama pull qwen2.5:7b

Whisper & SentenceTransformers: The Whisper small model and the all-MiniLM-L6-v2 embedding model will automatically download to your machine the first time you run the application

4. Run the Application
Start the interactive CLI menu:
python main.py