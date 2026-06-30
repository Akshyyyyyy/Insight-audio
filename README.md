# 🎙️ Podcast Intelligence System

An AI-powered Podcast Intelligence System that transforms podcast audio into an interactive knowledge base using Retrieval-Augmented Generation (RAG). The system automatically transcribes audio, generates semantic embeddings, performs intelligent document retrieval, answers questions using source-grounded responses, generates concise summaries, and creates quizzes from podcast content.

---

## ✨ Features

- 🎧 Automatic podcast transcription using OpenAI Whisper
- ✂️ Intelligent transcript chunking with timestamp preservation
- 🧠 Semantic embedding generation using Sentence Transformers
- ⚡ Fast vector similarity search with FAISS
- 💬 Retrieval-Augmented Generation (RAG) based Question Answering
- 📑 AI-generated podcast summaries
- 📝 Automatic multiple-choice quiz generation
- 📍 Timestamp-based source retrieval
- 📊 Similarity score and confidence estimation
- 🚫 Hallucination reduction through grounded context retrieval

---

## 🏗️ Architecture

```text
                Audio File
                     │
                     ▼
            OpenAI Whisper
                     │
                     ▼
       Timestamped Transcript
                     │
                     ▼
              Text Chunking
                     │
                     ▼
     Sentence Transformer Embeddings
                     │
                     ▼
              FAISS Vector Store
                     │
         User Question / Query
                     │
                     ▼
          Semantic Similarity Search
                     │
                     ▼
          Retrieved Transcript Chunks
                     │
                     ▼
         Qwen2.5 (via Ollama)
                     │
                     ▼
      Grounded AI Response + Timestamp
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Speech-to-Text | OpenAI Whisper |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Large Language Model | Qwen2.5 7B |
| LLM Runtime | Ollama |
| Libraries | Sentence Transformers, NumPy, Torch, Transformers, Scikit-learn |

---

## 📂 Project Structure

```text
Podcast-Intelligence-System/
│
├── backend/
│   ├── chunking.py
│   ├── embedding.py
│   ├── faiss_index.py
│   ├── generator.py
│   ├── models.py
│   ├── quiz.py
│   ├── retrieval.py
│   ├── summarizer.py
│   └── transcribe.py
│
├── data/
│   ├── transcripts/
│   ├── embeddings/
│   └── faiss_index/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Podcast-Intelligence-System.git
cd Podcast-Intelligence-System
```

### 2. (Optional) Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and Download the Model

Install Ollama from:

https://ollama.com/download

Then pull the required model:

```bash
ollama pull qwen2.5:7b
```

### 5. Run the Project

```bash
python main.py
```

---

## ▶️ Usage

After launching the application, choose from the interactive menu:

```text
==============================
Podcast AI System
==============================
1 - Index new audio
2 - Ask questions (RAG)
3 - Generate podcast summary
4 - Podcast Quiz Mode
5 - Exit
==============================
```

### Supported Audio Formats

- MP3
- WAV
- M4A

---

## 🧠 Models Used

| Task | Model |
|------|-------|
| Speech Recognition | OpenAI Whisper |
| Text Embeddings | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Language Model | Qwen2.5:7B |
| LLM Runtime | Ollama |

---

## 🔮 Future Improvements

- Streamlit Web Interface
- Multi-podcast knowledge base
- Speaker diarization
- Hybrid retrieval (BM25 + Vector Search)
- REST API
- Docker support
- Multi-language support
- Citation highlighting

---

## 👨‍💻 Author

**Akshay TP**

AI • Machine Learning • Natural Language Processing • Retrieval-Augmented Generation (RAG)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.    