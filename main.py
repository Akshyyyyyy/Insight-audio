import os

from backend.transcribe import transcribe_audio
from backend.chunking import create_chunks
from backend.embedding import generate_embeddings, save_embeddings_and_metadata
from backend.faiss_index import build_faiss_index, save_faiss_index
from backend.retrieval import search_query
from backend.summarizer import summarize_transcript
from backend.quiz import quiz_mode


SUPPORTED_FORMATS = [".wav", ".mp3", ".m4a"]


def index_audio(audio_file):
    """
    Full indexing pipeline.
    Run ONLY when new audio is added.
    """

    # Format validation
    if not any(audio_file.endswith(ext) for ext in SUPPORTED_FORMATS):
        print("Unsupported file format.")
        print("Supported formats:", ", ".join(SUPPORTED_FORMATS))
        return

    if not os.path.exists(audio_file):
        print("File not found.")
        return

    output_file = "data/transcripts/test_transcript.json"

    transcript = transcribe_audio(audio_file, output_file)

    chunks = create_chunks(transcript, max_words=100)
    print("Chunks created:", len(chunks))

    embeddings = generate_embeddings(chunks)
    print("Embedding shape:", embeddings.shape)

    save_embeddings_and_metadata(embeddings, chunks)

    index = build_faiss_index(embeddings)
    save_faiss_index(index)

    print("Indexing completed successfully.")


def interactive_mode():
    """
    RAG Q&A mode
    """
    print("\nRAG system ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask your question: ")

        if query.lower() == "exit":
            break

        search_query(query, top_k=3)


if __name__ == "__main__":

    while True:

        print("\n==============================")
        print("Podcast AI System")
        print("==============================")
        print("1 - Index new audio")
        print("2 - Ask questions (RAG)")
        print("3 - Generate podcast summary")
        print("4 - Podcast Quiz Mode")
        print("5 - Exit")
        print("==============================")

        choice = input("Enter your choice: ")

        if choice == "1":
            print("\nSupported formats:", ", ".join(SUPPORTED_FORMATS))
            path = input("Enter audio file path: ")
            index_audio(path)

        elif choice == "2":
            interactive_mode()

        elif choice == "3":
            summarize_transcript()

        elif choice == "4":
            quiz_mode()

        elif choice == "5" or choice.lower() == "exit":
            print("Exiting system. Goodbye.")
            break

        else:
            print("Invalid choice. Try again.")

