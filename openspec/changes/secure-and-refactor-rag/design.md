# Design

- Streamlit owns user interaction and session state.
- ingestion.py loads PDF and deletes temporary files in finally.
- retrieval.py constructs an in-memory FAISS index per session/document.
- llm.py constructs a configurable Groq client.
- config.py validates required environment variables.
- The SHA-256 digest of PDF bytes identifies the current document for session resets.
- Source page numbers reflect retrieved pages, not a guarantee that every claim is supported.
- Do not persist the vector index to a shared filesystem path.
