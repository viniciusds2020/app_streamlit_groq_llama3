# Secure and refactor RAG

## Why
The Streamlit PDF assistant needs safer credentials, independent components, document/session isolation and regression tests.

## What changes
- Load Groq credentials from the environment; revoke the previously exposed key outside GitHub.
- Isolate PDF ingestion, vector retrieval, chat state and model integration.
- Keep Streamlit, FAISS, Sentence Transformers and Groq.
- Add tests and document setup.

## Non-goals
FastAPI, Qdrant, authentication, automatic merge.
