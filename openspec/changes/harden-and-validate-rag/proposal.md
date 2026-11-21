# Harden and validate PDF RAG

## Why
The Streamlit + FastEmbed/FAISS + Groq PDF assistant has modular components and session-aware indexing, but static review identified gaps in upload validation, safe error handling, reproducible integration validation, and evidence-backed answers. A functioning UI is not sufficient evidence of safety or RAG quality.

## What changes
- Validate uploaded PDFs before ingestion: size, PDF signature, parsing failures and empty-text documents; apply safe resource limits and user-facing error categories.
- Sanitize errors shown in Streamlit; log diagnostic context without secrets or raw document contents.
- Add real-PDF ingestion / FAISS integration tests, mocked LLM tests, session and document isolation tests, and clean dependency/CI validation.
- Track source document, page and retrieved excerpt for answers; distinguish retrieved evidence from verified claims and define abstention behavior.
- Keep OpenSpec task status synchronized with test evidence.

## In scope
Streamlit application, PDF ingestion, FAISS retriever, Groq adapter, tests, GitHub Actions, docs. No provider credentials needed in CI.

## Out of scope
FastAPI, Qdrant, DuckDB, Langfuse, authentication, distributed scale, live paid Groq tests, or automatic PR merge.

## Success
Security checks, deterministic installation and tests, source traceability and evaluation cases demonstrated by reproducible evidence. Baseline behavior is preserved.