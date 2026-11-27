# Extract RAG backend to FastAPI

## Why
The current Streamlit app performs PDF validation, ingestion, FAISS indexing, Groq chain construction and question answering inside `app.py`. This coupling limits independent API testing and future reuse by other clients. The existing local experience works and must remain the regression baseline.

## What changes
- Introduce a FastAPI backend with versioned HTTP endpoints for document upload, question answering, session reset and health/readiness.
- Move RAG orchestration out of Streamlit into a framework-agnostic application service used by the API.
- Convert Streamlit into an HTTP client, retaining the current user experience and safe error messages.
- Preserve PDF hardening, FAISS/FastEmbed retrieval, Groq generation and document/session isolation.
- Add API contract, integration and Streamlit-adapter tests using mocked Groq; document local execution.

## In scope
FastAPI, service layer, HTTP contracts, session lifecycle, tests, local development docs and CI adjustments.

## Out of scope
DuckDB, Qdrant, Langfuse, cloud deployment, distributed session storage, production multi-replica hosting, replacing Groq, authentication rollout and automatic merge.

## Success
An uploaded PDF can be queried through Streamlit -> FastAPI -> FAISS -> Groq with equivalent answers and safely handled failures. Local automated tests prove the API and isolation semantics; optional opt-in live Groq smoke test is documented separately.
