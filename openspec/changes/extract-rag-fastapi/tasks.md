# Tasks — extract-rag-fastapi

## Specification and contracts
- [ ] Agree on API schemas, request IDs, error responses and session lifecycle, including session creation and upload ordering.
- [ ] Record ADR for backend-owned ephemeral state and local-only security boundary.

## Backend
- [ ] Extract framework-independent RAG service from `app.py`.
- [ ] Add FastAPI app, versioned routes, typed validation and OpenAPI documentation.
- [ ] Preserve existing PDF limits, safe error messaging and document/session isolation.
- [ ] Implement bounded in-memory session store, expiration/reset and failure cleanup.
- [ ] Bound blocking ingestion, concurrency and upload sizes; document limits.

## Frontend
- [ ] Replace direct RAG calls in Streamlit with HTTP client calls.
- [ ] Preserve chat flow, upload UX, page references and safe errors.

## Verification and delivery
- [ ] Keep existing security/regression tests green.
- [ ] Add API unit/integration tests, synthetic PDF + real FAISS/FastEmbed tests and mocked Groq behavior.
- [ ] Test cross-session isolation, reupload, invalid PDF, empty-text PDF, missing index, provider failure and timeouts.
- [ ] Update CI for backend API checks and dependency reproducibility.
- [ ] Document one-command local startup and manual end-to-end verification.
- [ ] Run local Streamlit -> FastAPI -> FAISS -> Groq live smoke test only when credentials are available; do not require secrets in public CI.
- [ ] Reconcile OpenSpec checkboxes with captured evidence before requesting implementation merge.
