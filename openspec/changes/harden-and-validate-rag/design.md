# Design — harden-and-validate-rag

## Context
Existing components: app.py (Streamlit), ingestion.py (PyPDFLoader), retrieval.py (FastEmbedEmbeddings + FAISS), llm.py (Groq), chat_state.py and config.py. The UI currently displays raw exception strings; PDF upload relies principally on Streamlit extension filtering; retrieved pages are shown but are not claim-level verified citations.

## Decisions
1. **Validate before parsing.** Centralize PDF checks (max bytes configurable, MIME not trusted, header `%PDF-`, parser failure and empty extracted text). Apply operational limits to page count/extracted content where practical; handle malformed/encrypted documents safely. Extension alone is not validation. Choose explicit limits from tested resources; no untested numeric defaults claimed.
2. **Separate UI errors from diagnostics.** Use typed/safe error messages; structured logging should never include API keys, raw PDF text or full exception payload from external services in user-visible output.
3. **Maintain per-session, per-document isolation.** Keep SHA-256 document identification and session-scoped in-memory index; add tests for reupload, changed PDF and failure resets.
4. **Test real local components and fake external generation.** Use small licensed/synthetic PDFs, a deterministic fake LLM and actual installed FastEmbed/FAISS for integration. CI runs on pull requests and main pushes. Optional live Groq smoke tests are opt-in and isolated.
5. **Trace evidence explicitly.** Preserve metadata (document ID, page, excerpt/chunk ID) in source references; show citations tied to retrieved excerpts, not a claim of automatically verified faithfulness. A reference question set measures retrieval and answer support; if relevant evidence is missing, the assistant must clearly state insufficient information.
6. **Avoid premature migration.** No new web framework or vector database in this change.

## Alternatives and trade-offs
- Mock-only tests are fast but cannot validate embedding/index compatibility; retain both unit and real local integration tests.
- Enforcing strict file and text caps may exclude legitimate large PDFs; make limits documented and configurable.
- Citing retrieved excerpts does not prove answer correctness; evaluation and abstention policies are separately required.

## Risks
- Some malformed files pass initial header checks: still rely on safe parsing and resource limits.
- Heavy embedding model downloads can destabilize CI: cache models or pin artifacts and report download failure.
- Provider errors may contain sensitive request context: scrub logs and avoid surfacing provider payloads.

## Rollout
Incremental code PR after this spec-only branch is reviewed; keep main untouched until tests and review.