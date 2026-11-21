# Tasks — harden-and-validate-rag

## P0 — Security
- [ ] Define documented configurable PDF size, page and extraction limits
- [ ] Reject invalid signature, malformed, encrypted/unsupported, empty-text and oversized PDFs with safe feedback
- [ ] Replace user-visible raw exception details with sanitized messages
- [ ] Ensure diagnostics and tests never expose GROQ_API_KEY, PDF text or sensitive payloads

## P1 — Integration and CI
- [ ] Create reproducible dependency lock or constraints from tested Python environment
- [ ] Add valid and malformed synthetic PDF fixtures
- [ ] Add tests for actual PDF parsing, embeddings creation, FAISS search and deterministic retrieval
- [ ] Add mocked end-to-end question/answer test without real Groq credentials
- [ ] Add cross-session/document replacement and temporary cleanup failure tests
- [ ] Configure CI on pull requests and main branch and publish pytest results
- [ ] Record environment/version, command, results and known limitations

## P1 — Source grounding and RAG quality
- [ ] Preserve doc ID, chunk metadata, page and retrieved excerpt in references
- [ ] Render usable source references without calling retrieved pages verified claims
- [ ] Implement explicit no-evidence fallback
- [ ] Create small golden QA dataset with source-page expectations and no-answer cases
- [ ] Measure retrieval hit/recall at K and manually assess answer groundedness; report observed results only

## Governance
- [ ] Reconcile prior secure-and-refactor-rag checklist with verified work
- [ ] Update README with threat model, tests, limits and evidence format
- [ ] Open review PR with changes, tests and remaining risks; no automatic merge

**State:** proposal/tasks only, not implementation or test confirmation.