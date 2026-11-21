# Integration and quality requirements

## Requirement INT-001 — Reproducible local pipeline
A clean supported Python environment MUST install declared dependencies and execute PDF parsing, embeddings, FAISS indexing/search and deterministic mocked generation.

### Scenario: valid PDF
**Given** a known small PDF fixture
**When** the pipeline indexes and queries it
**Then** the expected page/substring MUST appear among retrieved results.

## Requirement INT-002 — Isolation and cleanup
Distinct document uploads and sessions MUST NOT share conversation history or vector index. Temporary files MUST be deleted on success and exceptions.

### Scenario: replace PDF
**Given** indexed PDF A and a conversation
**When** PDF B is uploaded
**Then** old history and index MUST be discarded and queries MUST use PDF B only.

## Requirement INT-003 — CI verification
CI MUST run tests on pull requests and pushes to main and report results; no live Groq key is required. Unrun tests MUST NOT be reported as passing.
