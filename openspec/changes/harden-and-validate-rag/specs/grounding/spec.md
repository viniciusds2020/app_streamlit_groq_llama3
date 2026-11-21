# Source grounding and evaluation requirements

## Requirement RAG-001 — Auditable source references
Each response based on retrieval SHOULD expose document identity, page and relevant excerpt/chunk reference. The app MUST distinguish retrieved source references from independently verified per-claim support.

### Scenario: supported question
**Given** a question whose answer exists in a known PDF passage
**When** answered
**Then** the response includes a source reference pointing to that passage and page.

## Requirement RAG-002 — Abstain on missing evidence
The assistant MUST indicate that the document does not contain sufficient information when retrieval cannot support an answer; it MUST NOT invent citations.

### Scenario: unanswerable question
**Given** a fixture without a requested fact
**When** asked for the fact
**Then** the response indicates insufficient document evidence.

## Requirement RAG-003 — Baseline evaluation
A versioned golden question set SHOULD cover positive, negative and ambiguous examples. Results MUST include reproducible retrieval metrics and documented qualitative groundedness checks, not invented accuracy claims.
