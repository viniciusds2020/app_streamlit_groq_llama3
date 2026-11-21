# Security requirements

## Requirement SEC-001 — Validate uploaded PDFs
The system MUST validate uploaded PDF bytes before ingestion, with configurable resource limits and parser-level rejection of invalid or unsupported documents.

### Scenario: malformed input
**Given** a file named `example.pdf` containing invalid bytes
**When** the user uploads it
**Then** the app MUST reject it with a non-sensitive error and MUST NOT build an index.

### Scenario: oversized input
**Given** a PDF above the configured byte limit
**When** uploaded
**Then** it MUST be rejected before parsing or embedding.

### Scenario: no extractable text
**Given** a scan-only PDF lacking text extraction support
**When** parsed
**Then** the system MUST explain that no extractable text was found, without falsely claiming OCR.

## Requirement SEC-002 — Safe error handling
The UI MUST NOT render raw exception strings or secret-bearing provider payloads. Logs MUST omit API secrets and document text.

### Scenario: Groq API exception
**When** a provider call fails
**Then** the UI shows a generic actionable error while safe logs retain diagnostic correlation data.
