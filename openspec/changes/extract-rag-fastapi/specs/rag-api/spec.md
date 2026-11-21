# RAG API — requirements

## ADDED Requirements

### Requirement: Session creation and isolation
The API SHALL issue unpredictable session identifiers, store conversation/index state server-side for the local single-process MVP, and isolate data across sessions.

#### Scenario: Two different sessions
- **WHEN** two clients create sessions and upload distinct PDFs
- **THEN** each session's queries retrieve only its own PDF content.

#### Scenario: Reset and reupload
- **WHEN** a session is reset or its PDF is replaced
- **THEN** the previous index and conversation history are no longer available.

### Requirement: Secure PDF upload
The API SHALL retain existing upload validation and resource bounds, shall not trust filename or MIME alone, and SHALL return safe error messages.

#### Scenario: Invalid or oversized PDF
- **WHEN** a client uploads a malformed, encrypted, excessively large or empty-text PDF
- **THEN** ingestion is rejected safely and any temporary processing artifacts are cleaned up.

### Requirement: Question answering
The API SHALL accept questions only for sessions with a successfully indexed document, returning an answer and retrieved source-page metadata without claiming verified statement-level citations.

#### Scenario: No indexed document
- **WHEN** a session submits a question before uploading a valid PDF
- **THEN** the API returns a typed safe error without invoking Groq.

### Requirement: API observability and resilience
The API SHALL assign diagnostic request identifiers, hide provider exception details from clients and expose health endpoints that do not disclose secrets.

#### Scenario: LLM provider failure
- **WHEN** Groq is unavailable
- **THEN** the API returns a safe failure response and logs only sanitized diagnostic context.

### Requirement: Deterministic tests and compatibility
Automated tests SHALL verify FastAPI contracts and isolation using a mocked model, while existing Streamlit behavior and security tests remain passing.

#### Scenario: CI without provider secrets
- **WHEN** CI executes on a pull request without Groq credentials
- **THEN** all required unit and mocked integration tests execute without network access to Groq.
