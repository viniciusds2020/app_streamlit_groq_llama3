"""RAG orchestration and bounded, process-local sessions for the local MVP.

Session identifiers act as bearer tokens. This in-memory store is NOT appropriate
for a public-facing or multi-worker deployment without proper authentication,
authorization, distributed storage, and rate limiting.
"""
import hashlib
import logging
import os
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from langchain.chains import ConversationalRetrievalChain

from ingestion import load_pdf
from observability import trace_span
from retrieval import create_vector_store
from llm import load_llm

log = logging.getLogger(__name__)


class SessionNotFound(Exception):
    pass


class DocumentNotIndexed(Exception):
    pass


class ServiceBusy(Exception):
    """Raised when all bounded RAG operation slots are occupied."""


@dataclass
class Session:
    vector_store: object = None
    document_id: str = None
    history: list = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=threading.RLock)


class RAGService:
    """Single-worker, in-memory session store with idle expiry and capacity limits."""

    def __init__(self, *, ttl_seconds=None, max_sessions=None, max_active_operations=None):
        self.ttl_seconds = ttl_seconds or int(os.getenv("RAG_SESSION_TTL_SECONDS", "3600"))
        self.max_sessions = max_sessions or int(os.getenv("RAG_MAX_SESSIONS", "100"))
        self.max_active_operations = max_active_operations or int(
            os.getenv("RAG_MAX_CONCURRENT_OPERATIONS", "2")
        )
        if self.ttl_seconds < 1 or self.max_sessions < 1 or self.max_active_operations < 1:
            raise ValueError("Session and operation limits must be positive")
        self._sessions = {}
        self._lock = threading.RLock()
        self._operation_slots = threading.BoundedSemaphore(self.max_active_operations)

    @contextmanager
    def _operation_slot(self):
        if not self._operation_slots.acquire(blocking=False):
            raise ServiceBusy()
        try:
            yield
        finally:
            self._operation_slots.release()

    def _expire(self):
        now = time.monotonic()
        for token, session in list(self._sessions.items()):
            if now - session.touched >= self.ttl_seconds:
                del self._sessions[token]

    def create_session(self):
        with self._lock:
            self._expire()
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("Session capacity reached")
            token = secrets.token_urlsafe(32)
            self._sessions[token] = Session()
            return token

    def _get(self, token):
        with self._lock:
            self._expire()
            session = self._sessions.get(token)
            if session is None:
                raise SessionNotFound()
            session.touched = time.monotonic()
            return session

    def reset(self, token):
        with self._lock:
            self._expire()
            if token not in self._sessions:
                raise SessionNotFound()
            del self._sessions[token]

    def upload(self, token, data):
        """Validate, extract, index, and atomically replace a session's document."""
        session = self._get(token)
        with self._operation_slot():
            with session.lock:
                # New upload invalidates the previous document and chat history.
                session.vector_store = None
                session.document_id = None
                session.history.clear()
                from ingestion import validate_pdf_bytes

                validated = validate_pdf_bytes(data)

                class UploadedBytes:
                    def getvalue(self):
                        return validated

                docs = load_pdf(UploadedBytes())
                store = create_vector_store(docs)
                session.vector_store = store
                session.document_id = hashlib.sha256(validated).hexdigest()
                return session.document_id

    def ask(self, token, question, *, request_id=None):
        session = self._get(token)
        with self._operation_slot():
            with session.lock:
                if session.vector_store is None:
                    raise DocumentNotIndexed()
                chain = ConversationalRetrievalChain.from_llm(
                    llm=load_llm(),
                    retriever=session.vector_store.as_retriever(),
                    return_source_documents=True,
                )
                trace_metadata = {
                    "request_id": request_id,
                    "session_fingerprint": hashlib.sha256(token.encode("utf-8")).hexdigest()[:16],
                    "question_length": len(question),
                }
                started = time.monotonic()
                with trace_span("rag.answer", metadata=trace_metadata) as span:
                    result = chain.invoke({
                        "question": question,
                        "chat_history": list(session.history),
                    })
                answer = str(result["answer"])
                pages = sorted({
                    doc.metadata["page"] + 1
                    for doc in result.get("source_documents", [])
                    if isinstance(doc.metadata.get("page"), int) and doc.metadata["page"] >= 0
                })
                session.history.append((question, answer))
                if span is not None:
                    span.update(metadata={
                        "duration_ms": round((time.monotonic() - started) * 1000),
                        "retrieved_page_count": len(pages),
                        "outcome": "success",
                    })
                return {
                    "answer": answer,
                    "pages": pages,
                    "document_id": session.document_id,
                    "request_id": request_id,
                }
