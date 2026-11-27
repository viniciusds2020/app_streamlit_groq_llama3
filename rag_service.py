"""RAG orchestration with DuckDB metadata and persistent Qdrant collections.

Session identifiers act as bearer tokens. Tokens are hashed before persistence.
This local MVP is not appropriate for public hosting without authentication,
authorization, rate limits, and a multi-process storage design.
"""
import hashlib
import os
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from langchain.chains import ConversationalRetrievalChain

from ingestion import load_pdf
from llm import load_llm
from metadata_store import MetadataStore
from observability import trace_span
from retrieval import (
    collection_name_for,
    create_vector_store,
    delete_vector_store,
    load_vector_store,
)


class SessionNotFound(Exception):
    pass


class DocumentNotIndexed(Exception):
    pass


class ServiceBusy(Exception):
    """Raised when all bounded RAG operation slots are occupied."""


@dataclass
class Session:
    session_key: str
    document_id: str = None
    collection_name: str = None
    vector_store: object = None
    history: list = field(default_factory=list)
    touched: float = field(default_factory=time.time)
    lock: object = field(default_factory=threading.RLock)


class RAGService:
    """Single-process session cache backed by persistent DuckDB and Qdrant."""

    def __init__(
        self,
        *,
        ttl_seconds=None,
        max_sessions=None,
        max_active_operations=None,
        metadata_store=None,
    ):
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
        self._metadata = metadata_store or MetadataStore()

    @contextmanager
    def _operation_slot(self):
        if not self._operation_slots.acquire(blocking=False):
            raise ServiceBusy()
        try:
            yield
        finally:
            self._operation_slots.release()

    @staticmethod
    def _session_key(token):
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _expire(self):
        cutoff = time.time() - self.ttl_seconds
        expired = self._metadata.delete_expired(cutoff)
        for session_key, collection_name in expired:
            self._sessions.pop(session_key, None)
            delete_vector_store(collection_name)

    def create_session(self):
        with self._lock:
            self._expire()
            if self._metadata.count_sessions() >= self.max_sessions:
                raise RuntimeError("Session capacity reached")
            token = secrets.token_urlsafe(32)
            session_key = self._session_key(token)
            self._metadata.create_session(session_key)
            self._sessions[session_key] = Session(session_key=session_key)
            return token

    def _get(self, token):
        session_key = self._session_key(token)
        with self._lock:
            self._expire()
            session = self._sessions.get(session_key)
            if session is None:
                stored = self._metadata.get_session(session_key)
                if stored is None:
                    raise SessionNotFound()
                session = Session(
                    session_key=session_key,
                    document_id=stored["document_id"],
                    collection_name=stored["collection_name"],
                    history=[tuple(item) for item in stored["history"]],
                    touched=stored["touched"],
                )
                self._sessions[session_key] = session
            session.touched = time.time()
            self._metadata.touch(session_key, session.touched)
            return session

    def reset(self, token):
        session_key = self._session_key(token)
        with self._lock:
            self._expire()
            stored = self._metadata.get_session(session_key)
            if stored is None:
                raise SessionNotFound()
            collection_name = stored["collection_name"]
            self._metadata.delete_session(session_key)
            self._sessions.pop(session_key, None)
            delete_vector_store(collection_name)

    def upload(self, token, data):
        session = self._get(token)
        with self._operation_slot():
            with session.lock:
                from ingestion import validate_pdf_bytes

                validated = validate_pdf_bytes(data)
                docs = load_pdf(_UploadedBytes(validated))
                document_id = hashlib.sha256(validated).hexdigest()
                index_id = uuid.uuid4().hex
                collection_name = collection_name_for(session.session_key, index_id)

                # Build a new collection first. The old one stays available until
                # the new index and its metadata are committed successfully.
                store = create_vector_store(docs, collection_name)
                old_collection = session.collection_name
                try:
                    self._metadata.replace_document(
                        session.session_key,
                        document_id,
                        collection_name,
                    )
                except Exception:
                    delete_vector_store(collection_name)
                    raise

                session.document_id = document_id
                session.collection_name = collection_name
                session.vector_store = store
                session.history.clear()
                session.touched = time.time()
                if old_collection and old_collection != collection_name:
                    delete_vector_store(old_collection)
                return document_id

    def ask(self, token, question, *, request_id=None):
        session = self._get(token)
        with self._operation_slot():
            with session.lock:
                if not session.collection_name:
                    raise DocumentNotIndexed()
                store = session.vector_store or load_vector_store(session.collection_name)
                trace_metadata = {
                    "request_id": request_id,
                    "session_fingerprint": session.session_key[:16],
                    "question_length": len(question),
                }
                started = time.monotonic()
                with trace_span("rag.answer", metadata=trace_metadata) as span:
                    chain = ConversationalRetrievalChain.from_llm(
                        llm=load_llm(),
                        retriever=store.as_retriever(),
                        return_source_documents=True,
                    )
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
                    if span is not None:
                        span.update(metadata={
                            "duration_ms": round((time.monotonic() - started) * 1000),
                            "retrieved_page_count": len(pages),
                            "outcome": "success",
                        })
                session.history.append((question, answer))
                session.touched = time.time()
                self._metadata.save_history(session.session_key, session.history)
                return {
                    "answer": answer,
                    "pages": pages,
                    "document_id": session.document_id,
                    "request_id": request_id,
                }


class _UploadedBytes:
    def __init__(self, data):
        self._data = data

    def getvalue(self):
        return self._data
