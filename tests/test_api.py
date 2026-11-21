"""FastAPI contract tests with deterministic fake retrieval and no Groq secrets."""
import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

import api
import rag_service


def make_pdf(pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


@pytest.fixture
def client(monkeypatch):
    api.service = rag_service.RAGService(ttl_seconds=3600, max_sessions=10)

    def fake_load_pdf(upload):
        return [SimpleNamespace(page_content=str(len(upload.getvalue())), metadata={"page": 0})]

    class FakeStore:
        def __init__(self, docs):
            self.docs = docs

        def as_retriever(self):
            return self

    class FakeChain:
        @classmethod
        def from_llm(cls, llm, retriever, return_source_documents):
            instance = cls()
            instance.retriever = retriever
            return instance

        def invoke(self, request):
            return {
                "answer": "resposta " + self.retriever.docs[0].page_content,
                "source_documents": self.retriever.docs,
            }

    monkeypatch.setattr(rag_service, "load_pdf", fake_load_pdf)
    monkeypatch.setattr(rag_service, "create_vector_store", FakeStore)
    monkeypatch.setattr(rag_service, "load_llm", lambda: object())
    monkeypatch.setattr(rag_service, "ConversationalRetrievalChain", FakeChain)
    return TestClient(api.app)


def new_session(client):
    response = client.post("/api/v1/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def upload(client, session, data):
    return client.post(
        f"/api/v1/sessions/{session}/documents",
        files={"file": ("test.pdf", data, "application/pdf")},
    )


def ask(client, session, question="Qual conteúdo?"):
    return client.post(
        f"/api/v1/sessions/{session}/questions", json={"question": question},
    )


def test_health(client):
    assert client.get("/health/live").json()["status"] == "ok"
    assert client.get("/health/ready").json()["status"] == "ready"


def test_requires_index_before_answer(client):
    session = new_session(client)
    result = ask(client, session)
    assert result.status_code == 409
    assert result.json()["detail"]["code"] == "DOCUMENT_NOT_INDEXED"


def test_upload_query_and_isolate_two_sessions(client):
    first, second = new_session(client), new_session(client)
    assert upload(client, first, make_pdf(1)).status_code == 201
    assert upload(client, second, make_pdf(2)).status_code == 201
    result_a = ask(client, first)
    result_b = ask(client, second)
    assert result_a.status_code == 200
    assert result_b.status_code == 200
    assert result_a.json()["answer"] != result_b.json()["answer"]
    assert result_a.json()["pages"] == [1]


def test_reupload_invalidates_previous_index_and_history(client):
    session = new_session(client)
    assert upload(client, session, make_pdf()).status_code == 201
    assert ask(client, session).status_code == 200
    assert len(api.service._get(session).history) == 1
    invalid = upload(client, session, b"not a pdf")
    assert invalid.status_code == 422
    assert api.service._get(session).history == []
    assert ask(client, session).status_code == 409


def test_deleted_session_is_unavailable(client):
    session = new_session(client)
    assert client.delete(f"/api/v1/sessions/{session}").status_code == 204
    assert ask(client, session).status_code == 404


def test_oversized_upload_rejected(client, monkeypatch):
    session = new_session(client)
    monkeypatch.setenv("RAG_MAX_PDF_BYTES", "5")
    result = upload(client, session, b"%PDF-too-large")
    assert result.status_code == 413


def test_provider_error_does_not_leak(client, monkeypatch):
    session = new_session(client)
    assert upload(client, session, make_pdf()).status_code == 201

    class BrokenChain:
        @classmethod
        def from_llm(cls, **kwargs):
            raise RuntimeError("SECRET-PRIVATE-GROQ-TOKEN")

    monkeypatch.setattr(rag_service, "ConversationalRetrievalChain", BrokenChain)
    result = ask(client, session)
    assert result.status_code == 503
    assert "SECRET" not in result.text
    assert result.json()["detail"]["request_id"]


def test_blank_question_is_rejected(client):
    session = new_session(client)
    assert ask(client, session, "   ").status_code == 422
