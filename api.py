"""HTTP adapter for the local RAG service. Run on localhost only.

Do not expose publicly without authentication, authorization and rate limits.
"""
import logging
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ingestion import PDFValidationError
from rag_service import DocumentNotIndexed, RAGService, SessionNotFound

logger = logging.getLogger(__name__)
app = FastAPI(title="PDF RAG API", version="0.1.0")
service = RAGService()


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def _incident():
    return uuid.uuid4().hex[:12]


def _failure(exc):
    incident_id = _incident()
    logger.error("RAG backend error incident=%s type=%s", incident_id, type(exc).__name__)
    return HTTPException(
        status_code=503,
        detail={"code": "PROCESSING_FAILED", "message": "Não foi possível concluir a solicitação.", "request_id": incident_id},
    )


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    return {"status": "ready"}


@app.post("/api/v1/sessions", status_code=201)
def create_session():
    try:
        return {"session_id": service.create_session()}
    except Exception as exc:
        raise _failure(exc) from None


@app.delete("/api/v1/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    try:
        service.reset(session_id)
    except SessionNotFound:
        raise HTTPException(status_code=404, detail={"code": "SESSION_NOT_FOUND", "message": "Sessão não encontrada."})


@app.post("/api/v1/sessions/{session_id}/documents", status_code=201)
def upload_document(session_id: str, file: UploadFile = File(...)):
    maximum = int(os.getenv("RAG_MAX_PDF_BYTES", str(10 * 1024 * 1024)))
    if maximum <= 0:
        raise HTTPException(status_code=503, detail={"code": "CONFIGURATION_ERROR", "message": "Serviço indisponível."})
    try:
        # Avoid reading an unbounded multipart upload into application memory.
        data = file.file.read(maximum + 1)
        if len(data) > maximum:
            raise HTTPException(
                status_code=413,
                detail={"code": "PDF_TOO_LARGE", "message": "O PDF excede o tamanho máximo permitido."},
            )
        document_id = service.upload(session_id, data)
        return {"document_id": document_id, "status": "indexed"}
    except SessionNotFound:
        raise HTTPException(status_code=404, detail={"code": "SESSION_NOT_FOUND", "message": "Sessão não encontrada."})
    except PDFValidationError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PDF", "message": str(exc)})
    except HTTPException:
        raise
    except Exception as exc:
        raise _failure(exc) from None
    finally:
        file.file.close()


@app.post("/api/v1/sessions/{session_id}/questions")
def answer_question(session_id: str, query: Question):
    if not query.question.strip():
        raise HTTPException(status_code=422, detail={"code": "EMPTY_QUESTION", "message": "Informe uma pergunta."})
    try:
        result = service.ask(session_id, query.question.strip())
        return result
    except SessionNotFound:
        raise HTTPException(status_code=404, detail={"code": "SESSION_NOT_FOUND", "message": "Sessão não encontrada."})
    except DocumentNotIndexed:
        raise HTTPException(status_code=409, detail={"code": "DOCUMENT_NOT_INDEXED", "message": "Envie um PDF antes de perguntar."})
    except Exception as exc:
        raise _failure(exc) from None
