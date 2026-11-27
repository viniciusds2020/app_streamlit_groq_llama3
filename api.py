"""HTTP adapter for the local RAG service. Run on localhost only.

Do not expose publicly without authentication, authorization and rate limits.
"""
import logging
import os
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingestion import PDFValidationError
from rag_service import DocumentNotIndexed, RAGService, ServiceBusy, SessionNotFound

logger = logging.getLogger(__name__)
app = FastAPI(title="PDF RAG API", version="0.2.0")
service = RAGService()


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def _detail(request_id, code, message):
    return {"code": code, "message": message, "request_id": request_id}


def _failure(exc, request_id):
    logger.error("RAG backend error request_id=%s type=%s", request_id, type(exc).__name__)
    return HTTPException(
        status_code=503,
        detail=_detail(request_id, "PROCESSING_FAILED", "Não foi possível concluir a solicitação."),
    )


@app.middleware("http")
async def add_request_id_and_limit_upload(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    response = None
    if request.method == "POST" and request.url.path.endswith("/documents"):
        try:
            maximum = int(os.getenv("RAG_MAX_PDF_BYTES", str(10 * 1024 * 1024)))
            content_length = int(request.headers.get("content-length", "0"))
        except (TypeError, ValueError):
            maximum, content_length = 0, 0

        # The extra allowance covers multipart headers and boundaries. The file
        # itself is still independently bounded in the endpoint below.
        if maximum <= 0:
            response = JSONResponse(
                status_code=503,
                content={"detail": _detail(request_id, "CONFIGURATION_ERROR", "Serviço indisponível.")},
            )
        elif content_length > maximum + 65536:
            response = JSONResponse(
                status_code=413,
                content={"detail": _detail(request_id, "PDF_TOO_LARGE", "O PDF excede o tamanho máximo permitido.")},
            )

    if response is None:
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    return {"status": "ready"}


@app.post("/api/v1/sessions", status_code=201)
def create_session(request: Request):
    try:
        return {"session_id": service.create_session(), "request_id": request.state.request_id}
    except Exception as exc:
        raise _failure(exc, request.state.request_id) from None


@app.delete("/api/v1/sessions/{session_id}", status_code=204)
def delete_session(request: Request, session_id: str):
    try:
        service.reset(session_id)
    except SessionNotFound:
        raise HTTPException(
            status_code=404,
            detail=_detail(request.state.request_id, "SESSION_NOT_FOUND", "Sessão não encontrada."),
        )


@app.post("/api/v1/sessions/{session_id}/documents", status_code=201)
def upload_document(request: Request, session_id: str, file: UploadFile = File(...)):
    try:
        maximum = int(os.getenv("RAG_MAX_PDF_BYTES", str(10 * 1024 * 1024)))
    except (TypeError, ValueError):
        maximum = 0
    if maximum <= 0:
        raise HTTPException(
            status_code=503,
            detail=_detail(request.state.request_id, "CONFIGURATION_ERROR", "Serviço indisponível."),
        )
    try:
        # Avoid reading an unbounded multipart upload into application memory.
        data = file.file.read(maximum + 1)
        if len(data) > maximum:
            raise HTTPException(
                status_code=413,
                detail=_detail(request.state.request_id, "PDF_TOO_LARGE", "O PDF excede o tamanho máximo permitido."),
            )
        document_id = service.upload(session_id, data)
        return {
            "document_id": document_id,
            "status": "indexed",
            "request_id": request.state.request_id,
        }
    except SessionNotFound:
        raise HTTPException(
            status_code=404,
            detail=_detail(request.state.request_id, "SESSION_NOT_FOUND", "Sessão não encontrada."),
        )
    except ServiceBusy:
        raise HTTPException(
            status_code=503,
            detail=_detail(request.state.request_id, "SERVICE_BUSY", "Serviço ocupado. Tente novamente em instantes."),
        )
    except PDFValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=_detail(request.state.request_id, "INVALID_PDF", str(exc)),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _failure(exc, request.state.request_id) from None
    finally:
        file.file.close()


@app.post("/api/v1/sessions/{session_id}/questions")
def answer_question(request: Request, session_id: str, query: Question):
    if not query.question.strip():
        raise HTTPException(
            status_code=422,
            detail=_detail(request.state.request_id, "EMPTY_QUESTION", "Informe uma pergunta."),
        )
    try:
        return service.ask(
            session_id,
            query.question.strip(),
            request_id=request.state.request_id,
        )
    except SessionNotFound:
        raise HTTPException(
            status_code=404,
            detail=_detail(request.state.request_id, "SESSION_NOT_FOUND", "Sessão não encontrada."),
        )
    except DocumentNotIndexed:
        raise HTTPException(
            status_code=409,
            detail=_detail(request.state.request_id, "DOCUMENT_NOT_INDEXED", "Envie um PDF antes de perguntar."),
        )
    except ServiceBusy:
        raise HTTPException(
            status_code=503,
            detail=_detail(request.state.request_id, "SERVICE_BUSY", "Serviço ocupado. Tente novamente em instantes."),
        )
    except Exception as exc:
        raise _failure(exc, request.state.request_id) from None
